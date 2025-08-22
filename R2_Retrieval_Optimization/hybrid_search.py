import json
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from R1_Evaluation_Framework.ragas_eval import Test
from local_model import get_embedding_model, get_llm

llm = get_llm()
embedding = get_embedding_model()

current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)  # 返回上一级目录
# embedding = DashScopeEmbeddings(
#     model="text-embedding-v1",
#     dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
# )
db_name = 'chroma_db'
db_path = os.path.join(parent_path, 'R1_Evaluation_Framework', db_name)  # 错误  现在不是向量数据库
vector_db = Chroma(persist_directory=db_path, embedding_function=embedding)

question_list_name = 'golden_dataset.jsonl'
question_list = os.path.join(parent_path, 'R1_Evaluation_Framework', question_list_name)


class Hybrid_search():
    # 现在改为接收外部已经准备好的 "split_docs" 和 "retriever_vector" 作为参数。
    # 这样，重量级的操作（加载、切分、初始化向量检索器）就只需要在程序启动时执行一次。
    def __init__(self, split_docs, retriever_vector):
        self.split_docs = split_docs
        self.retriever_vector = retriever_vector  # 将高效的、持久化的向量检索器保存起来

        # BM25的初始化仍然放在这里，因为它依赖于split_docs，并且计算很快
        corpus_tokenized = [doc.page_content.split() for doc in self.split_docs]
        self.bm25 = BM25Okapi(corpus_tokenized)

    def bm25_retrieved(self, query=None, k=10):
        # bm25检索器
        # BM25返回的是原始文本块，我们需要找到对应的Document对象     range(len(split_docs))  生成文本块长度的序列 占位  就返回索引
        # 传给 BM25 一个“占位列表”，让它返回索引而不是原始文本。
        tokenized_query = query.split()
        bm25_doc_indices = self.bm25.get_top_n(tokenized_query, range(len(self.split_docs)), n=k)
        bm25_retrieved_docs = [self.split_docs[i] for i in bm25_doc_indices]

        return bm25_retrieved_docs

    ### 修改点 2: 修正 embeddings_retrieved 函数 ###
    # 这是之前代码中最核心的错误所在
    def embeddings_retrieved(self, query=None):
        # 向量检索器
        # embeddings = embedding # 您的注释
        # embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")) # 您的注释

        # ---------------------- 以下是您原来的错误逻辑，我们将其注释掉 ----------------------
        # vector_store = Chroma.from_documents(self.split_docs, embeddings)
        # retriever_vector = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":k}) # 让它返回Topk
        # ---------------------------------------------------------------------------------
        # 上面两行代码的错误在于：它为每一个问题都重新计算所有文档的向量，并在内存中创建一个全新的临时数据库。
        # 这是导致程序极慢且分数很低的主要原因。

        # 正确的逻辑：直接使用我们在 __init__ 中保存好的、基于持久化存储的高效检索器。
        vector_retrieved_docs = self.retriever_vector.invoke(query)

        return vector_retrieved_docs

    def reciprocal_rank_fusion(self, retrieved_results: list[list], k: int = 60) -> list:
        """
           使用倒数排名融合算法，对多个检索结果列表进行融合和重排序。
           Args:
               retrieved_results (list[list]): 一个包含多个检索结果列表的列表。 例如: [[doc1, doc2], [doc2, doc3]]
               k (int): RRF算法中的常数k。
           Returns:
               list: 一个融合、去重并按RRF分数重排后的文档列表。
           """
        fused_scores = {}  # 存储分数      key: doc.page_content (用于唯一标识文档)    value: RRF score

        for docs_list in retrieved_results:  # 遍历的是  不同检索器的检索结果
            for rank, doc in enumerate(docs_list, start=1):  # 遍历检索器内的结果
                if doc.page_content not in fused_scores:
                    fused_scores[doc.page_content] = 0

                fused_scores[doc.page_content] += 1 / (k + rank)

        reranked_results = sorted(
            fused_scores.items(),  # .items()会返回一个可迭代对象，每个元素是 (key, value) 的元组   排序的单位
            key=lambda item: item[1],  # 表示取 (doc, score) 的 score 来排序。  # lambda匿名函数 lambda 参数: 表达式  返回表达式的值
            reverse=True  # 降序排序
        )

        # 重新构建文档对象列表   需要一个方法从page_content找回原始的Document对象    先创建一个查找表
        # doc.page_content → 键      doc → 值（Document 对象）
        doc_map = {doc.page_content: doc for doc_list in retrieved_results for doc in doc_list}
        # (content, score) 是一个元组  通过元组的第一个匹配document对象
        final_reranked_docs = [doc_map[content] for content, score in reranked_results]

        return final_reranked_docs


if __name__ == "__main__":
    # 步骤 1: 一次性加载和切分所有文档 (原本在 __init__ 中)
    print("--- 步骤 1: 正在加载和切分文档 (仅执行一次) ---")
    pdf_path = os.path.join(parent_path, 'R1_Evaluation_Framework/PDF/ARES RAG Evaluation.pdf')
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    # 步骤 2: 基于持久化数据库，一次性创建高效的向量检索器
    print("--- 步骤 2: 正在初始化持久化向量检索器 (仅执行一次) ---")
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    # 步骤 3: 实例化 Hybrid_search，并传入准备好的材料
    print("--- 步骤 3: 正在实例化混合搜索模块 ---")
    hybrid_search = Hybrid_search(split_docs=split_docs, retriever_vector=vector_retriever)

    # 步骤 4: 开始循环处理问题，此时所有准备工作都已完成
    print("--- 步骤 4: 开始循环处理所有问题 ---")
    all_results = []
    resultdir = 'Hybrid.jsonl'
    test = Test(resultdir)
    with open(question_list, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            item = json.loads(line)
            question = item["question"]
            print(f"正在检索第{i}个问题：{question}")

            # 调用现在高效且正确的检索方法
            bm25_retrieved_docs = hybrid_search.bm25_retrieved(question, k=10)
            vector_retrieved_docs = hybrid_search.embeddings_retrieved(question)

            # 将两个检索结果放入一个列表中
            # 使用.extend()会破坏掉原始的排名信息。
            # 而RRF算法的核心，恰恰就是依赖于每个文档在各自检索结果中的原始排名来计算分数的。所以，.extend()的方法在这里是完全错误的。
            all_retrieval_results = [bm25_retrieved_docs, vector_retrieved_docs]

            # 调用RRF函数
            fused_docs = hybrid_search.reciprocal_rank_fusion(all_retrieval_results)

            # print("\n--- 融合排序 (RRF) 后的最终结果 ---")
            # for i, doc in enumerate(fused_docs):
            #     print(f"  Rank {i + 1}: {doc.page_content[:100]}...")

            question_dict = test.generate_answer(question, fused_docs)
            all_results.append(question_dict)

    print("正在写入")
    with open(resultdir, 'w', encoding='utf-8') as f:
        for result_item in all_results:
            f.write(result_item + '\n')

    print("开始评估")
    result = test.ragas_evaluate(resultdir)
    print(result)