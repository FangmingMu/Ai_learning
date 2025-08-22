import json
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from R1_Evaluation_Framework.ragas_eval import Test
from local_model import get_embedding_model,get_llm
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
    def __init__(self):
        current_path = os.path.dirname(__file__)
        PDF = os.path.join(os.path.dirname(current_path), 'R1_Evaluation_Framework/PDF/ARES RAG Evaluation.pdf')
        loader = PyPDFLoader(PDF)
        self.documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.split_docs = text_splitter.split_documents(self.documents)

        corpus_tokenized = [doc.page_content.split() for doc in self.split_docs]
        self.bm25 = BM25Okapi(corpus_tokenized)


    def bm25_retrieved(self,query=None, k=5):
        # bm25检索器

        # BM25返回的是原始文本块，我们需要找到对应的Document对象     range(len(split_docs))  生成文本块长度的序列 占位  就返回索引
        # 传给 BM25 一个“占位列表”，让它返回索引而不是原始文本。
        tokenized_query = query.split()
        bm25_doc_indices = self.bm25.get_top_n(tokenized_query, range(len(self.split_docs)), n=k)
        bm25_retrieved_docs = [self.split_docs[i] for i in bm25_doc_indices]

        return bm25_retrieved_docs


    def embeddings_retrieved(self,query=None, k=5):
        # 向量检索器
        embeddings = embedding
        # embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))
        vector_store = Chroma.from_documents(self.split_docs, embeddings)
        retriever_vector = vector_store.as_retriever(search_kwargs={"k": k}) # 让它返回Top 5

        vector_retrieved_docs = retriever_vector.invoke(query)

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

        for docs_list in retrieved_results:   # 遍历的是  不同检索器的检索结果
            for rank, doc in enumerate(docs_list, start=1):   # 遍历检索器内的结果
                if doc.page_content not in fused_scores:
                    fused_scores[doc.page_content] = 0

                fused_scores[doc.page_content] += 1 / (k + rank)

        reranked_results = sorted(
            fused_scores.items(),    # .items()会返回一个可迭代对象，每个元素是 (key, value) 的元组   排序的单位
            key=lambda item: item[1],    # 表示取 (doc, score) 的 score 来排序。  # lambda匿名函数 lambda 参数: 表达式  返回表达式的值
            reverse=True    # 降序排序
        )

        # 重新构建文档对象列表   需要一个方法从page_content找回原始的Document对象    先创建一个查找表
        # doc.page_content → 键      doc → 值（Document 对象）
        doc_map = {doc.page_content: doc for doc_list in retrieved_results for doc in doc_list}
        # (content, score) 是一个元组  通过元组的第一个匹配document对象
        final_reranked_docs = [doc_map[content] for content, score in reranked_results]

        return final_reranked_docs


if __name__ == "__main__":
    all_results=[]
    resultdir = 'Hybrid.jsonl'
    test = Test(resultdir)
    hybrid_search = Hybrid_search()
    with open(question_list, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            item = json.loads(line)
            question = item["question"]
            print(f"正在检索第{i}个问题：{question}")
            bm25_retrieved_docs = hybrid_search.bm25_retrieved(question)
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
