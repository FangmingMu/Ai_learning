import json

from langchain_openai import ChatOpenAI
import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain.chains import HypotheticalDocumentEmbedder
from R1_Evaluation_Framework.ragas_eval import Test
from local_model import get_embedding_model,get_llm

llm = get_llm()
embedding = get_embedding_model()

# llm = ChatOpenAI(
#         model_name="qwen-plus-2025-04-28",
#         api_key=os.getenv("DASHSCOPE_API_KEY"),
#         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#         extra_body={"enable_thinking": False},
# )
#
# embedding = DashScopeEmbeddings(
#             model = "text-embedding-v1",
#             dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
#            )

hyde_prompt_template = """
请根据以下问题，撰写一篇简洁、清晰、事实丰富的段落来回答它。
这篇段落将用于后续的信息检索。

问题: {question}
回答段落:
"""

hyde_prompt = ChatPromptTemplate.from_template(hyde_prompt_template)  # 根据一个字符串模板来创建 PromptTemplate 对象。
hyde_chain = hyde_prompt | llm | StrOutputParser()

current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)  # 返回上一级目录

db_name = 'chroma_db'
db_path = os.path.join(parent_path, 'R1_Evaluation_Framework', db_name)   #错误  现在不是向量数据库
vector_db = Chroma(persist_directory=db_path, embedding_function=embedding)

question_list_name = 'golden_dataset.jsonl'
question_list = os.path.join(parent_path, 'R1_Evaluation_Framework', question_list_name)

# 通过提问 把提问给llm让他生成答案   请你假装已经找到了完美的答案，然后写出一篇最能回答这个问题的、详尽的、假设性的文档
# 生成假设性文档 向量化  检索
def manual_dyde_retrieval(question:str, vector_db=vector_db):
    hyde_doxs = hyde_chain.invoke({"question":question})

    hyde_embedding = embedding.embed_query(hyde_doxs)   # embed_query()  对单条文本进行编码，返回该文本的嵌入向量。

    retrieved_docs = vector_db.similarity_search_by_vector(
        embedding=hyde_embedding,
        k=10
    )    # 是ChromaDB等库支持的底层方法

    return retrieved_docs



def langchain_hyde(question: str=None, base_retriever=None, llm=llm, embeddings=None):
    embeddings = embeddings or embedding
    # 创建一个HyDE嵌入器  它会自动处理“生成假设文档 -> 向量化”这个过程
    hyde_embedder = HypotheticalDocumentEmbedder.from_llm(
        llm=llm,
        base_embeddings=embeddings,
        prompt_key="web_search",  # LangChain内置不同场景的prompt, 'web_search'模拟 Web 搜索类的具体回答风格
        #prompt_key=hyde_prompt_template    # 也可以自定义
    )
    base_retriever = base_retriever or vector_db.as_retriever(search_type="similarity", search_kwargs={"k":10})   # 兜底赋值  如果传入  用传入的  没有就用后面的

    # 使用HyDE嵌入器来替换常规的嵌入器进行检索  这里我们直接用向量进行搜索，更接近底层
    hypothetical_embedding = hyde_embedder.embed_query(question)
    # base_retriever Retriever 对象  vectorstore  向量数据库对象
    # 高层 API   base_retriever.get_relevant_documents(query)   自动调用 embedding → 再去向量库里搜索
    # 底层 API   vectorstore.similarity_search_by_vector(vector, k=4)
    retrieved_docs = base_retriever.vectorstore.similarity_search_by_vector(
        embedding=hypothetical_embedding
    )
    # print("\n--- 内置HyDE检索到的最终文档 ---")
    # for i, doc in enumerate(retrieved_docs):
    #     print(f"  文档 {i + 1}: '{doc.page_content}'")

    return retrieved_docs


if __name__ == "__main__":
    result_dir = 'hyde_redults.jsonl'
    test = Test(result_dir)
    all_results=[]
    print("读取问题文档")
    with open(question_list, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            item = json.loads(line)
            question = item["question"]
            print(f"正在检索第{i}个问题：{question}")
            # retrieval_docs = langchain_hyde(question=question)
            retrieval_docs = manual_dyde_retrieval(question)
            question_dict = test.generate_answer(question, retrieval_docs)
            all_results.append(question_dict)
    print("正在写入")
    with open(result_dir, 'w', encoding='utf-8')as f:
        for result_item in all_results:
            f.write(result_item + '\n')
    print("开始评估")
    result = test.ragas_evaluate(result_dir)
    print(result)

