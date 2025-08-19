import json

from langchain_openai import ChatOpenAI
import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain.chains import HypotheticalDocumentEmbedder



llm = ChatOpenAI(
        model_name="qwen-plus-2025-04-28",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        extra_body={"enable_thinking": False},
)




embedding = DashScopeEmbeddings(
            model = "text-embedding-v1",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
           )


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
def manual_dyde_retrieval(question:str, vector_db):
    print("生成假设性文档")
    hyde_doxs = hyde_chain.invoke({"question":question})

    print("向量化")
    hyde_embedding = embedding.embed_query(hyde_doxs)   # embed_query()  对单条文本进行编码，返回该文本的嵌入向量。

    print("检索中")
    retrieved_docs = vector_db.similarity_search_by_vector(
        embedding=hyde_embedding,
        k=3
    )    # 是ChromaDB等库支持的底层方法

    return retrieved_docs


def own_hyde():
    with open(question_list, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            question = item["question"]

            result = manual_dyde_retrieval(question, vector_db)

            print("\n--- 手写HyDE检索到的最终文档 ---")
            for i, doc in enumerate(result):
                print(f"  文档 {i + 1}: '{doc.page_content}'")
            break
    return


def langchain_hyde(question: str=None, base_retriever=None, llm=llm, embeddings=None):
    embeddings = embeddings or embedding
    # 创建一个HyDE嵌入器  它会自动处理“生成假设文档 -> 向量化”这个过程
    hyde_embedder = HypotheticalDocumentEmbedder.from_llm(
        llm=llm,
        base_embeddings=embeddings,
        prompt_key="web_search",  # LangChain内置不同场景的prompt, 'web_search'模拟 Web 搜索类的具体回答风格
        #prompt_key=hyde_prompt_template    # 也可以自定义
    )
    base_retriever = base_retriever or vector_db.as_retriever()   # 兜底赋值  如果传入  用传入的  没有就用后面的

    # 使用HyDE嵌入器来替换常规的嵌入器进行检索  这里我们直接用向量进行搜索，更接近底层
    hypothetical_embedding = hyde_embedder.embed_query(question)
    # base_retriever Retriever 对象  vectorstore  向量数据库对象
    # 高层 API   base_retriever.get_relevant_documents(query)   自动调用 embedding → 再去向量库里搜索
    # 底层 API   vectorstore.similarity_search_by_vector(vector, k=4)
    retrieved_docs = base_retriever.vectorstore.similarity_search_by_vector(
        embedding=hypothetical_embedding
    )
    print("\n--- 内置HyDE检索到的最终文档 ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"  文档 {i + 1}: '{doc.page_content}'")

    return retrieved_docs


if __name__ == "__main__":
    with open(question_list, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            question = item["question"]
            retrieval_docs = langchain_hyde(question=question)

            answer_dict = {"question": question,
                           "ground_truth_contexts": item['ground_truth_contexts'],
                           "ground_truth_answer": item['ground_truth_answer'],
                           "retrieved_contexts": retrieval_docs,
                           "generated_answer": generated_answer}