import json

from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import LineListOutputParser #把LLM生成的、以换行符分隔的字符串，直接转换成一个字符串列表
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings


def create_query(original_question:str):
    llm = ChatOpenAI(
        model_name = "qwen-plus-2025-04-28",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        extra_body={"enable_thinking": False},
    )

    expansion_prompt_template = """
    你是一个精通信息检索的AI助手。
    你的任务是：根据用户提出的一个原始问题，生成3个不同角度的、更具体、更丰富的相关问题。
    这些生成的问题将用于并行检索，以提高召"回率。
    
    请确保生成的问题与原始问题紧密相关，但又能覆盖不同的方面或关键词。
    以换行符分隔每个生成的问题，不要包含任何编号或前缀。
    
    原始问题:
    {question}
    
    生成的相关问题:
    """
    # 生成重写查询模板的对象
    expansion_prompt = ChatPromptTemplate.from_template(expansion_prompt_template)

    query_expansion_chain = expansion_prompt | llm | LineListOutputParser() # 重新查询的llm链

    expanded_queries = query_expansion_chain.invoke({"question":original_question})

    all_queries = [original_question] + expanded_queries  # 把原始问题加在最前面

    return all_queries


def get_expanded_retrieved_contexts(retriever, all_queries):
    all_retrieved_docs = []
    for query in all_queries:
        # 先检索（不回答）  .get_relevant_documents()   专门用于 Retriever检索器  返回相关文档
        # 问+答，一步到位，.invoke()  用于 Runnable / Chain / Agent   返回链的结果  答案 相关文档
        retrieved_docs=retriever.get_relevant_documents(query)    # 和retriever.invoke(query) 一样
        all_retrieved_docs.extend(retrieved_docs)  # 把检索的文档一个个放入进去，组成大的相关列表  用extend  每个是Document对象

        # 字典推导式    key 唯一，所以如果多个文档的内容相同，只会保留最后一个
        unique_docs = {doc.page_content: doc for doc in all_retrieved_docs}.values()

        return list(unique_docs)

def create_related_josnl(question_list, retriever):
    with open(question_list, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads()
            question = item['question']
            relate_question = create_query(question)
            get_expanded_retrieved_contexts()


    llm = ChatOpenAI(
        model_name="qwen-plus-2025-04-28",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        extra_body={"enable_thinking": False},
    )
















