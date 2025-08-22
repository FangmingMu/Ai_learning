import json
from fileinput import filename
from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import LineListOutputParser # 把LLM生成的、以换行符分隔的字符串，直接转换成一个字符串列表
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from R1_Evaluation_Framework.ragas_eval import Test
from local_model import get_embedding_model,get_llm

llm = get_llm()
embedding = get_embedding_model()




DBPATH = '../R1_Evaluation_Framework/chroma_db'
vector_db = Chroma(
            persist_directory=DBPATH,
            embedding_function=embedding,
        )
# 定义模型路径

# llm = ChatOpenAI(
#         model_name = "qwen-plus-2025-04-28",
#         api_key=os.getenv("DASHSCOPE_API_KEY"),
#         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#         extra_body={"enable_thinking": False},
#     )
#
# embedding = DashScopeEmbeddings(
#         model="text-embedding-v1",
#         dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
#     )

def create_query(original_question:str):
    print("正在创建查询集")


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
    print("正在检索上下文")
    all_retrieved_docs = []
    for query in all_queries:
        # 先检索（不回答）  .get_relevant_documents()   专门用于 Retriever检索器  返回相关文档
        # 问+答，一步到位，.invoke()  用于 Runnable / Chain / Agent   返回链的结果  答案 相关文档
        retrieved_docs=retriever.get_relevant_documents(query)    # 和retriever.invoke(query) 一样
        # 字典推导式    key 唯一，所以如果多个文档的内容相同，只会保留最后一个
        unique_docs = {doc.page_content: doc for doc in all_retrieved_docs}.values()
    all_retrieved_docs.extend(retrieved_docs)  # 把检索的文档一个个放入进去，组成大的相关列表  用extend  每个是Document对象
    return list(unique_docs)


def generate_final_answer(llm, retrieved_docs:list, question:str)->str:
    print("生成答案中")
    # 列表解析（List Comprehension）  在列表的每两个元素之间插入指定的分隔符 \n\n---\n\n
    context_string = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    answer_generation_prompt_template = """
        你是一个严谨的问答机器人。
        你的任务是【只使用】下面提供的【上下文】来回答【原始问题】。
        确保答案简洁、准确，并且完全基于所提供的资料。
        如果上下文信息不足以回答，就明确指出“根据提供的资料无法回答”。

        上下文:
        {context}

        原始问题:
        {question}

        答案:
        """
    answer_prompt = ChatPromptTemplate.from_template(answer_generation_prompt_template)
    chain = answer_prompt | llm | StrOutputParser()
    generated_answer = chain.invoke({
        "context": context_string,
        "question": question
    })

    return generated_answer


def create_related_josnl(question_list):
    all_results=[]

    retrieval = vector_db.as_retriever()

    with open(question_list, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            question = item['question']
            all_queries = create_query(question)
            retrieved_docs = get_expanded_retrieved_contexts(retrieval, all_queries)
            answer = generate_final_answer(llm, retrieved_docs, question)

            answer_dict = {"question": question,
                           "ground_truth_contexts": item['ground_truth_contexts'],
                           "ground_truth_answer": item['ground_truth_answer'],
                           "retrieved_contexts": [doc.page_content for doc in retrieved_docs],
                           "generated_answer": answer}

            json_string = json.dumps(answer_dict, ensure_ascii=False)  # 字典转json  ensure_ascii=False 确保中文正确写入，不转换成编码

            all_results.append(json_string)
    print("正在写入")
    run_results = 'run_results.jsonl'
    with open(run_results, 'w', encoding='utf-8') as f:
        for result_item in all_results:
            f.write(result_item + '\n')




if __name__ == "__main__":
    current_path = os.path.dirname(__file__)  # 当前脚本所在目录
    parent_path = os.path.dirname(current_path)  # 上一级目录

    question_dir = 'golden_dataset.jsonl'
    question_list = os.path.join(parent_path, 'R1_Evaluation_Framework', question_dir)
    create_related_josnl(question_list)

    run_results = os.path.join(current_path, 'run_results.jsonl')
    test = Test(run_results)
    result = test.ragas_evaluate()
    print(result)













