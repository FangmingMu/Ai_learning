import json
from time import sleep

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms.tongyi import Tongyi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
import os
from dotenv import load_dotenv
import dashscope
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from local_model import get_llm, get_embedding_model, get_bge_embedding_model

task_instruction = "根据查询找到相关文档"
load_dotenv()
llm = get_llm()

# 选择 BGE 模型
# embedding = get_bge_embedding_model("BAAI/bge-large-en")  # small / base / large 都可以
embedding = get_embedding_model()


PDFNAME = "ARES RAG Evaluation.pdf"
question_name = "golden_dataset.jsonl"
baseline_run_results = "baseline_run_results.jsonl"
current_path = os.path.dirname(__file__)
pdf_path = os.path.join(current_path, "PDF", PDFNAME)
question_path = os.path.join(current_path, question_name)
DBPATH = 'chroma_db'
print(f"数据库的绝对路径是: {DBPATH}")




def create_answer_jsonl(pdf_path, DBPATH):
    if not os.path.exists(DBPATH):
        print("没有数据库，正在创建")
        print("找到文件")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print("成功加载文档")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        chunk = splitter.split_documents(docs)
        print("成功分割文本")


        vector_db = Chroma.from_documents(
            persist_directory=DBPATH,
            documents=chunk,
            embedding=embedding,
        )
        print("成功创建数据库")
    else:
        print("已经存在数据库，正在导入")
        vector_db = Chroma(
            persist_directory=DBPATH,
            embedding_function=embedding,
        )

    retrieval = vector_db.as_retriever(search_kwargs={
        "k": 20,
    })
    query = "ARES系统的全称是什么？"
    instructed_query = f"Instruct: {task_instruction}\nQuery: {query}"
    docs = retrieval.get_relevant_documents(instructed_query)

    print(f"检索到 {len(docs)} 条文档：\n")

    for i, doc in enumerate(docs):
        print(f"--- 文档 {i + 1} ---")
        # 输出内容前 300 个字符，防止太长
        print(doc.page_content[:300].replace("\n", " "))
        # 如果有 metadata，也可以打印
        if hasattr(doc, "metadata") and doc.metadata:
            print("metadata:", doc.metadata)
        print("-----------------\n")


    print("构建链")
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retrieval,
        return_source_documents=True  # 告诉链，请返回源文档
    )
    all_results = []

    # 打开文件指定编码  指定只读   jsonl 每行一个独立的JSON对象
    with open(question_path, 'r', encoding='utf-8') as f:  # with ... as f  下文管理器语法 自动管理资源 执行完毕后，会自动关闭文件
        for line in f:
            # print(type(line))  # 此时为str  json串  一个json对象
            item = json.loads(line)    # loads 解析字符串  load 调用read 解析文件   item为字典
            # print(type(item))

            # item = item.strip()   # strip 只能执行到字符串
            question = item['question']
            answer = rag_chain.invoke({"query": question})    # 返回的是字典

            generated_answer = answer.get('result')     # 字典的访问方式  获取答案
            # source_docs 是一个列表  每一项是一个 Document 对象  doc.page_content → 文本内容   doc.metadata  文档元信息（页码、来源、ID 等）
            source_docs = answer.get('source_documents', [])  # .get() 安全地取出字典里的 key，如果不存在则返回 []
            retrieved_contexts = [doc.page_content for doc in source_docs]   # 获取纯文本列表
            print(f"关于问题：{question}\n 检索的答案：{answer.get('result')}")

            answer_dict = {"question": question,
                           "ground_truth_contexts": item['ground_truth_contexts'],
                           "ground_truth_answer": item['ground_truth_answer'],
                           "retrieved_contexts": retrieved_contexts,
                           "generated_answer": generated_answer}

            json_string = json.dumps(answer_dict, ensure_ascii=False)   # 字典转json  ensure_ascii=False 确保中文正确写入，不转换成编码
            sleep(1)
            all_results.append(json_string)

    with open(baseline_run_results, 'w', encoding='utf-8') as f:
        for result_item in all_results:
            f.write(result_item + '\n')

if __name__ == "__main__":
    qa_chain = create_answer_jsonl(pdf_path, DBPATH)
    print("\n已经生成了基础的答案文档")
