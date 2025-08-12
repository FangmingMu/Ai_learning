"""(数据处理)： 阅读  理解  PDF参考资料   存入 向量数据库
开始考试 (问答循环)：
接到问题: 你提出一个问题。
翻阅资料 (检索)：  在  向量数据库   中搜索与问题最相关的几段原文（知识点）。
整理草稿 (增强)： 把 原始问题和 找到的参考资料放在一起，形成一个内容更丰富的新问题（Prompt）。
组织答案 (生成)： 把这份“包含了参考资料的草稿”提交给大语言模型（LLM），让LLM基于这些上下文，生成一个精准、流畅的最终答案。"""

import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA  # 封装好的RAG链  将检索 和 问答  连接起来。



db_path = "chroma_db"
embeding = DashScopeEmbeddings(   # 翻译官
    model="text-embedding-v1",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

def main():
    if os.path.exists(db_path):
        print(f"--- 正在从 '{db_path}' 加载已存在的向量数据库... ---")
        vector_db = Chroma(      # 构造函数  加载硬盘上的数据库   打开写好的
            persist_directory=db_path,
            embedding_function=embeding
        )
        print("--- 数据库加载成功！---")
    else:
        print(f"--- 未找到数据库，正在创建新的数据库... ---")
        pdf_name = "PCB.pdf"
        current_path = os.path.dirname(__file__)
        pdf_path = os.path.join(current_path, pdf_name)
        if not os.path.exists(pdf_path):
            print(f"错误：PDF文件 '{pdf_path}' 未找到！")
            return  # 退出程序
        print(f"正在加载 '{pdf_path}'...")
        pyload = PyPDFLoader(pdf_path)
        docs = pyload.load()      # 返回 Document对象列表  每一项代表PDF的一页
        print(f"加载文件完成  加载的文件有{len(docs)}页")

        print("正在分割...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,    # 用于计算长度的函数
            is_separator_regex=False,
        )

        spliter_docs = splitter.split_documents(docs)  # 返回 Document对象列表   每一项代表一个小的文本块
        print(f"分割完成 分割了{len(spliter_docs)}页")
        vector_db = Chroma.from_documents(    # 类方法 从文档中创建
            documents=spliter_docs,
            embedding=embeding,
            persist_directory=db_path
        )

        print(f"存储完成  存储了{vector_db._collection.count()}条数据")

    retriever = vector_db.as_retriever()    # 将向量数据库转换为一个“检索器”组件   接收一个查询字符串，返回最相关的文档列表
    # 组成链
    llm = ChatOpenAI(
        model="qwen-turbo",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        extra_body={"enable_thinking": False},
    )
    qa_chain = RetrievalQA.from_chain_type(
        retriever=retriever,
        chain_type="stuff",   # "stuff" 一次塞入所有文档   map_reduce 两步处理 先消化后汇总   refine  不断迭代答案  map_rerank 排名选最好
        llm=llm,
    )   #return_source_documents: True  返回结果中会包含原始文档  溯源

    print("\n--- RAG问答系统已启动，您可以开始提问了 ---")
    while True:
        query = input("\n请输入您的问题 (输入 '1' 来结束程序): ")
        if query.lower() == '1':        # .lower() 原始字符串全小写
            break
        if query:
            try:
                print("--- 正在检索并生成答案，请稍候... ---")
                result = qa_chain.invoke({"query": query})   # 调用组装好的RAG链  字典作为输入 链规定，输入的键必须是 query   它的值 是输入的query
                print(result['result'])    # result是字典  包含  query result source_documents等
            except Exception as e:
                print(f"在处理您的问题时发生错误: {e}")


if __name__ == "__main__":
    main()






