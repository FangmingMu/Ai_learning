import os
import time # <--- 导入 time 库，用于实现延时
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma   # Chroma向量数据库
from langchain_community.embeddings import DashScopeEmbeddings   #Embedding模型的类


pdf_name = 'PCB.pdf'
current_path = os.path.dirname(__file__)
pdf_path = os.path.join(current_path, pdf_name)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
    length_function = len,
    is_separator_regex = False,
)

embedding = DashScopeEmbeddings(      # DashScope Embedding模型的实例   翻译官
    model="text-embedding-v1",   # 向量化模型名称
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")    # API密钥
)


if not os.path.exists(pdf_path):
    print(f"错误：文件 '{pdf_path}' 不存在。请检查文件名和路径是否正确。")

else:
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f"--- PDF加载成功！共 {len(docs)} 页。---")

        split_docs = splitter.split_documents(docs)
        print(f"--- 文本分割完成！共 {len(split_docs)} 块。---")

        persist_directory = "chroma_db"    #文件夹名称   在硬盘上存储向量数据库

        print("\n--- 正在创建或加载向量数据库... ---")
        vector_db = Chroma(         # Chroma数据库的实例  操作和控制电视机的工具
            persist_directory=persist_directory,       # 数据库的位置  如果目录已存在，则会加载它
            embedding_function=embedding          # 使用Embedding模型来将文本转换成向量
        )

        batch_size = 16     #每个批次处理的文本块数量
        total_docs = len(split_docs)

        print(f"\n--- 开始分批处理，共 {total_docs} 个文本块，每批 {batch_size} 个... ---")
        for i in range(0, total_docs, batch_size):     # range(start, stop, step)    起始值  结束值   步长
            batch = split_docs[i : i + batch_size]

            vector_db.add_documents(documents=batch)   #  .add_documents() 将当前批次的文档添加到数据库中   自动完成向量化和存储

            print(f"已处理 {min(i + batch_size, total_docs)} / {total_docs} 个文本块...")    # min()确保最后一个批次显示的数字不会超过总数

            time.sleep(1)       # 避免被服务器拒绝

        print("\n--- 所有文本块已自动持久化到硬盘！ ---")

        vector_docs_count = vector_db._collection.count()      # ._collection.count()查询数据库中当前存储的条目总数
        print(f"\n--- 确认存储成功：当前数据库中共有 {vector_docs_count} 条数据。---")

        # --- 搜索测试 ---   验证向量数据库是否能够正常工作
        print("\n--- 进行一次相似度搜索测试... ---")
        query = "PCB的主要组成部分是什么？"
        match_docs = vector_db.similarity_search(   # vector_db.similarity_search() 这是向量数据库的核心功能
            query,  # 查询问题
            k=3     # k个最相关的结果
        )

        print(f"对于问题 '{query}'，找到的最相关的3个文本块是：")
        for i, doc in enumerate(match_docs):
            print(f"\n--- 相关块 {i + 1} (来源: {doc.metadata}) ---")
            print(doc.page_content)

    except Exception as e:
        print(f"处理过程中发生错误: {e}")