import os
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


PDF_NAME = "PCB.pdf"
DB_PATH = "chroma_db_refactored"
EMBEDDING_MODEL = "text-embedding-v1"
LLM_MODEL = "qwen-turbo"
llm = ChatOpenAI(      # 大模型  模型名字  api  链接   其他参数 -是否是思考模型
    model=LLM_MODEL,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    extra_body={"enable_thinking": False},
)

def creat_or_load_vector_df(pdf_path: str, persist_directory: str, embedding_model: DashScopeEmbeddings) -> Chroma:
    """
    创建或加载向量数据库  数据处理 模块，封装了所有考前准备工作   加载分割存储
    pdf_path (str): 要处理的PDF文件路径
    persist_directory (str): 数据库在硬盘上的路径
    embedding_model (DashScopeEmbeddings): 用于向量化的Embedding模型
    Chroma: 一个已经就绪的、可供检索的Chroma向量数据库实例
    """
    if os.path.exists(persist_directory):
        print("加载已有的数据库")
        vector_db = Chroma(    # Chroma数据库的实例  就是遥控器  需要翻译官翻译文本    存储的地方
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
    else:
        print("创建新的数据库")
        current_path = os.path.dirname(__file__)
        pdf_path = os.path.join(current_path, pdf_path)
        if not (os.path.exists(pdf_path)):
            raise FileNotFoundError(f"错误：PDF文件 '{pdf_path}' 未找到！")

        loader = PyPDFLoader(pdf_path)
        pdf_docs = loader.load()
        print("加载文件成功")

        splitter = RecursiveCharacterTextSplitter(    # 分割文本   分割的大小  重叠的部分  长度函数
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        chunk = splitter.split_documents(pdf_docs)
        print("已分割文本")

        # 分批处理
        batch_size = 16
        total_docs = len(chunk)
        print(f"\n 分批处理，共 {total_docs} 个，每批 {batch_size} 个... ---")
        vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        for i in range(0, total_docs, batch_size):
            batch = chunk[i: i + batch_size]
            vector_db.add_documents(documents=batch)
            print(f"已处理 {min(i + batch_size, total_docs)} / {total_docs} 个文本块...")
        # 一次性处理
        # vector_db = Chroma.from_documents(
        #     documents=chunk,
        #     embedding=embedding_model,
        #     persist_directory=persist_directory
        # )
    return vector_db

def create_rag_chain(llm: ChatOpenAI, retriever) -> RetrievalQA:
    """
    创建和配置RAG链
    llm (ChatOpenAI): 生成答案的大语言模型实例
    retriever: 一个配置好的检索器实例     由vector_db.as_retriever()创建
    RetrievalQA: 一个已经组装好的、可直接调用的RAG链。
    """
    qa_chain = RetrievalQA.from_chain_type(    #使用类方法创建链  需要 检索器来检索   提示词模板  大模型    用大模型检索  给出检索的工具  和检索的方法
        retriever=retriever,
        chain_type="stuff",
        llm=llm,
        return_source_documents=True
    )
    print("已创建RAG链")
    return qa_chain

def start_interactive_session(qa_chain: RetrievalQA):
    """
    负责启动和管理用户交互的问答循环。
    qa_chain (RetrievalQA): 一个已经创建好的RAG链。
    """
    print("\n--- RAG问答系统已启动，您可以开始提问了 ---")
    while True:
        query = input("\n请输入问题   输入 1  结束程序 ")
        if query.lower() == '1':
            break
        else:
            try:
                result = qa_chain.invoke({"query": query})
                print(result['result'])
                for doc in result["source_documents"]:
                    print(
                        f"来源文件: {os.path.basename(doc.metadata.get('source', '未知'))}, 页码: {doc.metadata.get('page', '未知')}")
                    print("片段内容:", doc.page_content[:100] + "...")  # 打印前100个字符作为预览
                    print("-" * 20)
            except Exception as e:
                print(f"在处理您的问题时发生错误: {e}")

def main():
    embedding = DashScopeEmbeddings(    # 翻译官   需要模型进行翻译  需要模型的名字  api
        model="text-embedding-v1",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    vector = creat_or_load_vector_df(PDF_NAME, DB_PATH, embedding)
    retriever = vector.as_retriever()    # 角色转换  从数据库对象  转换为  检索的工具
    qa_chain = create_rag_chain(llm, retriever)
    start_interactive_session(qa_chain)


if __name__ == "__main__":
    main()





