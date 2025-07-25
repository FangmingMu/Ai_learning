import os
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.chains import RetrievalQA
from pydantic import BaseModel  # pydantic库 定义数据模型

PDFNAME = 'PCB.pdf'
DBPATH = 'chroma_db'

print("正在创建数据模型")
class ChatRequest(BaseModel):    # 定义请求体模型：规定了客户端向/chat接口发送POST请求时，JSON应该有的结构
    query: str  # 必须包含一个名为"query"的字段，且其值必须是字符串

class ChatResponse(BaseModel):   # 定义响应体模型：规定了/chat接口返回给客户端的JSON的结构
    answer: str # 必须包含一个名为"answer"的字段，其值为字符串


print("正在创建RAG链")
if os.path.exists(DBPATH):
    print("正在加载数据库")
    vector_db = Chroma(
        persist_directory=DBPATH,
        embedding_function=DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
    )
else:
    print("正在创建数据库")
    current_path = os.path.dirname(__file__)
    pdf_path = os.path.join(current_path, PDFNAME)
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    chunk = splitter.split_documents(docs)

    vector_db = Chroma.from_documents(
        documents=chunk,
        embedding=DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        ),
        persist_directory=DBPATH
    )

retrieval = vector_db.as_retriever()
llm = ChatOpenAI(
    model_name = "qwen-turbo",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    extra_body={"enable_thinking": False},
)

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = retrieval,
    chain_type = "stuff",
    return_source_documents = True,
)

app = FastAPI(   # 这里的title, version, description等会显示在自动生成的API文档中  http://127.0.0.1:8000/docs
    title = "RAG问答系统API",
    version = "v1.0",
    description = "一个基于LangChain和FastAPI的、能够回答PDF文档相关问题的API",
)

@app.post("/chat", response_model=ChatResponse)
# 注册一个POST方法的路由，路径是 /chat  response_model=ChatResponse: 指定接口的响应体遵循我们定义的ChatResponse模型
def chat(request: ChatRequest):
    try:
        result = qa_chain.invoke({"query": request.query})
        answer = result["result"]   # 使用[]访问字典里的值  创建字典是{}
        print(f"生成答案: {answer}")
        return ChatResponse(answer=answer)   # 将答案封装在ChatResponse模型中返回   FastAPI会自动将其序列化为JSON
    except Exception as e:
        print(f"处理请求时发生错误: {e}")
        # 可以返回一个HTTP错误
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail="处理请求时发生内部错误")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# uvicorn t2_rag_api:app --reload
# 使用postman 进行post请求








