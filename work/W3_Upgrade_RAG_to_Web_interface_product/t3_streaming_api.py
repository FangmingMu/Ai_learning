from fastapi import FastAPI
from fastapi.openapi.models import APIKey
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
import os
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from openai import embeddings, models
from pydantic import BaseModel
import asyncio # 导入异步I/O库，用于处理异步生成器
from fastapi.responses import StreamingResponse
from sympy import false

PDFNAME = "PCB.pdf"
DBPATH = 'chroma_db'


def Requset(BaseModel):
    query:str


if not os.path.exists(PDFNAME):
    current_path = os.path.dirname(__file__)
    pdf_path = os.path.join(current_path, PDFNAME)
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=false
    )
    chunk = splitter.split_documents(docs)
    vector_db = Chroma.from_documents(
        persist_directory=DBPATH,
        documents=chunk,
        embedding_function=DashScopeEmbeddings(
            models = "text-embedding-v1",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        ),
    )

else:
    vector_db = Chroma(
        persist_directory=DBPATH,
        embedding_function=DashScopeEmbeddings(
            models="text-embedding-v1",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        ),
    )

retrieval = vector_db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm = ChatOpenAI(
        model_name = "qwen-turbo",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        extra_body={"enable_thinking": False},
    ),
    chain_type = "stuff",
    retrieval = retrieval,
)

app = FastAPI(title="RAG流式问答系统API", version="1.1")

async def stream_rag_response(query:str):   # 异步生成器函数 (Async Generator)  连接astream 和 FastAPI 的管道
    async for chunk in qa_chain.astream({"query":query}):   # async for遍历
        answer = chunk.get("result")   # chunk是一个字典，提取'result'键对应的值
        if answer:
            yield answer   # yield 关键字将 answer_piece 发送给客户端，然后暂停在这里  等待下一次迭代，而不会结束函数
            print(answer, end="", flush=True)
            await asyncio.sleep(0.01)  # 块之间加入微小暂停

@app.post("/chat")  # 不需要 response_model，因为响应是流式的，没有固定的模型
async def chat_astream(request:Requset):
    # 直接返回 StreamingResponse 对象  异步生成器函数 `stream_rag_response`   # media_type="text/plain; charset=utf-8"指定返回的是纯文本流，并UTF-8
    return StreamingResponse(stream_rag_response(request.query), media_type="text/plain; charset=utf-8")









