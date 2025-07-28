from fastapi import FastAPI
from fastapi.openapi.models import APIKey
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
print(f"数据库的绝对路径是: {DBPATH}")



class Requset(BaseModel):
    query:str

embedding = DashScopeEmbeddings(
            model = "text-embedding-v1",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
           )

if not os.path.exists(DBPATH):
    print("没有数据库，正在创建")
    current_path = os.path.dirname(__file__)
    pdf_path = os.path.join(current_path, PDFNAME)
    print("找到文件")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print("成功加载文档")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap=50,
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

retrieval = vector_db.as_retriever()
template = """
请根据以下上下文信息，用中文回答问题。
如果你在上下文中找不到答案，就说你不知道，不要试图编造答案。

上下文:
{context}

问题:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

output_parser = StrOutputParser()  # 输出解析器，它会将LLM的输出(AIMessage)转换成简单的字符串
llm = ChatOpenAI(
        model_name = "qwen-turbo",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        extra_body={"enable_thinking": False},
    )

print("构建链")
rag_chain = (
    {"context": retrieval, "question": RunnablePassthrough()}    # RAG链的初始输入需要如何被处理和分发    #RunnablePassthrough未经修改的原始赋值
    | prompt  # 提示词模板
    | llm
    | output_parser  # 输出解析器
)


app = FastAPI(title="RAG流式问答系统API", version="1.1")
async def stream_rag_response(query:str):   # 异步生成器函数 (Async Generator)  连接astream 和 FastAPI 的管道
    try:
        # astream会异步地流式返回结果的每个文本块(chunk)
        async for chunk in rag_chain.astream(query):  # # LCEL链的astream方法直接返回文本片段(chunks)，不再是复杂的字典
            yield chunk
    except Exception as e:
        # 如果在流式传输过程中发生任何错误，在此处捕获
        print(f"RAG链执行出错: {e}")
        # 在流中返回一个错误信息，这样前端也能知道发生了问题
        yield f"处理请求时发生错误，请检查服务器日志。错误: {e}"

# async def stream_rag_response(query: str):
#     """异步流式响应生成器 - 添加详细调试"""
#     # 测试检索功能
#     # docs = retrieval.get_relevant_documents(query)
#     # print(f"检索到 {len(docs)} 个相关文档")
#     # if docs:
#     #     print(f"第一个文档预览: {docs[0].page_content[:100]}...")
#     #
#     # chunk_count = 0
#     # full_response=""
#
#     async for chunk in rag_chain.astream(query):
#         if chunk:
#             # chunk_count += 1
#             # # 打印前几个chunk的详细信息
#             # if chunk_count <= 5:
#             #     print(f"Chunk {chunk_count}: type={type(chunk)}, content='{chunk}'")
#
#             yield chunk
#             # full_response += chunk
#             await asyncio.sleep(0.01)

    # print(f"流式响应完成: 总共{chunk_count}个块, 总长度{len(full_response)}字符")

    # if chunk_count == 0:
    #     error_msg = "没有生成任何响应内容"
    #     print(error_msg)
    #     yield error_msg


@app.post("/chat")  # 不需要 response_model，因为响应是流式的，没有固定的模型
async def chat_astream(request:Requset):
    print("执行调用函数")
    # 直接返回 StreamingResponse 对象  异步生成器函数 `stream_rag_response`   # media_type="text/plain; charset=utf-8"指定返回的是纯文本流，并UTF-8
    return StreamingResponse(
        stream_rag_response(request.query),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# uvicorn fastapi_app:app --reload
# netstat -ano | findstr :8000
# taskkill /PID 31104 /F
# streamlit run streamlit_app.py

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)





