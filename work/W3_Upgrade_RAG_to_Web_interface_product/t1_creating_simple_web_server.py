from fastapi import FastAPI

app = FastAPI(       # FastAPI核心   所有的API路由都通过它来注册   Web服务器的总调度中心
    title = "LangChain Server",
    version = "v1.0",
    description = "A simple server to serve your LangChain applications.",
)

@app.get("/")   #装饰器 (Decorator)  告诉FastAPI 当GET访问服务器的根路径 ("/") 时 执行紧跟它的那个函数（`read_root`）
def read_root():
    return {"message": "Hello, World"}    # 返回字典,FastAPI自动将它转换成JSON格式的响应

@app.get("/items/{item_id}")    # 注册另一个路由  处理对 "/items/{item_id}" 的GET请求
def read_item(item_id: int, q: str | None = None): # item_id: int  路径参数  类型提示  q: str | None = None  查询参数
    # 网址 http://127.0.0.1:8000/items/9   {item_id}直接送网址中获取的
    return {"item_id": item_id, "q": q}   # 返回一个字典，包含了从路径中捕获到的 item_id。

if __name__ == "__main__":
    print("这是一个FastAPI应用。请使用 uvicorn 启动它，例如：")
    # uvicorn: 要使用的服务器程序   :app: FastAPI实例   --reload: 它会监视你的代码文件，一旦修改并保存，自动重启加载最新的代码
    print("uvicorn t1_creating_simple_web_server:app --reload")
