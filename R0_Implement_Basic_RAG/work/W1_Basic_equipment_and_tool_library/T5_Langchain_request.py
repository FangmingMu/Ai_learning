import os
from langchain_openai import ChatOpenAI  # openai 的封装类

# 配置通义千问的大模型接口
llm = ChatOpenAI(
    model='qwen-turbo',
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    extra_body={"enable_thinking": False},
)

#简单的一次性输出
# result1 = llm.invoke("你是谁")

# 流式输出   用列表存储  慢慢输出   并且存储起来  flush=True 立刻显示出来
chunks = []
for chunk in llm.stream("你会干什么"):
    print(chunk.content, end="", flush=True)
    chunks.append(chunk.content)

# .join()  ""  空字符串，粘合时中间不加任何东西
full_response = "".join(chunks)
print(full_response)