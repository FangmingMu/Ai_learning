import os

from click import prompt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    extra_body={"enable_thinking": False},
)

prompt = PromptTemplate.from_template(
    "怎么评价人工智能"
)
chain = prompt | llm

steam = chain.stream({})

for chunk in steam:
    print(chunk.content, end="", flush=True)




