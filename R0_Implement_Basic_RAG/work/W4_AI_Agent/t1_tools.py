import math
import os
from dotenv import load_dotenv
from langchain_core.tools import tool  # 从LangChain核心库导入tool装饰器
from langchain_tavily import TavilySearch  # 导入Tavily搜索工具

# 使用 @tool 装饰的那个函数，必须包含一个文档字符串（docstring）
@tool(description="计算一个数字的平方根")   # 装饰器将任何Python函数封装成一个Tool
def simple_sqrt(x:float) -> float:
    if (x<0):
        return "错误，值小于0"
    return math.sqrt(x)

if os.getenv("TAVILY_API_KEY") is None:
    print("警告：TAVILY_API_KEY环境变量未设置，搜索工具将无法工作。")
    raise ValueError("TAVILY_API_KEY is not set!")

search_tool = TavilySearch(max_results=2)  #  LangChain自动使用TAVILY_API_KEY  max_results=2 每次搜索返回2条结果

if __name__ == '__main__':
    print("开始测试")

    print("测试平方根工具")
    x = input()
    result = simple_sqrt.invoke({"x":x})
    print(f"计算{x}的平方根是{result}")

    #测试负数
    result_fail = simple_sqrt.invoke({"x": -4})
    print(f"计算 -4 的平方根: {result_fail}")

    if os.getenv("TAVILY_API_KEY"):
        print("测试搜索工具")
        search = input("输入要搜索的内容")
        result_search = search_tool.invoke({"query":search})   # 返回一个 字典

        print(result_search['results'][0]["content"])

        for chunk in result_search['results']:
            # print(type(chunk))
            print(f"  来源 (URL): {chunk['url']}")
            print(f"  内容 (Content): {chunk['content']}")

"""{
    "query": "搜索的问题",
    "follow_up_questions": None,  # 无后续问题
    "answer": None,  # 无直接答案
    "images": [],  # 无图片
    "results": [  # 搜索结果列表（包含2条新闻）
        {
            "url": "新闻链接",
            "title": "新闻标题",
            "content": "新闻内容",
            "score": 相关性分数,
            "raw_content": None
        },"""


