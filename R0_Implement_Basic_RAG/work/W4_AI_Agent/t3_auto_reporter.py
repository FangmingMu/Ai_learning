import os
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from numpy.f2py.crackfortran import verbose

api = os.getenv("Tavily_API_KEY")

search_tool = TavilySearch()
tools = [search_tool]

llm = ChatOpenAI(
    model_name = "qwen-turbo",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    extra_body={"enable_thinking": False},
)

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


if __name__ == '__main__':
    print("--- 自动化新闻助理 Agent 已启动 ---")
    # 定义我们要研究的公司
    company_name = "特斯拉"

    task_prompt = f"""
        请你扮演一个专业的财经新闻记者。
        你的任务是：根据公司名称 "{company_name}"，自动上网搜索其近期（大约过去一个月内）的新闻，然后生成一段不超过80字的中文摘要。
        请确保摘要内容客观、中立，并涵盖主要的新闻要点。
        请开始你的工作。
        """

    print(f"\n正在为公司 '{company_name}' 生成新闻摘要...")

    response = agent_executor.invoke({'input':task_prompt})
    print("生成的新闻摘要如下：")
    print(response['output'])










