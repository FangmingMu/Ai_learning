"""ReAct 是 "Reasoning" (推理) 和 "Acting" (行动) 的缩写。它的核心思想非常符合直觉，就是模仿人类解决问题的思考模式。
ReAct框架就是让LLM来模仿这个 思考 -> 行动 -> 观察 的循环。
Thought (思考): LLM根据当前的目标和历史步骤，在内部进行推理，决定下一步该做什么（是直接回答，还是使用某个工具）。
Action (行动): LLM决定使用一个工具，并生成调用该工具所需要的输入。比如，选择 simple_sqrt 工具，并提供输入 {"x": 9}。
Observation (观察): Agent执行这个行动（即调用工具），并将工具返回的结果作为“观察”到的新信息。比如，simple_sqrt 工具返回了 3.0。
重复循环: LLM接收到这个新的观察结果，进入下一轮思考。它会想：“我已经观察到结果是3.0了，这足以回答最初的问题了。” 于是，它决定不再使用工具，而是直接生成最终答案。
这个循环赋予了LLM强大的能力，让它能处理复杂任务、使用外部工具、"""

import os
from tabnanny import verbose

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool  # 从LangChain核心库导入tool装饰器
import math
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent  # Agent的核心模块
from langchain import hub   # 拉取预设提示词的工具




@tool(description="对一个数字进行开方操作")
def simple_sqrt(x:float)->float:
    try:  # 进行数据类型转换和异常处理
        numeric_x = float(x)
    except (ValueError, TypeError):
        return "错误：输入值必须是一个有效的数字。"

    # 使用转换后的数字进行后续操作
    if numeric_x < 0:
        return "错误：输入值必须为非负数。"

    return math.sqrt(numeric_x)


tavily_api = os.getenv("TAVILY_API_KEY")

search_tool = TavilySearch(max_result=2)

llm = ChatOpenAI(
    model_name = "qwen-turbo",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    extra_body={"enable_thinking": False},
)

tools =[simple_sqrt, search_tool]   # 创建的工具放入列表中  Agent会从列表中选择需要使用的工具

prompt = hub.pull("hwchase17/react")  # 拉取一个预设的ReAct提示词模板  告诉LLM如何遵循ReAct的思考-行动-观察循环

agent = create_react_agent(llm, tools, prompt)  # 将LLM、工具列表和提示词模板“粘合”在一起，定义了Agent的行为逻辑

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # AgentExecutor执行引擎  负责实际运行Agent的思考-行动循环  verbose=True 打印完整的思考链

if __name__ == '__main__':
    print("测试agent")

    print("9的平方根测试")

    response1 = agent_executor.invoke({'input':'9的平方根是多少'})  # 9的平方根是多少  字符串传给函数

    print(response1['output'])


    print("搜索测试")
    response2 = agent_executor.invoke({'input':'今天上海什么天气，中文回答'})
    print(response2['output'])








