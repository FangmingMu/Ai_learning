# import os
# os.environ["OPENAI_API_KEY"] = '9e0115f4a5c704821455139573442737ce0b8243bc055622ba6689eb24a902a0'
# os.environ["SERPAPI_API_KEY"] = '9e0115f4a5c704821455139573442737ce0b8243bc055622ba6689eb24a902a0'
# from langchain.agents import load_tools, AgentType
# from langchain.agents import initialize_agent
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain.tools import tool
#
# llm = ChatOpenAI(
#     model="qwen-turbo",
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://api.openai.com/v1",
#     extra_body={"enable_thinking": False},
# )
#
# tools = load_tools(["serpapi"], llm=llm)
#
# # 如果搜索完想再计算一下可以这么写
# # tools = load_tools(['serpapi', 'llm-math'], llm=llm)
#
# # 如果搜索完想再让他再用python的print做点简单的计算，可以这样写
# # tools=load_tools(["serpapi","python_repl"])
#
# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
#
# agent.run("What's the date today? What great events have taken place today in history?")


import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate

# --- 设置环境变量 ---
# 建议使用更通用的环境变量名，或者直接在代码中传入
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
os.environ["TAVILY_API_KEY"] = "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" # Tavily 是一个更推荐的搜索工具

# --- 初始化LLM ---
llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    extra_body={"enable_thinking": False},
)
# --- 创建工具 ---
# 使用新方法直接创建工具，而不是通过 load_tools
# TavilySearchResults 是 LangChain 官方现在更推荐的搜索工具
tools = [TavilySearch(max_results=1)]

# --- 创建 ReAct 风格的 Prompt 模板 ---
# 这是新版 agent 的标准做法
template = """
Answer the following questions as best you can. You have access to the following tools:
请用搜索工具告诉我，现在的准确日期和时间是什么？
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Agent-scratchpad:{agent_scratchpad}
"""
prompt = PromptTemplate.from_template(template)


# --- 创建 Agent ---
# 使用新的 create_react_agent 方法
agent = create_react_agent(llm, tools, prompt)

# --- 创建 Agent 执行器 ---
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# --- 运行 Agent ---
result = agent_executor.invoke({
    "input": "请用搜索工具告诉我，足坛有什么大新闻  ？"
})

print(result['output'])