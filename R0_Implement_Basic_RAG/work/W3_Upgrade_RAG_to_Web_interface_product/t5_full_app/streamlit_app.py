import streamlit as st
import requests # 导入requests库，用于发送HTTP请求


st.title("📄 RAG 问答机器人")
st.write("你好！我是基于你提供的PDF文档的问答机器人。请问有什么可以帮你的吗？")

API_BASE_URL = "http://127.0.0.1:8000"  # 定义后端API的地址  同一台电脑上，地址 http://127.0.0.1:8000

user_query = st.text_input("请输入你的问题:", "")

if st.button("发送"):
    if user_query:
        try:
            st.write("好的，我正在思考，请稍等...")

           # 开始调用后端
            payload = {"query": user_query} # 构造请求体JSON

            response = requests.post(  # 使用 requests.post() 发送请求到后端的 /chat 接口
                f"{API_BASE_URL}/chat",
                json=payload,  # 请求体
                stream=True,   # stream=True: 以流式方式接收响应
                headers={"Content-Type": "application/json"}  # 请求头  一个标准的HTTP头字段  JSON格式声明
            )

            if response.status_code == 200:
                answer_placeholder = st.empty() # st.empty() 创建一个“占位符”  更新内容

                full_answer = ""   # full_answer 用于累积所有接收到的文本片段

                for chunk in response.iter_content(chunk_size=1, decode_unicode=True): # 遍历流式数据块 chunk_size每次只读取1个字节 decode_unicode=True 解码成UTF-8字符串
                    if chunk:
                        full_answer += chunk   # 更新占位符的内容，实现实时显示   += 一直续  = 会擦除
                        answer_placeholder.markdown(full_answer)  # 动态占位符  用Markdown格式更新内容
            else:
                # 如果API返回错误，则显示错误信息
                st.error(f"API请求失败，状态码: {response.status_code}, 错误信息: {response.text}")

        except requests.exceptions.RequestException as e:# 如果网络连接失败，则显示错误信息
            st.error(f"连接后端API失败: {e}")
    else:
        st.warning("你好像什么都还没问呢！")  # 如果用户没有输入就点击按钮，给出提示



#     uvicorn fastapi_app:app --reload

#     streamlit run streamlit_app.py