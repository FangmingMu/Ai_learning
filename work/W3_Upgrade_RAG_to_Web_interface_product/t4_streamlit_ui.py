import streamlit as st  # streamlit 库  约定俗成，让代码更简洁
import time

st.title("RAG问答")  # st.title() 设置网页标题

user_input = st.text_input("请输入问题", key="uers_query")  # st.text_input() 创建单行文本输入框  设置唯一的键

button = st.button("提交")  # 创建一个可以点击的按钮

if button:   # 当用户点击了“提交”按钮
    if user_input:
        st.write("正在思考")  # st.write() 万能打印命令，把任何东西（文本、数据、图表等）显示在网页上
        with st.spinner("正在思考"):  # st.spinner() 来显示一个加载动画   # with 创建一个临时的“上下文(Context)”环境，来自动管理一些资源的获取和释放
            time.sleep(3)
            answer = f"答案:"
        st.success("成功")  # st.success() 用一个绿色框来显示成功信息
        st.write(answer)
    else:
        st.warning("请输入你的问题！")   # st.warning() 用一个黄色的警告框来显示信息