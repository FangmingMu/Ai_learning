from os import getenv   #getenv 获取环境变量中的值   哈希取值
import requests   #请求
import os


"""
构建请求：
    请求头 (Headers)：告诉服务器我们是谁，我们要发送什么格式的数据。
    请求体 (Body/Payload)：我们具体要发送的内容，比如我们的问题。
    请求方法 (Method)：通常是 POST，因为我们要向服务器提交数据。
"""

#前置工作
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
api = getenv("DASHSCOPE_API_KEY")

# 构建请求头
headers = {
    'Content-Type': 'application/json',
    # 'Authorization' 的值必须是 "Bearer " 后面跟上你的API Key，注意中间有个空格
    'Authorization': f'Bearer {api}',   #如果是使用变量的值  需要{api}   api是字符串 不行
}

# 构建请求体
body = {
    'model': 'qwen-turbo',
    'messages': [
        {"role": "system", "content": "你叫小美"},
        {"role": "user", "content": "你今年23"},
    ]
}

# post请求 使用post方法   json: requests库非常智能，你直接把Python字典传给json参数，它会自动帮你序列化成JSON字符串，并设置好请求体。
response = requests.post(URL, headers=headers, json=body)   #请求到的是结构化的对象  response

if response.status_code == 200:
    print("--- 请求成功！模型返回的原始数据如下：---")

dict = response.json()         #整理成字典

#输出的内容是   {'choices': [{'message': {'role': 'assistant', 'content': '哎呀，你这么会......
print(dict['choices'][0]['message']['content'])  #查看content内容   {}是字典（哈希）   []是列表  数组


# print(dict.content)      AttributeError: 'dict' object has no attribute 'content'    对象 vs. 字典 (点. vs. 方括号[])




