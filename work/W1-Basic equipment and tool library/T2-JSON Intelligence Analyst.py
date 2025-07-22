import requests


api_url = "https://v2.jinrishici.com/one.json"

print("正在发送 GET 请求至:", api_url)
response = requests.get(api_url)
print("请求完成，接收到响应:", response) # 打印响应对象本身，会显示状态码

if response.status_code == 200:


    response.encoding = 'utf-8'
    print("已将响应编码强制设置为:", response.encoding)


    print("\n正在将 JSON 文本解析为 Python 字典...")
    data_dict = response.json()

    full_poem_list = data_dict['data']['origin']['content']
    author_list = data_dict['data']['origin']['author']

    print("\n--- [最终结果] 提取到的全诗内容 ---")
    for sentence in full_poem_list:
        print(sentence)

    print("author: ")
    for font in author_list:
        print(font)

# 如果状态码不是 200，则进入这个 else 块。
else:
    # 打印错误提示，包含状态码，方便排查问题（如网络不通、URL错误、服务器故障等）。
    print(f"\n请求失败！HTTP 状态码: {response.status_code}")
    print("错误原因可能为：网络问题、API地址失效、服务器故障等。")