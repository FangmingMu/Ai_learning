import requests   # request 方法 get  head  post  delete

api_url = "https://v2.jinrishici.com/one.json"  # API 的URL。“今日诗词”，随机返回一条中国古诗词

print("正在发送 GET 请求至:", api_url)
response = requests.get(api_url)  # .get()向URL发送HTTPGET请求。最常见的HTTP方法，从服务器获取读取数据。 程序会暂停，等待通信完成，返回一个响应对象,包含了服务器返回的所有信息
print("请求完成，接收到响应:", response) # 打印响应对象本身，会显示状态码   返回的是Response 对象


if response.status_code == 200:  # HTTP 状态码 200 代表 已成功处理   1** 继续  2** 成功 3** 重定向  4**客户端错误  5**服务器错误
    response.encoding = 'utf-8'      # 编码处理
    print("已将响应编码强制设置为:", response.encoding)


    print("\n--- 原始二进制内容 (response.content) ---")
    print(response.content)  # 返回的是最原始的、未经任何解码的二进制数据 (bytes)。 数据的裸奔形态  处理非文本内容必须使用 .content


    print("\n--- 解码后的文本内容 (response.text) ---") # 返回的是指定的编码解码后的字符串 (string)。 处理 HTML 页面、JSON 字符串等文本数据。
    print(response.text)


    print("\n正在将 JSON 文本解析为 Python 字典...")
    data_dict = response.json()    # 会做   1 读取响应内容  2 使用内置的 JSON 解析器将其从 JSON 格式的字符串转换成 Python 的数据结构（字典或列表）
    print("\n--- 解析后的 Python 字典 (data_dict) ---")
    print(data_dict)


    full_poem_list = data_dict['data']['origin']['content']  # 字典中，通过键（key）提取需要的数据

    print("\n--- [最终结果] 提取到的全诗内容 ---")
    # 使用 for 循环遍历包含诗句的列表，并逐行打印，格式更美观。
    for sentence in full_poem_list:
        print(sentence)

# 如果状态码不是 200，则进入这个 else 块。
else:
    # 打印错误提示，包含状态码，方便排查问题（如网络不通、URL错误、服务器故障等）。
    print(f"\n请求失败！HTTP 状态码: {response.status_code}")
    print("错误原因可能为：网络问题、API地址失效、服务器故障等。")