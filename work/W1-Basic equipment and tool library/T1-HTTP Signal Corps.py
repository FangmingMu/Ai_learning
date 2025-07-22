# --------------------------------------------------------------------------
# 模块导入区 (Module Import Area)
# --------------------------------------------------------------------------
# 导入 requests 库，这是 Python 中用于发送 HTTP 网络请求的第三方标准库。
# 几乎所有与网络数据交互的 Python 程序都会用到它。
import requests

# --------------------------------------------------------------------------
# 全局变量定义区 (Global Variable Definitions)
# --------------------------------------------------------------------------
# 定义我们将要请求的 API (Application Programming Interface) 的 URL (Uniform Resource Locator)。
# 将 URL 定义成变量是一个好习惯，便于后续修改和维护。
# 这个 API 来自“今日诗词”，它会随机返回一条中国古诗词的数据。
api_url = "https://v2.jinrishici.com/one.json"

# --------------------------------------------------------------------------
# 核心逻辑执行区 (Core Logic Execution)
# --------------------------------------------------------------------------
# 使用 requests.get() 方法向指定的 URL 发送一个 HTTP GET 请求。
# GET 是最常见的 HTTP 方法，通常用于从服务器“获取”或“读取”数据。
# 程序会在这里暂停，等待网络通信完成，直到服务器返回一个响应对象。
# 这个响应对象包含了服务器返回的所有信息，我们将其存储在 `response` 变量中。
print("正在发送 GET 请求至:", api_url)
response = requests.get(api_url)
print("请求完成，接收到响应:", response) # 打印响应对象本身，会显示状态码

# 检查请求是否成功。HTTP 状态码 200 代表 "OK"，表示请求已成功处理。
# 这是一个非常重要的健壮性检查，确保我们只在请求成功时才处理数据。
if response.status_code == 200:

    # --------------------------------------------------------------------------
    # 响应内容处理区 (Response Content Handling)
    # --------------------------------------------------------------------------

    # --- 1. 编码处理 (Encoding Handling) ---
    # response.encoding 属性是 requests 库根据服务器响应头（Headers）猜测的编码格式。
    # 如果服务器没有明确指定编码，requests 可能会猜错，导致后续 .text 解码时出现乱码。
    # 手动将其设置为 'utf-8' 是处理中文内容时一个非常保险和常见的做法。
    # UTF-8 是目前互联网上最通用的字符编码，能兼容几乎所有语言。
    response.encoding = 'utf-8'
    print("已将响应编码强制设置为:", response.encoding)

    # --- 2. 查看不同形式的响应内容 ---
    # response.content: 返回的是最原始的、未经任何解码的二进制数据 (bytes)。
    # 打印出来会看到一串 b'...' 开头的内容，这是数据的“裸奔”形态。
    # 在处理非文本内容（如图片、视频、文件下载）时，必须使用 .content。
    # print("\n--- 原始二进制内容 (response.content) ---")
    # print(response.content)

    # response.text: 返回的是根据 response.encoding 指定的编码解码后的字符串 (string)。
    # 因为我们前面已经设置了 encoding='utf-8'，所以这里能正确显示中文。
    # 主要用于处理 HTML 页面、JSON 字符串等文本数据。
    # print("\n--- 解码后的文本内容 (response.text) ---")
    # print(response.text)

    # --- 3. 解析 JSON 数据 ---
    # response.json() 是一个非常方便的方法，它会做两件事：
    #   a. 读取响应内容。
    #   b. 使用内置的 JSON 解析器将其从 JSON 格式的字符串转换成 Python 的数据结构（通常是字典或列表）。
    # 注意：必须使用 `()` 来调用这个方法。
    print("\n正在将 JSON 文本解析为 Python 字典...")
    data_dict = response.json()
    # print("\n--- 解析后的 Python 字典 (data_dict) ---")
    # print(data_dict)

    # --- 4. 提取并打印目标数据 ---
    # 从解析后的字典中，像剥洋葱一样，通过键（key）逐层提取我们需要的数据。
    # 这种链式调用 `['data']['origin']['content']` 的前提是我们已经清楚了解 JSON 的结构。
    full_poem_list = data_dict['data']['origin']['content']

    print("\n--- [最终结果] 提取到的全诗内容 ---")
    # 使用 for 循环遍历包含诗句的列表，并逐行打印，格式更美观。
    for sentence in full_poem_list:
        print(sentence)

# 如果状态码不是 200，则进入这个 else 块。
else:
    # 打印错误提示，包含状态码，方便排查问题（如网络不通、URL错误、服务器故障等）。
    print(f"\n请求失败！HTTP 状态码: {response.status_code}")
    print("错误原因可能为：网络问题、API地址失效、服务器故障等。")