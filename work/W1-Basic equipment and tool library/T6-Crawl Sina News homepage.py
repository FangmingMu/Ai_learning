# 1发送网络请求  获取网页HTML      2 BeautifulSoup解析HTML，提取新闻标题    3  打印结果

import requests
from bs4 import BeautifulSoup

# 发送请求 要有请求头
URL = "https://news.sina.com.cn/"
header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
}
# timeout=10   超时   10秒没有应答就是超时
response = requests.get(URL, headers=header, timeout=10)
response.encoding = 'utf-8'
print(response.status_code)
html = response.text   # html  巨大的字符串 (string)   什么东西都有  代码也有

# 2
try:
    soup = BeautifulSoup(html, 'html.parser')   # 'html.parser': 指定使用Python内置的解析器    soup是BeautifulSoup 对象 按网页的结构（父子关系、层级）展开
    new_titles = []
    for item in soup.find_all('a'):       # soup.find_all('a') 会查找所有的<a>标签（超链接）
        title = item.get_text().strip()   # get_text() 获取标签内的所有文本内容     strip()  去除文本两端的空白字符
        href = item.get('href')     # get('href') 获取<a>标签的href属性   也就是链接地址

        if len(title) > 5 and href and href.startswith('http'):   # 需求  获取有链接的 字数有限制的标题
            new_titles.append(title)    # append 列表增加元素


# 3
    unique_titles = list(set(new_titles))  # 去除重复的标题，然后再转换回列表，每个标题只出现一次。

    for index, title in enumerate(unique_titles):    # enumerate 枚举函数 枚举列表
        print(f"{index + 1}. {title}")   # f""  表示格式化字符串  {}代替变量


except requests.exceptions.RequestException as e:
    # 如果在网络请求阶段发生任何requests相关的错误，就在这里捕获并打印。
    print(f"爬取失败，网络请求错误: {e}")

except Exception as e:
    # 捕获其他所有可能的未知错误。
    print(f"发生未知错误: {e}")




