# 1  准备环境 安装 LangChain 和解析PDF库
# 2 定位文件：告诉程序你的PDF文件放在了电脑的哪个位置。
# 3 调用加载器：使用 LangChain 提供的 PyPDFLoader 来加载并解析这个文件。
# 4 查看结果：打印出加载后的文档列表长度，验证操作是否成功。

import os
from langchain_community.document_loaders import PyPDFLoader

pdf_name = "PCB.pdf"
cur_path = os.path.dirname(__file__)    #os.path.dirname(__file__) 获取当前脚本所在的文件夹路径。
pdf_path =os.path.join(cur_path, pdf_name)   #os.path.join() 智能拼接路径

if not os.path.exists(pdf_path):
    print(f"错误：文件 '{pdf_path}' 不存在。请检查文件名和路径是否正确。")

else:
    try:
        loader = PyPDFLoader(pdf_path)   #创建一个 PyPDFLoader 的实例对象
        docs = loader.load()     # .load() 执行加载和解析操作
        #加载PDF后  自动按页分割  `docs` 是一个列表   列表中的每一个元素都是一个 `Document` 对象   就是PDF中的一页。

        print("长度: ", len(docs))   #len() 获取 `documents` 列表的长度   等于PDF的总页数
        print("内容: ", docs[0].page_content[:100])
        print(docs[0].page_content[:200] + "...")   #.page_content 属性存放了该页的文本内容
        # docs[0] 第一页的Document对象   [:200]   字符串切片  开始位置留空  从第0个字符开始   结束位置 200，取到第199个字符

    except Exception as e:
        # 捕获在加载过程中可能发生的任何错误。
        print(f"加载PDF文件时发生错误: {e}")