import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter     # 递归字符文本分割器

pdf_name = "PCB.pdf"
cur_path = os.path.dirname(__file__)
pdf_path = os.path.join(cur_path, pdf_name)

if not os.path.exists(pdf_path):
    print(f"错误：文件 '{pdf_path}' 不存在。请检查文件名和路径是否正确。")

else:
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()      # .load() 方法返回一个文档列表，每个文档代表PDF的一页
        print(f"--- PDF加载成功！总共加载了 {len(docs)} 页。---")

        text_splitter = RecursiveCharacterTextSplitter(      # 创建一个文本分割器实例
            chunk_size=500,    # 每个文本块的最大长度
            chunk_overlap=50,   # 块与块之间的重叠字符数   避免在分割点切断一个完整的句子或重要的上下文
            length_function=len,    # 用来计算文本长度的函数  默认len()
            is_separator_regex=False,   # 是否将分隔符视为正则表达式
        )

        split_docs = text_splitter.split_documents(docs) # 使用.split_documents() 分割documents列表  遍历每一页 并按规则进行切分
        print(f"--- 分割完成！总共分割成了 {len(split_docs)} 个文本块。---")


        for i, chunk in enumerate(split_docs):
            if (i == 5):
                break
            print(f"第{i+1}块\n")
            print(chunk.page_content)    #chunk 是一个 Document 对象   .page_content 属性存放着文本块的内容。
            print(f"来源: {chunk.metadata}")
            print("-" * 20)  # 打印一个分隔线







    except Exception as e:
        # 捕获所有可能的错误
        print(f"处理过程中发生错误: {e}")

