from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

markdown_text = """
# RAG项目简介

## 模块一：数据处理与索引

这是数据处理模块的详细介绍。我们使用了最新的技术来确保数据质量。

### 步骤1.1: 文档加载

我们支持PDF、TXT和Markdown等多种格式的文档加载。

### 步骤1.2: 文本分割

文本分割是关键一步，我们采用了智能分块策略。

## 模块二：检索与生成

检索模块的目标是高效、准确。

### 步骤2.1: 向量化

我们使用了业界领先的嵌入模型。
"""

print("--- 原始Markdown文本 ---")
print(markdown_text)
print("-" * 50)

print("\n--- 🔪 使用通用分割器(RecursiveCharacterTextSplitter) ---")
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,      # 设置一个较小的尺寸，以便观察切割效果
    chunk_overlap=10
)

recursive_chunks = recursive_splitter.create_documents([markdown_text])
print(f"通用分割器切出了 {len(recursive_chunks)} 个文本块:")
for i, chunk in enumerate(recursive_chunks):
    print(f"  块 {i+1}: '{chunk.page_content}'")
    # 注意：通用分割器无法感知标题等元数据
    print(f"  元数据: {chunk.metadata}")


print("\n--- ✨ 使用专业的Markdown分割器(MarkdownHeaderTextSplitter) ---")

# 配置：告诉分割器，哪些标题是我们的“分割点”
# 它会根据这些标题，将文档分成不同的部分
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

# 注意：它的调用方式是 .split_text()，它返回的也是Document列表
# 它会把标题作为元数据，自动附加到每个文本块上
markdown_chunks = markdown_splitter.split_text(markdown_text)

print(f"Markdown分割器切出了 {len(markdown_chunks)} 个文本块:")
for i, chunk in enumerate(markdown_chunks):
    print(f"  块 {i+1}: '{chunk.page_content}'")
    # 观察这里的元数据，它包含了完整的标题层级！
    print(f"  元数据: {chunk.metadata}")
