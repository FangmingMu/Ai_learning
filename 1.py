# 文件名: test_rag_flow.py

import os
from time import sleep
import json
from scipy.spatial.distance import cosine
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from local_model import get_embedding_model, get_bge_embedding_model
from tqdm import tqdm

# ----------------------------
# 配置
# ----------------------------
PDFNAME = "ARES RAG Evaluation.pdf"
DBPATH = "chroma_test_db"
current_path = os.path.dirname(__file__)
pdf_path = os.path.join(current_path, "R1_Evaluation_Framework\\PDF", PDFNAME)

# ----------------------------
# 1️⃣ 读取 PDF 并分块
# ----------------------------
print("=== 读取 PDF 并分块 ===")
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"共加载 {len(docs)} 个文档")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False
)
chunks = splitter.split_documents(docs)
print(f"分块后共 {len(chunks)} 个 chunk 示例:")
print(chunks[0].page_content[:200])

# ----------------------------
# 2️⃣ 嵌入模型测试
# ----------------------------
print("\n=== 嵌入模型测试 ===")
# embedding = get_bge_embedding_model("BAAI/bge-small-en")  # small 模型
embedding = get_embedding_model()
sample_texts = [chunk.page_content for chunk in chunks[:3]]  # 取前3块测试
vectors = embedding.embed_documents(sample_texts)
print(f"向量数量: {len(vectors)}, 向量维度: {len(vectors[0])}")

# 余弦相似度示例
sim = 1 - cosine(vectors[0], vectors[1])
print(f"前两块文本余弦相似度: {sim:.4f}")

# ----------------------------
# 3️⃣ 创建 Chroma 数据库
# ----------------------------
print("\n=== 创建 Chroma 向量数据库 ===")
if os.path.exists(DBPATH):
    import shutil
    shutil.rmtree(DBPATH)  # 清空旧数据库
vector_db = Chroma.from_documents(
    persist_directory=DBPATH,
    documents=chunks[:3],  # 为测试只取前3块
    embedding=embedding
)
print(f"Chroma 数据库存储成功，路径: {DBPATH}")

# ----------------------------
# 4️⃣ 查询检索测试
# ----------------------------
retrieval = vector_db.as_retriever(search_kwargs={"k": 2})
query = "ARES系统的全称是什么？"
retrieved_docs = retrieval.get_relevant_documents(query)
print(f"\n查询: {query}")
print(f"检索到 {len(retrieved_docs)} 条文档")

for i, doc in enumerate(retrieved_docs):
    print(f"--- 文档 {i+1} ---")
    print(doc.page_content[:200].replace("\n", " "))
    print("----------------\n")

# ----------------------------
# 5️⃣ 验证检索结果相似度
# ----------------------------
print("\n=== 验证检索结果与 query 的余弦相似度 ===")
query_vec = embedding.embed_query(query)
for i, doc in enumerate(retrieved_docs):
    doc_vec = embedding.embed_documents([doc.page_content])[0]
    sim = 1 - cosine(query_vec, doc_vec)
    print(f"文档 {i+1} 相似度: {sim:.4f}")

print("\n✅ 流程测试完成！")
