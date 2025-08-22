# 文件名: local_model_yovole.py

import os
import httpx
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# 确保.env文件被加载
load_dotenv()

# --- 3. 为 Hugging Face BGE 模型创建加载函数 ---
from sentence_transformers import SentenceTransformer


from sentence_transformers import SentenceTransformer

class HuggingFaceBGEEmbedding:
    """封装 BGE 模型以兼容 Chroma"""
    def __init__(self, model_name="BAAI/bge-small-en"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts):
        """方便直接调用"""
        return self.model.encode(texts, convert_to_tensor=False)

    def embed_documents(self, texts):
        """Chroma 需要的方法"""
        return self.model.encode(texts, convert_to_tensor=False)

    def embed_query(self, text):
        """Chroma 需要的方法"""
        # 单条文本也返回向量
        return self.model.encode([text], convert_to_tensor=False)[0]


def get_bge_embedding_model(model_name="BAAI/bge-small-en") -> HuggingFaceBGEEmbedding:
    """返回 BGE 嵌入模型包装对象"""
    return HuggingFaceBGEEmbedding(model_name)



def get_bge_embedding_model(model_name="BAAI/bge-large-en") -> HuggingFaceBGEEmbedding:
    """
    返回 BGE 嵌入模型包装对象
    """
    return HuggingFaceBGEEmbedding(model_name)


# --- 1. 为你的 Embedding 模型创建加载函数 ---
def get_embedding_model() -> OpenAIEmbeddings:
    # 从.env文件中读取你的配置
    api_key = os.getenv("QWEN_EMBEDDING_API_KEY")
    model_name = "Qwen3-Embedding-8B"  # 这是你指定的模型名称
    base_url = "https://ds-api.yovole.com/v1"  # 这是你的API基础地址

    if not api_key:
        raise ValueError("请在.env文件中配置你的API密钥 (YOVOLE_API_KEY)")

    # 使用ChatOpenAI兼容模式来配置Embedding
    # 因为你的URL结构是 /v1/embeddings，这与OpenAI的 /v1/embeddings 路径完全兼容
    embedding_model = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=base_url,  # 直接使用 base_url 参数
        # 对于标准的OpenAI兼容接口，通常不需要自定义http_client
    )
    return embedding_model


# --- 2. 为你的 LLM 创建加载函数 ---
def get_llm() -> ChatOpenAI:
    # 从.env文件中读取你的配置
    api_key = os.getenv("QWEN_LLM_API_KEY")
    model_name = "qwen3-coder-plus"  # 这是你指定的模型名称
    base_url = "http://ds-api.yovole.com/v1"  # 这是你的API基础地址

    if not api_key:
        raise ValueError("请在.env文件中配置你的API密钥 (YOVOLE_API_KEY)")

    llm = ChatOpenAI(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,  # 直接使用 base_url 参数
        temperature=0.0  # 在RAG任务中，建议将温度设为0，以获得更稳定、更具确定性的输出
    )
    return llm


# --- 3. 测试代码块 ---
if __name__ == '__main__':
    print("--- 正在测试 Yovole API 配置 ---")
    try:
        embedding = get_embedding_model()
        vector = embedding.embed_query("这是一个嵌入模型测试")
        print("\n✅ Yovole 嵌入模型测试成功！")
        print(f"   向量维度: {len(vector)}")
    except Exception as e:
        print(f"\n❌ Yovole 嵌入模型测试失败: {e}")

    print("-" * 40)

    try:
        llm = get_llm()
        response = llm.invoke("你好，请用一句话介绍一下你自己。")
        print("\n✅ Yovole 语言模型测试成功！")
        print(f"   模型回答: {response.content}")
    except Exception as e:
        print(f"\n❌ Yovole 语言模型测试失败: {e}")
