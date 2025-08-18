# ragas_eval.py
import os
import json
from datasets import Dataset
from langchain_community.embeddings import DashScopeEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# 初始化 LLM 和 Embeddings
llm = ChatOpenAI(
    model_name="qwen-turbo",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
llm_for_ragas = LangchainLLMWrapper(llm)

ragas_embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 设置 LLM 给各个指标
faithfulness.llm = llm_for_ragas
answer_relevancy.llm = llm_for_ragas
context_precision.llm = llm_for_ragas

def load_answer(file_path) -> Dataset:
    """从 JSONL 文件加载评估数据并转成 Dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data_list = [json.loads(line) for line in f]

    data_dict = {
        'question': [item['question'] for item in data_list],
        'answer': [item['generated_answer'] for item in data_list],
        'contexts': [item['retrieved_contexts'] for item in data_list],
        'ground_truth': [item['ground_truth_answer'] for item in data_list]
    }

    return Dataset.from_dict(data_dict)

def ragas_evaluate(file_path):
    """对指定 JSONL 文件执行 RAGAs 评估，返回结果字典"""
    dataset = load_answer(file_path)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=ragas_embeddings
    )
    return result

