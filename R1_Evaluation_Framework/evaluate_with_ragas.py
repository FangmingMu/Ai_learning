# RAGAs 框架就像一个专业的“医生”，在诊断之前，它需要我们把病人的信息（我们的评估数据）整理成它习惯阅读的“病历”格式。
# 这个“病历”格式，就是 datasets 库中的 Dataset 对象。
import os
import json
from datasets import Dataset # 导入Hugging Face的datasets库
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

# RAGAs在评估时，需要一个强大的LLM来做“裁判”（LLM-as-a-Judge）
llm = ChatOpenAI(
        model_name="qwen-turbo",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
llm_for_ragas = LangchainLLMWrapper(llm)    # 包装成RAGAs认识的格式。

# 定义评估用的Embedding模型 (这是关键的新增部分！)
# 我们必须告诉RAGAs，在需要计算嵌入时，也要用我们自己的模型
ragas_embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 将我们配置好的裁判模型，设置为RAGAs所有评估指标的默认模型
faithfulness.llm = llm_for_ragas
answer_relevancy.llm = llm_for_ragas
context_precision.llm = llm_for_ragas
# context_recall不需要LLM，因为它只做文本匹配


def load_answer(file_path) -> Dataset:
    with open(file_path, 'r', encoding='utf-8') as f:
        data_list = [json.loads(line) for line in f]   # 整体解析出来 是充满字典的列表

    # 将数据列表转换成RAGAs要求的字典格式  每个键对应一个包含所有样本值的列表。
    # {'question': ['问题1', '问题2',...], 'answer': ['答案1', '答案2',...]}
    data_dict = {
        'question': [item['question'] for item in data_list],   # [ ... ] → 把所有提取的值放进一个新列表  item['question'] 获取question值
        'answer': [item['generated_answer'] for item in data_list],
        'contexts': [item['retrieved_contexts'] for item in data_list],
        'ground_truth': [item['ground_truth_answer'] for item in data_list]
    }

    dataset = Dataset.from_dict(data_dict)  # 把一个 Python 字典转换成 Hugging Face 的 Dataset 对象

    print("数据加载和转换完成！")
    print("\n转换后的Dataset对象预览:")
    print(dataset)

    return dataset

if __name__ == "__main__":
    filename = 'baseline_run_results.jsonl'
    ragas_dataset = load_answer(filename)

    result = evaluate(
        dataset=ragas_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=llm,  # 显式指定所有指标都用这个LLM作为裁判
        embeddings=ragas_embeddings,  # 显式指定所有需要嵌入的计算都用这个模型
        # raise_exceptions=False  # 在调试时保持为False，避免因单个错误中断
    )


    #打印评估报告
    print("\n--- RAGAs评估完成 ---")
    print("评估报告:")
    # {'faithfulness': 0.7690, 'answer_relevancy': 0.5268, 'context_recall': 0.5750, 'context_precision': 0.6486}
    print(result)


# faithfulness (忠实度): 这是评估幻觉的核心指标。
# 它的工作原理是：RAGAs会逐句分析生成的answer，然后让LLM判断每一句话是否都能在提供的contexts中找到直接或间接的支撑。
# 如果找不到，该指标得分就会降低。它回答的问题是：“你的答案是不是在胡说八道？”

# answer_relevancy (答案相关性): 这个指标关注的是答案是否切题。
# RAGAs会让LLM去比较question和answer，判断答案是否有效地回答了问题，而不是答非所问。它回答的问题是：“你的回答有用吗？”

# context_precision (上下文精准率): 这个指标评估检索的质量。
# 它会分析retrieved_contexts中的每个文本块，让LLM判断它们对于回答question是否真的有必要。
# 如果你的检索器找回了很多不相关的“噪音”信息，这个分数就会很低。它回答的问题是：“你找回来的信息，是不是都是精华？”

# context_recall (上下文召回率): 这是评估检索的另一个关键指标。
# 它会去比较你的retrieved_contexts和我们人工标注的ground_truth_contexts
# （在Dataset中，RAGAs会自动寻找ground_truth这个字段，并假设它就是标准答案，但对于上下文召回，它实际需要的是ground_truth_contexts，这一点RAGAs的设计有点不直观，但它内部会处理）。
# 它判断的是：“所有应该被找到的关键信息，你都找到了吗？”。这个指标直接依赖于我们黄金数据集的标注质量。
