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
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


JUDGE_LLM_MODEL = "qwen-turbo"
EMBEDDING_MODEL = "text-embedding-v1"
INPUT_FILE = "baseline_run_results.jsonl"


class Test:
    def __init__(self, file_path=None):
        """
        初始化 Test 类，设置文件路径、LLM 和嵌入器
        """
        self.file_path = file_path
        self.question_list = os.path.join(os.path.dirname(__file__), 'golden_dataset.jsonl')

        self.dataset_map = self._golden_dataset_map(self.question_list)
        # 初始化 LLM
        self.llm = ChatOpenAI(
            model_name=JUDGE_LLM_MODEL,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # 包装 LLM 以用于 RAGAs
        self.llm_for_ragas = LangchainLLMWrapper(self.llm)

        # 初始化 Embeddings
        self.ragas_embeddings = DashScopeEmbeddings(
            model=EMBEDDING_MODEL,
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )

        # 设置 LLM 给各个指标
        faithfulness.llm = self.llm_for_ragas
        answer_relevancy.llm = self.llm_for_ragas
        context_precision.llm = self.llm_for_ragas

    def load_answer(self, file_path: str = None) -> Dataset:
        """
        从 JSONL 文件加载评估数据并转成 Dataset
        :param file_path: 指定的 JSONL 文件路径，如果为空则使用初始化的 self.file_path
        :return: Dataset 对象
        """
        path = file_path or self.file_path

        with open(path, 'r', encoding='utf-8') as f:
            data_list = [json.loads(line) for line in f]

        data_dict = {
            'question': [item['question'] for item in data_list],
            'answer': [item['generated_answer'] for item in data_list],
            'contexts': [item['retrieved_contexts'] for item in data_list],
            'ground_truth': [item['ground_truth_answer'] for item in data_list]
        }

        return Dataset.from_dict(data_dict)

    def ragas_evaluate(self, file_path=None):
        """
        对指定 JSONL 文件执行 RAGAs 评估，返回结果字典
        :param file_path: 指定的 JSONL 文件路径，如果为空则使用初始化的 self.file_path
        :return: RAGAs 评估结果字典
        """
        path = file_path or self.file_path
        dataset = self.load_answer(path)

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=self.llm_for_ragas,
            embeddings=self.ragas_embeddings
        )

        return result

    def _golden_dataset_map(self, file_path: str) -> dict:
        """创建问题为key的字典,返回值是新字典
        一个私有的辅助方法，只在初始化时调用一次。"""
        print(f"--- 正在加载并映射黄金数据集: {file_path} ---")
        dataset_map = {}     # 类型dict[str, dict]
        # question_list_str=[]
        with open(self.question_list, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)  # 把每行 JSON 字符串解析成 Python 字典
                question = item.get('question')  # 从字典中取 'question' 的值
                if question:
                    dataset_map[question] = item  # 以 question 作为 key，把整个字典存入 dataset_map

        # 生成器表达式  遍历question_list_str中字典的key值
        # next() 函数从生成器中取出下一个元素。  因为生成器是按需生成的，所以 next() 会立即返回第一个满足条件的字典。  没找到返回空
        # item = next((q for q in question_list_str if q['question'] == question), None)  太慢了  item是符合的字典
        # 把 JSONL 文件里的每条记录整理成一个字典映射 dataset_map， 用 dataset_map[question] 直接拿到对应的回答和上下文，而不必每次遍历整个 JSONL 文件

        print("--- 黄金数据集加载并映射完成 ---")
        return dataset_map

    def generate_answer(self, question:str=None, retrieved_contexts=None):
        """根据检索的内容生成答案并且返回json后的内容，可以直接进行写入操作"""
        answer_generation_prompt_template = """
            你是一个严谨的问答机器人。
            你的任务是【只使用】下面提供的【上下文】来回答【原始问题】。
            确保答案简洁、准确，并且完全基于所提供的资料。
            如果上下文信息不足以回答，就明确指出“根据提供的资料无法回答”。

            上下文:
            {context}

            原始问题:
            {question}

            答案:
            """
        answer_prompt = ChatPromptTemplate.from_template(answer_generation_prompt_template)
        answer_chain = answer_prompt | self.llm | StrOutputParser()

        generated_answer = answer_chain.invoke({
            "context": retrieved_contexts,
            "question": question
        })

        ground_truth_item = self.dataset_map.get(question)
        if ground_truth_item is None:
            raise ValueError(f"在预加载的数据集中未找到问题: {question}")

        answer_dict = {"question": question,
                       "ground_truth_contexts": ground_truth_item['ground_truth_contexts'],
                       "ground_truth_answer": ground_truth_item['ground_truth_answer'],
                       "retrieved_contexts": [doc.page_content for doc in retrieved_contexts],   # 获取纯文本列表,
                       "generated_answer": generated_answer}

        json_string = json.dumps(answer_dict, ensure_ascii=False)  # 字典转json  ensure_ascii=False 确保中文正确写入，不转换成编码
        return json_string
