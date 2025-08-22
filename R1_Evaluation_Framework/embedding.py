import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

# --- 配置模型路径 ---
# 确保这些路径相对于您运行脚本的位置是正确的
# 如果此脚本放在 Ai_learning 文件夹下, 路径就是正确的
chat_model_path = "../Qwen2.5-0.5B-Instruct"
embedding_model_path = "../Qwen3-Embedding-0.6B"

# --- 自动检测可用设备 (GPU或CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备是: {device}")
print("-" * 50)


def test_chat_model():
    """
    测试 Qwen2.5-0.5B-Instruct 聊天模型
    """
    print("开始加载聊天模型...")
    try:
        # 加载分词器和模型
        # device_map="auto" 会自动将模型加载到可用的GPU或CPU上
        tokenizer = AutoTokenizer.from_pretrained(chat_model_path)
        model = AutoModelForCausalLM.from_pretrained(
            chat_model_path,
            torch_dtype="auto",  # 自动选择精度
            device_map="auto"
        )
        print("聊天模型加载成功！")
    except Exception as e:
        print(f"加载聊天模型失败: {e}")
        return

    print("\n--- 开始测试聊天模型 ---")
    prompt = "你好，请介绍一下你自己。"
    messages = [
        {"role": "system", "content": "你是一个乐于助人的人工智能助手。"},
        {"role": "user", "content": prompt}
    ]

    # 使用 apply_chat_template 来格式化输入
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 生成回复
    # 使用 ** 将 model_inputs 字典中的所有内容 (包括 input_ids 和 attention_mask) 一起传递
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"用户提问: {prompt}")
    print(f"模型回答: {response}")
    print("-" * 50)


def test_embedding_model():
    """
    测试 Qwen3-Embedding-0.6B 嵌入模型
    """
    print("开始加载嵌入模型...")
    try:
        # 加载分词器和模型
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
        model = AutoModel.from_pretrained(embedding_model_path).to(device)
        model.eval()  # 设置为评估模式
        print("嵌入模型加载成功！")
    except Exception as e:
        print(f"加载嵌入模型失败: {e}")
        return

    print("\n--- 开始测试嵌入模型 ---")
    sentences = [
        "今天天气真好",
        "我喜欢吃苹果",
        "这是一个测试句子"
    ]

    # 使用分词器处理文本
    # padding=True: 将句子填充到批次中最长句子的长度
    # truncation=True: 如果句子超过模型最大长度，则截断
    # return_tensors="pt": 返回 PyTorch 张量
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)

    # 使用 no_grad 来禁用梯度计算，节省计算资源
    with torch.no_grad():
        # 获取模型的输出
        outputs = model(**inputs)
        # 模型的最后一层隐藏状态，包含了每个token的向量表示
        last_hidden_states = outputs.last_hidden_state

    # --- 进行池化操作 (Mean Pooling) ---
    # 为了获得整个句子的单一向量表示，我们进行平均池化
    # 我们只对有效的token（非padding部分）进行平均
    attention_mask = inputs['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    masked_embeddings = last_hidden_states * mask
    summed = torch.sum(masked_embeddings, 1)
    counted = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / counted

    # 也可以使用 CLS token 的向量作为句子表示，但平均池化通常效果更稳健
    # cls_embedding = outputs.last_hidden_state[:, 0]

    embeddings = mean_pooled.cpu().numpy()

    print(f"要编码的句子: {sentences}")
    print(f"生成的嵌入向量维度 (句子数, 向量维度): {embeddings.shape}")
    print("第一个句子的嵌入向量 (前5个值):", embeddings[0][:5])
    print("-" * 50)


if __name__ == "__main__":
    test_chat_model()
    test_embedding_model()