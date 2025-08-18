# R2: 检索系统深度优化 🎯

## 🎯 模块核心目标

本模块的核心目标是**全面提升信息检索的精准度、召回率和相关性**。在 `R1` 中，我们可能已经发现，许多“糟糕的答案”源于“糟糕的上下文”。本模块将系统性地学习和实践工业界最主流的检索优化策略，将 RAG 系统的“天花板”尽可能地抬高。我们的每一项优化，都将通过 `R1` 建立的评估框架进行量化验证。

## 🧠 核心概念

1.  **语义 VS 词法 (Semantic vs. Lexical):** 向量检索是基于“语义”相似度的，而传统的关键词检索（如BM25）是基于“词法”匹配的。两者各有优劣，结合使用效果更佳。
2.  **分块的艺术 (The Art of Chunking):** 如何切分文档，直接决定了检索结果的质量。好的分块应该在“保持语义完整性”和“保证内容聚焦”之间找到平衡。
3.  **查询的意图 (Query Intent):** 用户的原始问题，不一定是机器最“喜欢”的查询。通过改写和扩展用户的查询，可以显著提升检索效果。
4.  **召回与排序 (Recall & Rerank):** 一个完整的检索系统，通常分为两步：首先，快速、广泛地“召回”一批可能相关的候选文档（Recall）；然后，用一个更强大的模型，对这批候选文档进行“精排”（Rerank）。

## 🛠️ 技术栈与工具

*   **核心框架:** `LangChain`, `LlamaIndex`
*   **关键词检索:** `rank_bm25`
*   **重排模型:** `FlagEmbedding` (for BAAI/bge-reranker)
*   **向量数据库:** (可选) `Milvus`, `Weaviate`
*   **评估框架:** (复用R1) `ragas`, `langsmith-sdk`

---

## 🚀 行动计划：逐个击破检索瓶颈

### **步骤一：智能文档分块 (Intelligent Chunking)**

这是优化的第一步，也是最基础的一步。垃圾分块，神仙难救。

* [ ] **2.1. 学习语义分割策略:**
    *   深入研究 `RecursiveCharacterTextSplitter` 的工作原理，理解它是如何利用 `["\n\n", "\n", " ", ""]` 等分隔符来尝试保留语义块的。
    *   **实践任务:** 调整 `chunk_size` 和 `chunk_overlap` 参数，使用 `R1` 的评估框架，测试不同参数对检索性能（特别是`Context Recall`）的影响，并记录最佳参数组合。

    chunk_overlap=50   
        chunk_size=256  {'faithfulness': 0.6301, 'answer_relevancy': 0.4396, 'context_recall': 0.3333, 'context_precision': 0.4083}
        chunk_size=512  {'faithfulness': 0.7690, 'answer_relevancy': 0.5268, 'context_recall': 0.5750, 'context_precision': 0.6486}
        chunk_size=1024 {'faithfulness': 0.8542, 'answer_relevancy': 0.6490, 'context_recall': 0.7750, 'context_precision': 0.6389}
    chunk_overlap=100
        chunk_size=256  {'faithfulness': 0.7908, 'answer_relevancy': 0.4140, 'context_recall': 0.2667, 'context_precision': 0.3750}
        chunk_size=512  {'faithfulness': 0.6679, 'answer_relevancy': 0.6318, 'context_recall': 0.5167, 'context_precision': 0.5458}
        chunk_size=1024 {'faithfulness': 0.7498, 'answer_relevancy': 0.5768, 'context_recall': 0.6250, 'context_precision': 0.6347}

*   [ ] **2.2. 探索特定格式的分割器:**
    *   **学习任务:** 阅读 LangChain 文档，学习 `MarkdownHeaderTextSplitter` (用于Markdown文件) 和 `PythonCodeTextSplitter` (用于代码文件)。
    *   **实践任务:** 找一个`.md`或`.py`文件，用相应的分割器进行处理，观察其与通用分割器的区别。

### **步骤二：查询重写与扩展 (Query Transformation)**

不要假设用户的原始问题是最好的查询。

*   [ ] **2.3. 实现查询扩展 (Query Expansion):**
    *   **学习思想:** 对于一个模糊的问题，如“RAG有什么缺点？”，我们可以让LLM生成几个更具体的子问题，如“RAG的幻觉问题是什么？”、“RAG的知识时效性如何？”、“RAG的检索成本高吗？”。
    *   **实践任务:** 编写一个Chain，接收用户原始问题，调用LLM生成3-5个相关的子问题。然后，将原始问题和所有子问题**一起**发送给检索器进行检索。使用`R1`评估，对比该方法是否提升了`Context Recall`。
    baseline chunk_size=1024,chunk_overlap=50  {'faithfulness': 0.7871, 'answer_relevancy': 0.6463, 'context_recall': 0.7250, 'context_precision': 0.5917}
    

*   [ ] **2.4. 实现假设性文档嵌入 (HyDE - Hypothetical Document Embeddings):**
    *   **学习思想:** 先不直接用问题去检索，而是让LLM根据问题，先“凭空想象”一篇最能回答该问题的“理想文档”。然后，用这篇想象出的文档的向量，去数据库中寻找最相似的真实文档。
    *   **实践任务:** 使用LangChain中的HyDE实现，或自己搭建一个Chain来实现此流程。在`R1`上进行评估，看看它在哪些类型的问题上表现更好。

### **步骤三：混合检索 (Hybrid Search)**

结合“语义”与“词法”的力量。

*   [ ] **2.5. 学习并实现BM25关键词检索:**
    *   学习BM25算法的基本原理（TF-IDF的改进版）。
    *   **实践任务:** 使用`rank_bm25`库，为你`R0`中的文档库，创建一个BM25检索器。

*   [ ] **2.6. 融合向量与关键词检索结果:**
    *   **学习思想:** 分别从向量检索器和BM25检索器获取Top K个结果。然后，需要一个策略来合并和重排序这两组结果。最常用的方法是**倒数排名融合 (Reciprocal Rank Fusion, RRF)**。
    *   **实践任务:** 编写一个函数，实现RRF算法。将向量检索和BM25检索的结果进行融合，得到最终的候选列表。在`R1`上评估，对比混合检索相比单一检索的性能提升。

### **步骤四：重排模型 (Reranking)**

在“召回”之后，进行“精排”。

*   [ ] **2.7. 学习并使用Cross-Encoder模型:**
    *   **学习思想:** Cross-Encoder模型会同时接收“查询”和“文档块”，并输出一个0到1之间的相关性分数。它比向量相似度更精准，但计算成本更高，因此只适合用于对少量候选结果进行重排。
    *   **实践任务:**
        1.  加载一个开源的、高性能的重排模型，如 `BAAI/bge-reranker-base`。
        2.  在你现有的检索流程之后，加入一个重排步骤：先用混合检索召回Top 20个文档，然后用重排模型对这20个文档进行打分，最终只选择得分最高的Top 3-5个文档作为最终上下文。
        3.  **在`R1`上进行最终评估**，观察`Context Precision`是否有显著提升。

---

## ✅ 最终产出

当你完成本模块所有任务后，你的`R2/`文件夹下应该包含：

1.  一系列独立的实验脚本，如 `exp_chunking.py`, `exp_hyde.py`, `exp_hybrid_search.py`, `exp_reranker.py`。
2.  `optimized_retriever.py`: 一个集成了你所有优化策略的、最终版本的“超级检索器”模块。
3.  `OPTIMIZATION_REPORT.md`: 一份详细的优化报告文档，包含：
    *   一个清晰的表格，对比**基准RAG**和**优化后RAG**在`R1`评估集上的所有核心指标（`Context Precision/Recall`, `Faithfulness`, `Answer Relevancy`）的变化。
    *   对每一种优化策略（如HyDE, Reranking）的有效性进行分析，并附上实验数据支撑。
    *   记录你在优化过程中遇到的挑战，以及你解决这些挑战的思路。

## 🚀 下一步

现在，你拥有了一个性能强悍的检索器。但“好米”还需要“好厨师”来烹饪。带着你高质量的上下文，进入 **`R3: 生成质量与可控性优化`** 阶段，学习如何让LLM更好地利用这些信息，生成更完美的答案。