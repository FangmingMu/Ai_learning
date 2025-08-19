import os
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class BM25():
    def __init__(self, filename=None):
        self.filename = filename

        current_path = os.path.dirname(__file__)
        PDF = os.path.join(os.path.dirname(current_path), 'R1_Evaluation_Framework/PDF/ARES RAG Evaluation.pdf')
        filename = filename or PDF

        loader = PyPDFLoader(filename)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)

        # 分词列表和原始文本列表
        self.corpus_tokenized = [doc.page_content.split() for doc in split_docs]
        self.corpus_original = [doc.page_content for doc in split_docs]

        # 创建BM25索引
        self.bm25 = BM25Okapi(self.corpus_tokenized)

    def search(self, query=None, index=3):
        # 准备查询
        if not query:
            raise ValueError("查询 query 不能为空")
        tokenized_query = query.split()

        # 执行检索，但把【原始文本列表】作为返回值的来源
        top_n_docs_text = self.bm25.get_top_n(tokenized_query, self.corpus_original, n=index)

        # for i, doc_text in enumerate(top_n_docs_text):      返回字符串
        #     print(f"  文档 {i + 1} (预览): '{doc_text[:200]}...'")

        return top_n_docs_text
