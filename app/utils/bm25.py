# app/utils/bm25.py
import os
import pickle
import jieba
from rank_bm25 import BM25Okapi
from typing import List, Tuple
from app.config import settings

class BM25Retriever:
    def __init__(self):
        self.persist_path = settings.BM25_PATH
        self.bm25 = None
        self.documents = []
        self.metadatas = []
        self.tokenized_corpus = []
        
        self.load_index()

    def load_index(self):
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data["documents"]
                    self.metadatas = data["metadatas"]
                    self.tokenized_corpus = data["tokenized_corpus"]
                    self.bm25 = BM25Okapi(self.tokenized_corpus)
                print(f"✅ [BM25] 索引已加载，包含 {len(self.documents)} 条文档")
            except Exception as e:
                print(f"⚠️ [BM25] 索引加载失败: {e}")

    def save_index(self):
        data = {
            "documents": self.documents,
            "metadatas": self.metadatas,
            "tokenized_corpus": self.tokenized_corpus
        }
        # 确保目录存在
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump(data, f)

    def add_documents(self, docs: List[str], metas: List[dict]):
        if not docs: return
        
        print(f"🔨 [BM25] 正在增量更新索引 ({len(docs)} docs)...")
        new_tokenized = [list(jieba.cut_for_search(doc)) for doc in docs]
        
        self.documents.extend(docs)
        self.metadatas.extend(metas)
        self.tokenized_corpus.extend(new_tokenized)
        
        # BM25Okapi 需要全量重构
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.save_index()

    def search(self, query: str, top_k: int = 20) -> Tuple[List[str], List[dict]]:
        if not self.bm25:
            return [], []
            
        tokenized_query = list(jieba.cut_for_search(query))
        docs = self.bm25.get_top_n(tokenized_query, self.documents, n=top_k)
        
        # 找回 metadata (简化版逻辑)
        results_docs = []
        results_metas = []
        for d in docs:
            try:
                # 注意：如果文档完全重复，这里可能会取错 index，但在 RAG 场景通常可接受
                idx = self.documents.index(d)
                results_docs.append(d)
                results_metas.append(self.metadatas[idx])
            except ValueError:
                continue
                
        return results_docs, results_metas

# 单例导出
bm25_retriever = BM25Retriever()