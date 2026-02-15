# app/core.py
import os
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
#from langchain.text_splitter import RecursiveCharacterTextSplitter
# 新版langchain注意中间是下划线 _
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 引入我们刚才写好的 Utils 和 Config
from app.config import settings
from app.utils.ocr import ocr_engine
from app.utils.bm25 import bm25_retriever
from app.schemas import SourceDocument

class RAGService:
    def __init__(self):
        print("🚀 [Core] 正在初始化 RAG 核心服务...")
        
        # 1. 初始化 Chroma
        self.chroma_client = chromadb.PersistentClient(path=settings.DB_PATH)
        self.collection = self.chroma_client.get_or_create_collection(name=settings.DB_NAME)
        
        # 2. 初始化 Embedder (向量模型)
        print(f"   Load Embedding: {settings.MODEL_PATH}")
        self.embed_model = SentenceTransformer(settings.MODEL_PATH, local_files_only=True)
        
        # 3. 初始化 Reranker (精排模型)
        print(f"   Load Reranker: {settings.RERANKER_PATH}")
        self.reranker = CrossEncoder(settings.RERANKER_PATH, local_files_only=True)
        
        # 4. 初始化 OpenAI 客户端
        self.llm_client = OpenAI(
            api_key=settings.AI_API_KEY,
            base_url=settings.AI_BASE_URL
        )
        
        # 5. 确保 BM25 和 Chroma 同步 (系统启动时检查)
        # 如果 BM25 是空的但 Chroma 有数据，尝试重建(此处略，为加速启动暂不自动全量重建)
        
        print("✅ [Core] 服务初始化完成")

    def _rrf_fusion(self, vector_results, bm25_results, k=60):
        """倒数排名融合算法 (RRF)"""
        fused_scores = {}
        
        # 归一化数据结构
        # vector_results: {'documents': [[...]], 'metadatas': [[...]]}
        # bm25_results: ([docs], [metas])
        
        vec_docs = vector_results['documents'][0]
        vec_metas = vector_results['metadatas'][0]
        
        bm25_docs = bm25_results[0]
        bm25_metas = bm25_results[1]
        
        # 建立内容到元数据的映射
        content_map = {}
        
        # 积分
        for rank, doc in enumerate(vec_docs):
            if doc not in fused_scores: fused_scores[doc] = 0
            fused_scores[doc] += 1 / (k + rank + 1)
            content_map[doc] = vec_metas[rank]
            
        for rank, doc in enumerate(bm25_docs):
            if doc not in fused_scores: fused_scores[doc] = 0
            fused_scores[doc] += 1 / (k + rank + 1)
            content_map[doc] = bm25_metas[rank]
            
        # 排序
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs, content_map

    def search(self, query: str, top_k: int = 3) -> tuple[list, list, list]:
        """
        混合检索入口：Vector(20) + BM25(20) -> RRF -> Reranker -> TopK
        """
        # 1. 向量检索
        query_vec = self.embed_model.encode([query]).tolist()
        vec_res = self.collection.query(query_embeddings=query_vec, n_results=settings.DEFAULT_TOP_K)
        
        # 2. BM25 检索
        bm25_docs, bm25_metas = bm25_retriever.search(query, top_k=settings.DEFAULT_TOP_K)
        
        # 3. RRF 融合
        sorted_candidates, content_map = self._rrf_fusion(vec_res, (bm25_docs, bm25_metas))
        
        # 取前 20 个做精排
        candidates = sorted_candidates[:20]
        
        if not candidates:
            return [], [], []

        # 4. Rerank 重排序
        rerank_inputs = [[query, doc[0]] for doc in candidates]
        scores = self.reranker.predict(rerank_inputs)
        
        # 组合结果 (Score, Doc, Meta)
        final_results = []
        for i, score in enumerate(scores):
            doc_content = rerank_inputs[i][1]
            final_results.append({
                "content": doc_content,
                "meta": content_map[doc_content],
                "score": float(score)
            })
            
        # 按 Reranker 分数排序并截断
        final_results.sort(key=lambda x: x["score"], reverse=True)
        final_results = final_results[:top_k]
        
        return (
            [x["content"] for x in final_results],
            [x["meta"] for x in final_results],
            [x["score"] for x in final_results]
        )

    def chat(self, query: str, history: list, top_k: int = 3):
        """
        对话主逻辑：Search -> Prompt -> LLM
        """
        # (可选) 这里可以加 Query Rewrite 逻辑
        
        # 执行搜索
        docs, metas, scores = self.search(query, top_k)
        
        # 构造 Prompt
        if not docs:
            return {"answer": "知识库中未找到相关信息。", "docs": [], "metas": [], "scores": []}
            
        context_str = "\n\n".join([f"片段{i+1}: {d}" for i, d in enumerate(docs)])
        
        system_prompt = f"""你是一个智能制造领域的专家助手。请基于以下参考资料回答用户问题。
        如果参考资料不足以回答，请明确告知。
        
        【参考资料】
        {context_str}
        """
        
        # 调用 LLM
        response = self.llm_client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.3
        )
        
        return {
            "answer": response.choices[0].message.content,
            "docs": docs,
            "metas": metas,
            "scores": scores
        }

    def process_upload(self, temp_path: str, filename: str, use_ocr: bool):
        """
        文件处理流程：提取 -> 切片 -> 存向量库 -> 存BM25
        """
        # 1. 提取
        pages = ocr_engine.extract_text(temp_path, force_ocr=use_ocr)
        if not pages: return 0
        
        # 2. 切片
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=settings.CHUNK_SIZE,
        #     chunk_overlap=settings.CHUNK_OVERLAP
        # )
        # 2. 切片 (代码不变，但底层调用的库变了)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""] # 显式指定中文分隔符更稳
        )

        docs_to_add = []
        metas_to_add = []
        ids_to_add = []
        
        for page_num, text in pages:
            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                docs_to_add.append(chunk)
                metas_to_add.append({
                    "source": filename,
                    "page": page_num
                })
                ids_to_add.append(f"{filename}_p{page_num}_c{i}")
        
        # 3. 存入 Chroma (自动计算向量)
        if docs_to_add:
            # 这里的 batch 处理通常由 Chroma 内部处理，但量大建议分批
            embeddings = self.embed_model.encode(docs_to_add).tolist()
            self.collection.upsert(
                documents=docs_to_add,
                embeddings=embeddings,
                metadatas=metas_to_add,
                ids=ids_to_add
            )
            
            # 4. 存入 BM25
            bm25_retriever.add_documents(docs_to_add, metas_to_add)
            
        return len(docs_to_add)

# 初始化全局单例
rag_service = RAGService()