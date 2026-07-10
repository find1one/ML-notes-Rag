"""
检索优化模块
"""

import hashlib
import logging
import re
from typing import List, Dict, Any
import time

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""
    
    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        """
        初始化检索优化模块
        
        Args:
            vectorstore: FAISS向量存储
            chunks: 文档块列表
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.chunk_by_id = {self._document_id(chunk): chunk for chunk in chunks}
        self.setup_retrievers()
        self.last_metrics = {}

    def setup_retrievers(self):
        """设置向量检索器和BM25检索器"""
        logger.info("正在设置检索器...")

        # 向量检索器
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # BM25检索器
        self.bm25_retriever = BM25Retriever.from_documents(
            [self._bm25_document(chunk) for chunk in self.chunks],
            k=5
        )



        logger.info("检索器设置完成")
    
    def hybrid_search(self, query: str, top_k: int = 3, candidate_k: int = None) -> List[Document]:
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF重排

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """
        candidate_k = max(candidate_k or 0, top_k * 5, 80)
        # 分别获取向量检索和BM25检索结果
        retrieval_start = time.perf_counter()

        faiss_start = time.perf_counter()
        vector_docs = self.vectorstore.similarity_search(query, k=candidate_k)
        faiss_ms = int((time.perf_counter() - faiss_start) * 1000)

        bm25_start = time.perf_counter()
        self.bm25_retriever.k = candidate_k
        bm25_docs = self.bm25_retriever.invoke(query)
        bm25_docs = [self.chunk_by_id.get(self._document_id(doc), doc) for doc in bm25_docs]
        bm25_ms = int((time.perf_counter() - bm25_start) * 1000)

        # 使用RRF重排
        rrf_start = time.perf_counter()
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs, query=query)
        rrf_ms = int((time.perf_counter() - rrf_start) * 1000)

        retrieval_ms = int((time.perf_counter() - retrieval_start) * 1000)
        self.last_metrics = {
            "faiss_ms": faiss_ms,
            "bm25_ms": bm25_ms,
            "rrf_ms": rrf_ms,
            "total_retrieval_ms": retrieval_ms,
            "candidate_k": candidate_k,
        }

        logger.info(
        "Hybrid search timing: faiss_ms=%s bm25_ms=%s rrf_ms=%s retrieval_total_ms=%s",
        faiss_ms,
        bm25_ms,
        rrf_ms,
        retrieval_ms,
        )

        return reranked_docs[:top_k]
    
    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        带元数据过滤的检索
        
        Args:
            query: 查询文本
            filters: 元数据过滤条件
            top_k: 返回结果数量
            
        Returns:
            过滤后的文档列表
        """
        # 先进行混合检索，获取更多候选
        docs = self.hybrid_search(query, top_k * 3)
        
        # 应用元数据过滤
        filtered_docs = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if key in doc.metadata:
                    if isinstance(value, list):
                        if doc.metadata[key] not in value:
                            match = False
                            break
                    else:
                        if doc.metadata[key] != value:
                            match = False
                            break
                else:
                    match = False
                    break
            
            if match:
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break
        
        if len(filtered_docs) >= top_k:
            return filtered_docs
        # A strict post-filter can discard every useful result when the candidate
        # pool is sparse. Keep topic matches first, then return the best fallback.
        filtered_ids = {self._document_id(doc) for doc in filtered_docs}
        return (filtered_docs + [doc for doc in docs if self._document_id(doc) not in filtered_ids])[:top_k]

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60, query: str = "") -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            k: RRF参数，用于平滑排名

        Returns:
            重排后的文档列表
        """
        doc_scores = {}
        doc_objects = {}

        # 计算向量检索结果的RRF分数
        for rank, doc in enumerate(vector_docs):
            doc_id = self._document_id(doc)
            doc_objects[doc_id] = doc

            # RRF公式: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"向量检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 计算BM25检索结果的RRF分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = self._document_id(doc)
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"BM25检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 按最终RRF分数排序
        for doc_id, doc in doc_objects.items():
            doc_scores[doc_id] += self._metadata_boost(query, doc)
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                # 将RRF分数添加到文档元数据中
                doc.metadata['rrf_score'] = final_score
                reranked_docs.append(doc)
                logger.debug(f"最终排序 - 文档: {doc.page_content[:50]}... 最终RRF分数: {final_score:.4f}")

        logger.info(f"RRF重排完成: 向量检索{len(vector_docs)}个文档, BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档")

        return reranked_docs

    @staticmethod
    def _document_id(doc: Document) -> str:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        source = doc.metadata.get("relative_path", doc.metadata.get("source", ""))
        return hashlib.sha256(f"{source}|{doc.page_content}".encode("utf-8")).hexdigest()

    @staticmethod
    def _metadata_boost(query: str, doc: Document) -> float:
        query_terms = {term for term in re.findall(r"[a-z0-9][a-z0-9_-]+", query.lower()) if len(term) >= 3}
        if not query_terms:
            return 0.0
        metadata_text = " ".join([
            str(doc.metadata.get("title", "")),
            str(doc.metadata.get("relative_path", "")),
            str(doc.metadata.get("topic", "")),
        ]).lower()
        return min(0.10, 0.02 * sum(term in metadata_text for term in query_terms))

    @staticmethod
    def _bm25_document(doc: Document) -> Document:
        """Index title/path tokens without leaking synthetic text into generation."""
        metadata = dict(doc.metadata)
        title = str(metadata.get("title", ""))
        path = str(metadata.get("relative_path", "")).replace("/", " ").replace("-", " ").replace("_", " ")
        topic = str(metadata.get("topic", ""))
        prefix = " ".join([title, title, path, topic])
        return Document(page_content=f"{prefix}\n{doc.page_content}", metadata=metadata)
