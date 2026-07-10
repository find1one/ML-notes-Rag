"""Deterministic RAG execution path used by HTTP endpoints."""

from dataclasses import dataclass
import logging
import re
import time
from typing import Any, Iterable, List, Optional

from main import MLNotesRAGSystem
from rag_modules import DataPreparationModule

logger = logging.getLogger(__name__)


class RAGUnavailableError(RuntimeError):
    pass


@dataclass
class RAGPrepared:
    question: str
    route_type: str
    topic: Optional[str]
    retrieval_query: str
    docs: List[Any]
    sources: List[dict]
    metrics: dict[str, int]
    gate_decision: str
    rejection_reason: Optional[str] = None


@dataclass
class RAGExecution:
    answer: str
    route_type: str
    topic: Optional[str]
    retrieval_query: Optional[str]
    sources: List[dict]
    metrics: dict[str, int]
    gate_decision: str = "passed"
    terminal_event: str = "done"
    rejection_reason: Optional[str] = None


class RAGService:
    LIST_KEYWORDS = ("哪些", "有什么", "列表", "目录", "包括", "列出")
    DETAIL_KEYWORDS = ("怎么", "如何", "步骤", "原理", "假设", "优缺点", "为什么", "公式", "是什么")
    TERM_MAP = {
        "线性回归": "linear regression OLS ordinary least squares",
        "变量选择": "variable selection feature selection p-value backward elimination",
        "多项式回归": "polynomial regression",
        "支持向量回归": "support vector regression SVR",
        "逻辑回归": "logistic regression",
        "支持向量机": "support vector machines SVM",
        "决策树": "decision tree",
        "随机森林": "random forest ensemble decision trees",
        "朴素贝叶斯": "naive bayes classifier probability",
        "隐马尔可夫": "hidden markov models HMM",
        "聚类": "clustering",
        "k-means": "K-means clustering algorithm",
        "高斯混合": "gaussian mixture model expectation maximization",
        "层次聚类": "hierarchical clustering",
        "关联规则": "association rule learning support confidence lift",
        "降维": "dimensionality reduction",
        "主成分": "principal component analysis PCA",
        "强化学习": "reinforcement learning",
        "自然语言处理": "natural language processing NLP",
        "深度学习": "deep learning neural networks",
        "推荐系统": "recommendation engines",
        "时间序列": "time series",
        "状态空间": "state space models time series",
        "约束满足": "constraint satisfaction problems CSP",
        "统计": "statistics",
        "线性代数": "linear algebra",
    }

    def __init__(self, rag_system: MLNotesRAGSystem):
        self.rag_system = rag_system

    @property
    def ready(self) -> bool:
        return bool(self.rag_system.retrieval_module and self.rag_system.generation_module)

    def execute(self, question: str) -> RAGExecution:
        prepared = self.prepare(question)
        if prepared.gate_decision != "passed":
            return self._rejected_execution(prepared)

        generation_start = time.perf_counter()
        answer = self.generate_answer(prepared)
        prepared.metrics["generation_ms"] = self._elapsed_ms(generation_start)
        return RAGExecution(
            answer=answer.strip(),
            route_type=prepared.route_type,
            topic=prepared.topic,
            retrieval_query=prepared.retrieval_query,
            sources=prepared.sources,
            metrics=prepared.metrics,
            gate_decision=prepared.gate_decision,
            terminal_event="done",
        )

    def prepare(self, question: str) -> RAGPrepared:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        if not self.ready:
            raise RAGUnavailableError("RAG generation service is not ready")

        metrics: dict[str, int] = {}
        route_start = time.perf_counter()
        route_type = self._route(question)
        metrics["route_ms"] = self._elapsed_ms(route_start)

        expand_start = time.perf_counter()
        topic = DataPreparationModule.match_topic_from_query(question)
        retrieval_query = self._expand_query(question)
        metrics["expand_ms"] = self._elapsed_ms(expand_start)

        retrieval_start = time.perf_counter()
        if route_type == "list" and topic:
            docs = self.rag_system.data_module.filter_documents_by_topic(topic)
            metrics["total_retrieval_ms"] = self._elapsed_ms(retrieval_start)
        elif topic:
            docs = self.rag_system.retrieval_module.metadata_filtered_search(
                retrieval_query, {"topic": topic}, top_k=self.rag_system.config.top_k
            )
            metrics.update(self.rag_system.retrieval_module.last_metrics)
        else:
            docs = self.rag_system.retrieval_module.hybrid_search(
                retrieval_query,
                top_k=self.rag_system.config.top_k,
                candidate_k=self.rag_system.config.candidate_k,
            )
            metrics.update(self.rag_system.retrieval_module.last_metrics)

        sources = self.format_sources(docs)
        gate_decision, rejection_reason = self._gate_evidence(retrieval_query, topic, docs, sources)
        metrics["source_count"] = len(sources)
        return RAGPrepared(
            question=question,
            route_type=route_type,
            topic=topic,
            retrieval_query=retrieval_query,
            docs=docs,
            sources=sources,
            metrics=metrics,
            gate_decision=gate_decision,
            rejection_reason=rejection_reason,
        )

    def generate_answer(self, prepared: RAGPrepared) -> str:
        if prepared.route_type == "list":
            return self.rag_system.generation_module.generate_list_answer(
                prepared.question,
                self.rag_system.data_module.get_parent_documents(prepared.docs)
                if prepared.docs and prepared.docs[0].metadata.get("doc_type") == "child"
                else prepared.docs,
            )
        if prepared.route_type == "detail":
            answer = self.rag_system.generation_module.generate_step_by_step_answer(prepared.question, prepared.docs)
        else:
            answer = self.rag_system.generation_module.generate_basic_answer(prepared.question, prepared.docs)
        if isinstance(answer, str) and answer.strip():
            return answer
        logger.warning("LLM returned an empty response; route_type=%s", prepared.route_type)
        raise RAGUnavailableError("LLM returned an empty response")

    def stream_answer(self, prepared: RAGPrepared) -> Iterable[str]:
        if prepared.route_type == "list":
            yield self.generate_answer(prepared)
            return
        if prepared.route_type == "detail":
            yield from self.rag_system.generation_module.generate_step_by_step_answer_stream(prepared.question, prepared.docs)
            return
        yield from self.rag_system.generation_module.generate_basic_answer_stream(prepared.question, prepared.docs)

    def _rejected_execution(self, prepared: RAGPrepared) -> RAGExecution:
        return RAGExecution(
            answer="当前笔记中没有找到足够依据。",
            route_type=prepared.route_type,
            topic=prepared.topic,
            retrieval_query=prepared.retrieval_query,
            sources=prepared.sources,
            metrics=prepared.metrics,
            gate_decision=prepared.gate_decision,
            terminal_event="rejected",
            rejection_reason=prepared.rejection_reason,
        )

    def _route(self, question: str) -> str:
        if any(keyword in question for keyword in self.LIST_KEYWORDS):
            return "list"
        if any(keyword in question for keyword in self.DETAIL_KEYWORDS):
            return "detail"
        return "general"

    def _expand_query(self, question: str) -> str:
        terms = [english for chinese, english in self.TERM_MAP.items() if chinese.lower() in question.lower()]
        topic = DataPreparationModule.match_topic_from_query(question)
        if topic:
            terms.append(topic)
        return f"{question} {' '.join(dict.fromkeys(terms))}".strip()

    def _gate_evidence(
        self,
        retrieval_query: str,
        topic: Optional[str],
        docs: List[Any],
        sources: List[dict],
    ) -> tuple[str, Optional[str]]:
        non_empty_docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
        if not non_empty_docs:
            return "rejected", "no_non_empty_chunks"
        top_doc = non_empty_docs[0]
        if topic and str(top_doc.metadata.get("topic", "")).lower() == topic.lower():
            return "passed", None
        query_terms = self._english_terms(retrieval_query)
        if query_terms:
            evidence_text = " ".join([
                str(top_doc.metadata.get("title", "")),
                str(top_doc.metadata.get("topic", "")),
                str(top_doc.metadata.get("relative_path", "")),
                top_doc.page_content[:800],
            ]).lower()
            if any(term in evidence_text for term in query_terms):
                return "passed", None
        if sources:
            return "rejected", "top_source_does_not_match_query_terms"
        return "rejected", "no_sources"

    @staticmethod
    def _english_terms(text: str) -> set[str]:
        stop_words = {
            "what", "why", "how", "the", "and", "for", "with", "from", "this", "that",
            "which", "when", "where", "are", "does", "是什么", "怎么", "如何",
        }
        return {
            term
            for term in re.findall(r"[a-z][a-z0-9_-]{2,}", text.lower())
            if term not in stop_words
        }

    @staticmethod
    def format_sources(docs: List[Any]) -> List[dict]:
        sources = []
        for index, doc in enumerate(docs, start=1):
            excerpt = re.sub(r"\s+", " ", doc.page_content.strip())[:280]
            sources.append({
                "id": f"S{index}",
                "title": doc.metadata.get("title", "Untitled"),
                "topic": doc.metadata.get("topic", "Unknown topic"),
                "section": doc.metadata.get("section_path", "Unknown section"),
                "path": doc.metadata.get("relative_path", doc.metadata.get("source", "")),
                "score": doc.metadata.get("rrf_score"),
                "excerpt": excerpt,
            })
        return sources

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)
