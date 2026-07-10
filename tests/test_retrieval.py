from langchain_core.documents import Document

from rag_modules.retrieval_optimization import RetrievalOptimizationModule


def _doc(chunk_id, title, text):
    return Document(page_content=text, metadata={"chunk_id": chunk_id, "title": title, "relative_path": title + ".md", "topic": "Classification"})


def test_rrf_uses_stable_chunk_ids_and_metadata_boost():
    first = _doc("first", "Decision Tree", "shared")
    second = _doc("second", "Overview", "shared")
    retriever = RetrievalOptimizationModule.__new__(RetrievalOptimizationModule)
    ranked = retriever._rrf_rerank([second], [first], query="decision tree classification")
    assert ranked[0].metadata["chunk_id"] == "first"


def test_topic_filter_keeps_fallback_when_candidates_are_sparse(monkeypatch):
    retriever = RetrievalOptimizationModule.__new__(RetrievalOptimizationModule)
    other = _doc("other", "Other", "content")
    monkeypatch.setattr(retriever, "hybrid_search", lambda query, top_k: [other])
    assert retriever.metadata_filtered_search("query", {"topic": "Regression"}, top_k=1) == [other]
