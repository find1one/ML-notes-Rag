from langchain_core.documents import Document

from evaluate_parent_child import (
    ExperimentChildRetriever,
    ParentChildRetriever,
    _evaluation_summary,
    _metrics_gate,
    build_contextual_children,
    build_parent_child_corpus,
    compare_case_reports,
)


def _parent(chunk_id, text):
    return Document(
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "relative_path": f"{chunk_id}.md",
            "section_path": chunk_id,
            "topic": "Regression",
        },
    )


def test_child_chunks_use_200_size_20_overlap_and_map_to_parent():
    text = "".join(chr(0x4E00 + index) for index in range(450))

    parent_by_id, children = build_parent_child_corpus([_parent("parent-1", text)])

    assert list(parent_by_id) == ["parent-1"]
    assert len(children) == 3
    assert max(len(child.page_content) for child in children) <= 200
    assert children[0].page_content[-20:] == children[1].page_content[:20]
    assert children[1].page_content[-20:] == children[2].page_content[:20]
    assert {child.metadata["parent_chunk_id"] for child in children} == {"parent-1"}
    assert all(child.metadata["experiment_role"] == "child" for child in children)


def test_contextual_children_use_deterministic_title_section_content_text():
    _, raw_children = build_parent_child_corpus(
        [
            Document(
                page_content="raw child content",
                metadata={
                    "chunk_id": "parent-1",
                    "title": "Decision Trees",
                    "section_path": "Decision Trees > Entropy",
                },
            )
        ]
    )

    contextual_children = build_contextual_children(raw_children)

    assert raw_children[0].page_content == "raw child content"
    assert contextual_children[0].page_content == (
        "Document: Decision Trees\n"
        "Section: Decision Trees > Entropy\n"
        "Content: raw child content"
    )
    assert contextual_children[0].metadata == raw_children[0].metadata
    assert contextual_children[0] is not raw_children[0]


class FakeChildRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.requests = []
        self.last_metrics = {"candidate_k": 80}

    def hybrid_search(self, query, top_k, candidate_k=None):
        self.requests.append((top_k, candidate_k))
        return self.documents[:top_k]


def test_retriever_deduplicates_parents_in_highest_child_rank_order():
    parents = {
        "a": _parent("a", "parent a"),
        "b": _parent("b", "parent b"),
    }
    children = [
        Document(page_content="a1", metadata={"parent_chunk_id": "a"}),
        Document(page_content="a2", metadata={"parent_chunk_id": "a"}),
        Document(page_content="b1", metadata={"parent_chunk_id": "b"}),
    ]
    child_retriever = FakeChildRetriever(children)
    retriever = ParentChildRetriever(child_retriever, parents, len(children))

    results = retriever.hybrid_search("query", top_k=2)

    assert [result.metadata["chunk_id"] for result in results] == ["a", "b"]


def test_retriever_expands_child_results_to_find_enough_unique_parents():
    parents = {
        "a": _parent("a", "parent a"),
        "b": _parent("b", "parent b"),
        "c": _parent("c", "parent c"),
    }
    children = [
        Document(page_content=f"a{index}", metadata={"parent_chunk_id": "a"})
        for index in range(20)
    ] + [
        Document(page_content="b", metadata={"parent_chunk_id": "b"}),
        Document(page_content="c", metadata={"parent_chunk_id": "c"}),
    ]
    child_retriever = FakeChildRetriever(children)
    retriever = ParentChildRetriever(child_retriever, parents, len(children))

    results = retriever.hybrid_search("query", top_k=3)

    assert child_retriever.requests == [(22, 80)]
    assert [result.metadata["chunk_id"] for result in results] == ["a", "b", "c"]


def test_retriever_keeps_candidate_k_fixed_at_80():
    parents = {"a": _parent("a", "parent a")}
    children = [Document(page_content="a1", metadata={"parent_chunk_id": "a"})]
    child_retriever = FakeChildRetriever(children)

    ParentChildRetriever(child_retriever, parents, len(children)).hybrid_search(
        "query", top_k=1
    )

    assert child_retriever.requests == [(1, 80)]


def test_experiment_child_retriever_uses_80_candidates_per_route(monkeypatch):
    requested = {}

    class VectorStore:
        def similarity_search(self, query, k):
            requested["faiss"] = k
            return []

    class BM25:
        k = 5

        def invoke(self, query):
            requested["bm25"] = self.k
            return []

    retriever = ExperimentChildRetriever.__new__(ExperimentChildRetriever)
    retriever.vectorstore = VectorStore()
    retriever.bm25_retriever = BM25()
    retriever.chunk_by_id = {}

    retriever.hybrid_search("query", top_k=80, candidate_k=80)

    assert requested == {"faiss": 80, "bm25": 80}
    assert retriever.last_metrics["candidate_k"] == 80


def test_case_comparison_reports_rank_and_path_changes():
    pc0 = [
        {
            "name": "case",
            "evaluable": True,
            "top1_hit": True,
            "topk_hit": True,
            "topic_hit": True,
            "source_rank": 1,
            "topic_rank": 2,
            "source_rr": 1.0,
            "topic_rr": 0.5,
            "paths": ["expected.md", "other.md"],
        }
    ]
    pc1 = [
        {
            **pc0[0],
            "top1_hit": False,
            "source_rank": 2,
            "source_rr": 0.5,
            "paths": ["other.md", "expected.md"],
        }
    ]

    changes = compare_case_reports(pc0, pc1)

    assert changes == [
        {
            "name": "case",
            "evaluable": True,
            "outcome_changed": True,
            "ranking_changed": True,
            "paths_changed": True,
            "pc0": pc0[0],
            "pc1": pc1[0],
        }
    ]


def test_metrics_gate_rejects_improvement_from_only_one_case():
    pc0 = {
        "source_mrr_at_3": 0.8,
        "topic_mrr_at_3": 0.9,
        "top3_source_accuracy": 0.9,
    }
    pc1 = {
        "source_mrr_at_3": 0.81,
        "topic_mrr_at_3": 0.9,
        "top3_source_accuracy": 0.9,
    }

    assert not _metrics_gate(pc0, pc1, top_k=3, source_rank_change_count=1)
    assert _metrics_gate(pc0, pc1, top_k=3, source_rank_change_count=2)


def test_experiment_summary_includes_source_and_topic_mrr_at_3():
    results = [
        {
            "evaluable": True,
            "top1_hit": True,
            "topk_hit": True,
            "topic_hit": True,
            "source_rr": 1.0,
            "topic_rr": 0.5,
        },
        {
            "evaluable": True,
            "top1_hit": False,
            "topk_hit": True,
            "topic_hit": False,
            "source_rr": 1 / 3,
            "topic_rr": 0.0,
        },
    ]

    summary = _evaluation_summary(results, top_k=3)

    assert summary["source_mrr_at_3"] == (1.0 + 1 / 3) / 2
    assert summary["topic_mrr_at_3"] == 0.25
