import pytest

from langchain_core.documents import Document

from evaluate_parent_child import (
    ExperimentChildRetriever,
    ParentChildRetriever,
    SourceAwareMaxRetriever,
    SourceAwareTop2Retriever,
    _evaluation_summary,
    _length_bias_diagnostic,
    _metrics_gate,
    _source_size_stats,
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


def test_source_aware_max_returns_best_parent_once_per_source():
    parents = {
        "a1": _parent("a1", "weaker parent from source a"),
        "a2": _parent("a2", "best parent from source a"),
        "b1": _parent("b1", "parent from source b"),
    }
    parents["a1"].metadata["relative_path"] = "source-a.md"
    parents["a2"].metadata["relative_path"] = "source-a.md"
    parents["b1"].metadata["relative_path"] = "source-b.md"
    children = [
        Document(page_content="b", metadata={"parent_chunk_id": "b1", "rrf_score": 0.8}),
        Document(page_content="a1", metadata={"parent_chunk_id": "a1", "rrf_score": 0.7}),
        Document(page_content="a2", metadata={"parent_chunk_id": "a2", "rrf_score": 0.9}),
        Document(page_content="a1 weaker", metadata={"parent_chunk_id": "a1", "rrf_score": 0.4}),
    ]
    child_retriever = FakeChildRetriever(children)
    retriever = SourceAwareMaxRetriever(child_retriever, parents, len(children))

    results = retriever.hybrid_search("query", top_k=3)

    assert [result.metadata["chunk_id"] for result in results] == ["a2", "b1"]
    assert [result.metadata["relative_path"] for result in results] == [
        "source-a.md",
        "source-b.md",
    ]
    assert retriever.last_metrics["unique_parents_found"] == 3
    assert retriever.last_metrics["unique_sources_found"] == 2


def test_source_aware_top2_uses_weighted_child_and_parent_support():
    parents = {
        "a1": _parent("a1", "best supported parent from source a"),
        "a2": _parent("a2", "second parent from source a"),
        "b1": _parent("b1", "single parent from source b"),
    }
    parents["a1"].metadata["relative_path"] = "source-a.md"
    parents["a2"].metadata["relative_path"] = "source-a.md"
    parents["b1"].metadata["relative_path"] = "source-b.md"
    children = [
        Document(
            page_content="b1",
            metadata={"chunk_id": "b1-c1", "parent_chunk_id": "b1", "rrf_score": 0.9},
        ),
        Document(
            page_content="a1 first",
            metadata={"chunk_id": "a1-c1", "parent_chunk_id": "a1", "rrf_score": 0.6},
        ),
        Document(
            page_content="a1 second",
            metadata={"chunk_id": "a1-c2", "parent_chunk_id": "a1", "rrf_score": 0.5},
        ),
        Document(
            page_content="a2 first",
            metadata={"chunk_id": "a2-c1", "parent_chunk_id": "a2", "rrf_score": 0.7},
        ),
        Document(
            page_content="a2 second",
            metadata={"chunk_id": "a2-c2", "parent_chunk_id": "a2", "rrf_score": 0.1},
        ),
    ]
    retriever = SourceAwareTop2Retriever(
        FakeChildRetriever(children),
        parents,
        len(children),
    )

    results = retriever.hybrid_search("query", top_k=2)

    assert [result.metadata["chunk_id"] for result in results] == ["a1", "b1"]
    evidence = retriever.last_aggregation["ranked_sources"]
    assert evidence[0]["source"] == "source-a.md"
    assert evidence[0]["source_score"] == 1.075
    assert evidence[0]["parents"][0]["parent_score"] == 0.85
    assert evidence[0]["parents"][1]["parent_score"] == 0.75
    assert evidence[0]["parents"][0]["children"][1]["weighted_contribution"] == 0.25
    assert evidence[0]["parents"][1]["weighted_contribution"] == pytest.approx(0.225)
    assert retriever.last_metrics["parents_with_second_child"] == 2
    assert retriever.last_metrics["sources_with_second_parent"] == 1


def test_source_aware_top2_treats_missing_or_duplicate_second_evidence_as_zero():
    parents = {"a": _parent("a", "parent a")}
    child = Document(
        page_content="only child",
        metadata={"chunk_id": "a-c1", "parent_chunk_id": "a", "rrf_score": 0.8},
    )
    retriever = SourceAwareTop2Retriever(
        FakeChildRetriever([child, child]),
        parents,
        child_count=2,
    )

    results = retriever.hybrid_search("query", top_k=1)

    assert [result.metadata["chunk_id"] for result in results] == ["a"]
    source = retriever.last_aggregation["ranked_sources"][0]
    assert source["source_score"] == 0.8
    assert len(source["parents"]) == 1
    assert len(source["parents"][0]["children"]) == 1
    assert retriever.last_metrics["parents_with_second_child"] == 0
    assert retriever.last_metrics["sources_with_second_parent"] == 0


def test_source_aware_top2_has_deterministic_source_tie_break():
    parents = {
        "z": _parent("z", "source z"),
        "a": _parent("a", "source a"),
    }
    children = [
        Document(
            page_content="z",
            metadata={"chunk_id": "z-c", "parent_chunk_id": "z", "rrf_score": 0.5},
        ),
        Document(
            page_content="a",
            metadata={"chunk_id": "a-c", "parent_chunk_id": "a", "rrf_score": 0.5},
        ),
    ]
    retriever = SourceAwareTop2Retriever(
        FakeChildRetriever(children),
        parents,
        len(children),
    )

    results = retriever.hybrid_search("query", top_k=2)

    assert [result.metadata["relative_path"] for result in results] == ["z.md", "a.md"]


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
            "change_type": "regressed",
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


def test_source_size_stats_and_length_bias_diagnostic_are_deterministic():
    parents = {
        "long": _parent("long", "long source"),
        "medium": _parent("medium", "medium source"),
        "short": _parent("short", "short source"),
        "tiny": _parent("tiny", "tiny source"),
    }
    children = [
        Document(page_content=str(index), metadata={"parent_chunk_id": "long"})
        for index in range(4)
    ] + [
        Document(page_content="m", metadata={"parent_chunk_id": "medium"}),
        Document(page_content="s", metadata={"parent_chunk_id": "short"}),
        Document(page_content="t", metadata={"parent_chunk_id": "tiny"}),
    ]
    sizes = _source_size_stats(parents, children)
    changes = [
        {
            "name": "long-improvement-1",
            "evaluable": True,
            "pc2": {"source_rank": None},
            "h1": {"source_rank": 3, "expected_path": "long.md"},
        },
        {
            "name": "long-improvement-2",
            "evaluable": True,
            "pc2": {"source_rank": 2},
            "h1": {"source_rank": 1, "expected_path": "long.md"},
        },
    ]

    diagnostic = _length_bias_diagnostic(changes, "pc2", "h1", sizes)

    assert sizes["long.md"] == {"parent_count": 1, "child_count": 4}
    assert diagnostic["highest_child_count_quartile_sources"] == ["long.md"]
    assert diagnostic["high_quartile_improvement_count"] == 2
    assert diagnostic["manual_review"] is True


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
