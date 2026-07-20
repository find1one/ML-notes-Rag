"""Isolated parent-child retrieval experiment for the ML notes corpus."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Sequence, Tuple

sys.path.append(str(Path(__file__).parent))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DEFAULT_CONFIG
from evaluate_retrieval import EVAL_CASES
from rag_modules import DataPreparationModule, IndexConstructionModule, RetrievalOptimizationModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated Parent-Child retrieval ablations.")
    parser.add_argument("--child-size", type=int, default=200, help="Child chunk size in characters.")
    parser.add_argument("--child-overlap", type=int, default=20, help="Child overlap in characters.")
    parser.add_argument("--top-k", type=int, default=3, help="Evaluation top-k.")
    parser.add_argument("--online", action="store_true", help="Allow Hugging Face network access.")
    parser.add_argument(
        "--experiment",
        choices=("pc1", "pc2", "h1"),
        default="pc1",
        help="Candidate experiment; h1 is compared directly with PC2.",
    )
    parser.add_argument("--report-json", type=Path, default=None, help="Optionally write the full A/B report as JSON.")
    return parser.parse_args()


def build_parent_child_corpus(
    parents: Sequence[Document],
    child_size: int = 200,
    child_overlap: int = 20,
) -> Tuple[Dict[str, Document], List[Document]]:
    if child_size <= 0:
        raise ValueError("child_size must be greater than zero.")
    if child_overlap < 0 or child_overlap >= child_size:
        raise ValueError("child_overlap must be non-negative and smaller than child_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    parent_by_id: Dict[str, Document] = {}
    children: List[Document] = []

    for parent_index, parent in enumerate(parents):
        parent_id = str(parent.metadata.get("chunk_id") or _fallback_parent_id(parent, parent_index))
        parent_metadata = dict(parent.metadata)
        parent_metadata.update({"parent_chunk_id": parent_id, "experiment_role": "parent"})
        parent_by_id[parent_id] = Document(page_content=parent.page_content, metadata=parent_metadata)

        for child_index, child_text in enumerate(splitter.split_text(parent.page_content)):
            child_id = hashlib.md5(
                f"{parent_id}|{child_index}|{child_text}".encode("utf-8")
            ).hexdigest()
            child_metadata = dict(parent.metadata)
            child_metadata.update(
                {
                    "chunk_id": child_id,
                    "parent_chunk_id": parent_id,
                    "doc_type": "experiment_child",
                    "experiment_role": "child",
                    "child_index": child_index,
                    "chunk_size": len(child_text),
                }
            )
            children.append(Document(page_content=child_text, metadata=child_metadata))

    return parent_by_id, children


def build_contextual_children(children: Sequence[Document]) -> List[Document]:
    """Clone children with deterministic structural context as retrieval text."""
    contextual_children = []
    for child in children:
        title = str(child.metadata.get("title", ""))
        section_path = str(child.metadata.get("section_path", ""))
        contextual_children.append(
            Document(
                page_content=(
                    f"Document: {title}\n"
                    f"Section: {section_path}\n"
                    f"Content: {child.page_content}"
                ),
                metadata=dict(child.metadata),
            )
        )
    return contextual_children


class ExperimentChildRetriever(RetrievalOptimizationModule):
    """Hybrid child retriever whose candidate pool is isolated from production logic."""

    def hybrid_search(self, query: str, top_k: int = 3, candidate_k: int = 80) -> List[Document]:
        fixed_candidate_k = candidate_k or 80
        retrieval_start = time.perf_counter()

        faiss_start = time.perf_counter()
        vector_docs = self.vectorstore.similarity_search(query, k=fixed_candidate_k)
        faiss_ms = int((time.perf_counter() - faiss_start) * 1000)

        bm25_start = time.perf_counter()
        self.bm25_retriever.k = fixed_candidate_k
        bm25_docs = self.bm25_retriever.invoke(query)
        bm25_docs = [
            self.chunk_by_id.get(self._document_id(doc), doc)
            for doc in bm25_docs
        ]
        bm25_ms = int((time.perf_counter() - bm25_start) * 1000)

        rrf_start = time.perf_counter()
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs, query=query)
        rrf_ms = int((time.perf_counter() - rrf_start) * 1000)
        self.last_metrics = {
            "faiss_ms": faiss_ms,
            "bm25_ms": bm25_ms,
            "rrf_ms": rrf_ms,
            "total_retrieval_ms": int((time.perf_counter() - retrieval_start) * 1000),
            "candidate_k": fixed_candidate_k,
        }
        return reranked_docs[:top_k]


def _fallback_parent_id(parent: Document, parent_index: int) -> str:
    seed = "|".join(
        [
            str(parent.metadata.get("relative_path", "")),
            str(parent.metadata.get("section_path", "")),
            str(parent_index),
            parent.page_content,
        ]
    )
    return hashlib.md5(seed.encode("utf-8")).hexdigest()


class ParentChildRetriever:
    """Retrieve child chunks, then return unique parents in first-child rank order."""

    def __init__(
        self,
        child_retriever: RetrievalOptimizationModule,
        parent_by_id: Dict[str, Document],
        child_count: int,
    ):
        self.child_retriever = child_retriever
        self.parent_by_id = parent_by_id
        self.child_count = child_count
        self.last_metrics = {}

    def hybrid_search(self, query: str, top_k: int = 3, candidate_k: int = None) -> List[Document]:
        if top_k <= 0 or self.child_count == 0:
            return []

        fixed_candidate_k = candidate_k if candidate_k is not None else 80
        requested = min(self.child_count, fixed_candidate_k)
        child_docs = self.child_retriever.hybrid_search(
            query,
            top_k=requested,
            candidate_k=fixed_candidate_k,
        )
        parents = self._unique_parents(child_docs)
        self.last_metrics = dict(self.child_retriever.last_metrics)
        self.last_metrics.update(
            {
                "child_results_requested": requested,
                "unique_parents_found": len(parents),
            }
        )
        return parents[:top_k]

    def _unique_parents(self, children: Sequence[Document]) -> List[Document]:
        parents = []
        seen = set()
        for child in children:
            parent_id = child.metadata.get("parent_chunk_id")
            if not parent_id or parent_id in seen or parent_id not in self.parent_by_id:
                continue
            seen.add(parent_id)
            parents.append(self.parent_by_id[parent_id])
        return parents


class SourceAwareMaxRetriever(ParentChildRetriever):
    """PC2: max child per parent, then max parent per source."""

    def hybrid_search(self, query: str, top_k: int = 3, candidate_k: int = None) -> List[Document]:
        if top_k <= 0 or self.child_count == 0:
            return []

        fixed_candidate_k = candidate_k if candidate_k is not None else 80
        requested = min(self.child_count, fixed_candidate_k)
        child_docs = self.child_retriever.hybrid_search(
            query,
            top_k=requested,
            candidate_k=fixed_candidate_k,
        )

        parent_scores = {}
        for rank, child in enumerate(child_docs):
            parent_id = child.metadata.get("parent_chunk_id")
            if not parent_id or parent_id not in self.parent_by_id:
                continue
            score = float(child.metadata.get("rrf_score", -rank))
            previous = parent_scores.get(parent_id)
            if previous is None or score > previous[0]:
                parent_scores[parent_id] = (score, rank)

        source_scores = {}
        for parent_id, (parent_score, first_rank) in parent_scores.items():
            parent = self.parent_by_id[parent_id]
            source = str(
                parent.metadata.get("relative_path")
                or parent.metadata.get("source")
                or parent_id
            )
            previous = source_scores.get(source)
            if previous is None or parent_score > previous[0]:
                source_scores[source] = (parent_score, first_rank, parent)

        ranked_sources = sorted(
            source_scores.values(),
            key=lambda item: (-item[0], item[1]),
        )
        self.last_metrics = dict(self.child_retriever.last_metrics)
        self.last_metrics.update(
            {
                "child_results_requested": requested,
                "unique_parents_found": len(parent_scores),
                "unique_sources_found": len(source_scores),
            }
        )
        return [parent for _, _, parent in ranked_sources[:top_k]]


class SourceAwareTop2Retriever(ParentChildRetriever):
    """H1: weighted Top-2 children per parent and parents per source."""

    child_support_weight = 0.5
    parent_support_weight = 0.3

    def __init__(
        self,
        child_retriever: RetrievalOptimizationModule,
        parent_by_id: Dict[str, Document],
        child_count: int,
    ):
        super().__init__(child_retriever, parent_by_id, child_count)
        self.last_aggregation = {"ranked_sources": []}

    @staticmethod
    def _child_id(child: Document, parent_id: str, rank: int) -> str:
        return str(
            child.metadata.get("chunk_id")
            or hashlib.md5(
                f"{parent_id}|{rank}|{child.page_content}".encode("utf-8")
            ).hexdigest()
        )

    @staticmethod
    def _source_id(parent: Document, parent_id: str) -> str:
        return str(
            parent.metadata.get("relative_path")
            or parent.metadata.get("source")
            or parent_id
        )

    def hybrid_search(self, query: str, top_k: int = 3, candidate_k: int = None) -> List[Document]:
        if top_k <= 0 or self.child_count == 0:
            self.last_aggregation = {"ranked_sources": []}
            return []

        fixed_candidate_k = candidate_k if candidate_k is not None else 80
        requested = min(self.child_count, fixed_candidate_k)
        child_docs = self.child_retriever.hybrid_search(
            query,
            top_k=requested,
            candidate_k=fixed_candidate_k,
        )

        children_by_parent = {}
        for rank, child in enumerate(child_docs):
            parent_id = str(child.metadata.get("parent_chunk_id") or "")
            if not parent_id or parent_id not in self.parent_by_id:
                continue
            child_id = self._child_id(child, parent_id, rank)
            score = float(child.metadata.get("rrf_score", -rank))
            child_entry = {
                "child_id": child_id,
                "score": score,
                "rank": rank,
            }
            parent_children = children_by_parent.setdefault(parent_id, {})
            previous = parent_children.get(child_id)
            if previous is None or (-score, rank, child_id) < (
                -previous["score"],
                previous["rank"],
                previous["child_id"],
            ):
                parent_children[child_id] = child_entry

        parent_entries = {}
        for parent_id, child_entries_by_id in children_by_parent.items():
            children = sorted(
                child_entries_by_id.values(),
                key=lambda item: (-item["score"], item["rank"], item["child_id"]),
            )[:2]
            parent_score = children[0]["score"]
            if len(children) > 1:
                parent_score += self.child_support_weight * children[1]["score"]
            child_evidence = []
            for index, child in enumerate(children):
                weight = 1.0 if index == 0 else self.child_support_weight
                child_evidence.append(
                    {
                        **child,
                        "weighted_contribution": weight * child["score"],
                    }
                )
            parent_entries[parent_id] = {
                "parent_id": parent_id,
                "parent": self.parent_by_id[parent_id],
                "parent_score": parent_score,
                "best_child_rank": children[0]["rank"],
                "children": child_evidence,
            }

        parents_by_source = {}
        for parent_id, parent_entry in parent_entries.items():
            source = self._source_id(parent_entry["parent"], parent_id)
            parents_by_source.setdefault(source, []).append(parent_entry)

        source_entries = []
        for source, parent_candidates in parents_by_source.items():
            parents = sorted(
                parent_candidates,
                key=lambda item: (
                    -item["parent_score"],
                    item["best_child_rank"],
                    item["parent_id"],
                ),
            )[:2]
            source_score = parents[0]["parent_score"]
            if len(parents) > 1:
                source_score += self.parent_support_weight * parents[1]["parent_score"]
            parent_evidence = []
            for index, parent in enumerate(parents):
                weight = 1.0 if index == 0 else self.parent_support_weight
                parent_evidence.append(
                    {
                        "parent_id": parent["parent_id"],
                        "parent_score": parent["parent_score"],
                        "best_child_rank": parent["best_child_rank"],
                        "weighted_contribution": weight * parent["parent_score"],
                        "children": parent["children"],
                    }
                )
            source_entries.append(
                {
                    "source": source,
                    "source_score": source_score,
                    "best_child_rank": parents[0]["best_child_rank"],
                    "returned_parent": parents[0]["parent"],
                    "returned_parent_id": parents[0]["parent_id"],
                    "parents": parent_evidence,
                }
            )

        ranked_sources = sorted(
            source_entries,
            key=lambda item: (
                -item["source_score"],
                item["best_child_rank"],
                item["source"],
            ),
        )
        self.last_aggregation = {
            "ranked_sources": [
                {
                    key: value
                    for key, value in source.items()
                    if key != "returned_parent"
                }
                for source in ranked_sources[:top_k]
            ]
        }
        self.last_metrics = dict(self.child_retriever.last_metrics)
        self.last_metrics.update(
            {
                "child_results_requested": requested,
                "unique_parents_found": len(parent_entries),
                "unique_sources_found": len(source_entries),
                "parents_with_second_child": sum(
                    len(parent["children"]) > 1 for parent in parent_entries.values()
                ),
                "sources_with_second_parent": sum(
                    len(source["parents"]) > 1 for source in source_entries
                ),
            }
        )
        return [source["returned_parent"] for source in ranked_sources[:top_k]]


def evaluate_ablation_cases(retriever, cases: Sequence, top_k: int) -> List[dict]:
    """Evaluate retrieval variants through one shared ranking path."""
    results = []
    data_root = Path(DEFAULT_CONFIG.data_path)
    for case in cases:
        docs = retriever.hybrid_search(case.retrieval_query, top_k=top_k)
        paths = [doc.metadata.get("relative_path", "") for doc in docs]
        topics = [doc.metadata.get("topic", "") for doc in docs]
        expected_file = data_root / case.expected_path_contains
        evaluable = expected_file.is_dir() or (
            expected_file.is_file() and expected_file.stat().st_size >= 100
        )
        source_rank = next(
            (
                rank
                for rank, path in enumerate(paths, start=1)
                if evaluable and case.expected_path_contains in path
            ),
            None,
        )
        topic_rank = next(
            (rank for rank, topic in enumerate(topics, start=1) if topic == case.expected_topic),
            None,
        )
        results.append(
            {
                "case": case,
                "docs": docs,
                "paths": paths,
                "topics": topics,
                "top1_hit": source_rank == 1,
                "topk_hit": source_rank is not None,
                "topic_hit": topic_rank is not None,
                "evaluable": evaluable,
                "source_rank": source_rank,
                "topic_rank": topic_rank,
                "source_rr": 1.0 / source_rank if source_rank else 0.0,
                "topic_rr": 1.0 / topic_rank if topic_rank else 0.0,
                "retrieval_metrics": dict(getattr(retriever, "last_metrics", {})),
                "aggregation_evidence": dict(
                    getattr(retriever, "last_aggregation", {})
                ),
            }
        )
    return results


def _evaluation_summary(results: Sequence[dict], top_k: int) -> dict:
    evaluable = [result for result in results if result["evaluable"]]
    evaluable_total = len(evaluable)
    divisor = evaluable_total or 1
    return {
        "case_count": len(results),
        "source_evaluable_count": evaluable_total,
        "top_k": top_k,
        "top1_source_accuracy": sum(result["top1_hit"] for result in evaluable) / divisor,
        f"top{top_k}_source_accuracy": sum(result["topk_hit"] for result in evaluable) / divisor,
        f"top{top_k}_topic_accuracy": sum(result["topic_hit"] for result in evaluable) / divisor,
        f"source_mrr_at_{top_k}": sum(result["source_rr"] for result in evaluable) / divisor,
        f"topic_mrr_at_{top_k}": sum(result["topic_rr"] for result in evaluable) / divisor,
    }


def _case_report(results: Sequence[dict]) -> List[dict]:
    return [
        {
            "name": result["case"].name,
            "evaluable": result["evaluable"],
            "top1_hit": result["top1_hit"],
            "topk_hit": result["topk_hit"],
            "topic_hit": result["topic_hit"],
            "source_rank": result["source_rank"],
            "topic_rank": result["topic_rank"],
            "source_rr": result["source_rr"],
            "topic_rr": result["topic_rr"],
            "paths": result["paths"],
            "retrieval_metrics": result["retrieval_metrics"],
            "expected_path": result["case"].expected_path_contains,
            "aggregation_evidence": result["aggregation_evidence"],
        }
        for result in results
    ]


def compare_case_reports(
    baseline: Sequence[dict],
    candidate: Sequence[dict],
    candidate_label: str = "pc1",
    baseline_label: str = "pc0",
) -> List[dict]:
    changes = []
    for baseline_case, candidate_case in zip(baseline, candidate):
        if baseline_case["name"] != candidate_case["name"]:
            raise ValueError("Baseline and candidate case order does not match.")
        outcome_changed = any(
            baseline_case[key] != candidate_case[key]
            for key in ("top1_hit", "topk_hit", "topic_hit")
        )
        ranking_changed = any(
            baseline_case[key] != candidate_case[key]
            for key in ("source_rank", "topic_rank")
        )
        paths_changed = baseline_case["paths"] != candidate_case["paths"]
        if outcome_changed or ranking_changed or paths_changed:
            baseline_rank = baseline_case["source_rank"]
            candidate_rank = candidate_case["source_rank"]
            if baseline_rank is None and candidate_rank is not None:
                change_type = "entered"
            elif baseline_rank is not None and candidate_rank is None:
                change_type = "dropped"
            elif baseline_rank is not None and candidate_rank < baseline_rank:
                change_type = "improved"
            elif candidate_rank is not None and baseline_rank < candidate_rank:
                change_type = "regressed"
            elif baseline_case["topic_rank"] != candidate_case["topic_rank"]:
                change_type = "topic_rank_change"
            else:
                change_type = "path_change"
            changes.append(
                {
                    "name": baseline_case["name"],
                    "evaluable": baseline_case["evaluable"],
                    "change_type": change_type,
                    "outcome_changed": outcome_changed,
                    "ranking_changed": ranking_changed,
                    "paths_changed": paths_changed,
                    baseline_label: baseline_case,
                    candidate_label: candidate_case,
                }
            )
    return changes


def _length_stats(documents: Sequence[Document]) -> dict:
    lengths = [len(document.page_content) for document in documents]
    return {
        "count": len(lengths),
        "average_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "maximum_length": max(lengths, default=0),
    }


def _source_size_stats(
    parent_by_id: Dict[str, Document],
    children: Sequence[Document],
) -> dict:
    stats = {}
    parent_sources = {}
    for parent_id, parent in parent_by_id.items():
        source = str(
            parent.metadata.get("relative_path")
            or parent.metadata.get("source")
            or parent_id
        )
        parent_sources[parent_id] = source
        source_stats = stats.setdefault(source, {"parent_count": 0, "child_count": 0})
        source_stats["parent_count"] += 1
    for child in children:
        parent_id = str(child.metadata.get("parent_chunk_id") or "")
        source = parent_sources.get(parent_id)
        if source is not None:
            stats[source]["child_count"] += 1
    return stats


def _length_bias_diagnostic(
    changes: Sequence[dict],
    baseline_label: str,
    candidate_label: str,
    source_size_stats: dict,
) -> dict:
    highest_quartile_count = math.ceil(len(source_size_stats) / 4)
    highest_quartile_sources = [
        source
        for source, _ in sorted(
            source_size_stats.items(),
            key=lambda item: (-item[1]["child_count"], item[0]),
        )[:highest_quartile_count]
    ]
    highest_quartile_set = set(highest_quartile_sources)
    improved_cases = []
    for change in changes:
        if not change["evaluable"]:
            continue
        baseline_rank = change[baseline_label]["source_rank"]
        candidate_rank = change[candidate_label]["source_rank"]
        baseline_value = baseline_rank if baseline_rank is not None else math.inf
        candidate_value = candidate_rank if candidate_rank is not None else math.inf
        if candidate_value < baseline_value:
            improved_cases.append(
                {
                    "name": change["name"],
                    "expected_path": change[candidate_label]["expected_path"],
                    "in_highest_child_count_quartile": (
                        change[candidate_label]["expected_path"] in highest_quartile_set
                    ),
                }
            )
    high_quartile_improvement_count = sum(
        case["in_highest_child_count_quartile"] for case in improved_cases
    )
    return {
        "source_size_stats": source_size_stats,
        "highest_child_count_quartile_sources": highest_quartile_sources,
        "improved_cases": improved_cases,
        "improvement_count": len(improved_cases),
        "high_quartile_improvement_count": high_quartile_improvement_count,
        "manual_review": bool(improved_cases)
        and high_quartile_improvement_count > len(improved_cases) / 2,
    }


def _metrics_gate(
    baseline: dict,
    candidate: dict,
    top_k: int,
    source_rank_change_count: int,
) -> bool:
    baseline_topk = baseline[f"top{top_k}_source_accuracy"]
    candidate_topk = candidate[f"top{top_k}_source_accuracy"]
    return (
        candidate[f"source_mrr_at_{top_k}"] > baseline[f"source_mrr_at_{top_k}"]
        and candidate_topk >= baseline_topk
        and candidate[f"topic_mrr_at_{top_k}"] >= baseline[f"topic_mrr_at_{top_k}"]
        and source_rank_change_count > 1
    )


def _retrieval_stats(results: Sequence[dict]) -> dict:
    latencies = [
        result["retrieval_metrics"].get("total_retrieval_ms", 0)
        for result in results
    ]
    unique_parents = [
        result["retrieval_metrics"].get("unique_parents_found", 0)
        for result in results
    ]
    unique_sources = [
        result["retrieval_metrics"].get("unique_sources_found", 0)
        for result in results
        if "unique_sources_found" in result["retrieval_metrics"]
    ]
    parents_with_second_child = [
        result["retrieval_metrics"].get("parents_with_second_child", 0)
        for result in results
        if "parents_with_second_child" in result["retrieval_metrics"]
    ]
    sources_with_second_parent = [
        result["retrieval_metrics"].get("sources_with_second_parent", 0)
        for result in results
        if "sources_with_second_parent" in result["retrieval_metrics"]
    ]
    parent_support_ratios = [
        result["retrieval_metrics"].get("parents_with_second_child", 0)
        / result["retrieval_metrics"].get("unique_parents_found", 1)
        for result in results
        if result["retrieval_metrics"].get("unique_parents_found", 0) > 0
        and "parents_with_second_child" in result["retrieval_metrics"]
    ]
    source_support_ratios = [
        result["retrieval_metrics"].get("sources_with_second_parent", 0)
        / result["retrieval_metrics"].get("unique_sources_found", 1)
        for result in results
        if result["retrieval_metrics"].get("unique_sources_found", 0) > 0
        and "sources_with_second_parent" in result["retrieval_metrics"]
    ]
    return {
        "average_retrieval_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "maximum_retrieval_ms": max(latencies, default=0),
        "average_unique_parents": (
            sum(unique_parents) / len(unique_parents) if unique_parents else 0.0
        ),
        "average_unique_sources": (
            sum(unique_sources) / len(unique_sources) if unique_sources else 0.0
        ),
        "average_parents_with_second_child": (
            sum(parents_with_second_child) / len(parents_with_second_child)
            if parents_with_second_child
            else 0.0
        ),
        "average_sources_with_second_parent": (
            sum(sources_with_second_parent) / len(sources_with_second_parent)
            if sources_with_second_parent
            else 0.0
        ),
        "average_parent_second_child_ratio": (
            sum(parent_support_ratios) / len(parent_support_ratios)
            if parent_support_ratios
            else 0.0
        ),
        "average_source_second_parent_ratio": (
            sum(source_support_ratios) / len(source_support_ratios)
            if source_support_ratios
            else 0.0
        ),
    }


def main() -> int:
    args = parse_args()
    if not args.online:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = DEFAULT_CONFIG
    data_module = DataPreparationModule(
        config.data_path,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    data_module.load_documents()
    parents = data_module.chunk_documents()
    parent_by_id, children = build_parent_child_corpus(
        parents,
        child_size=args.child_size,
        child_overlap=args.child_overlap,
    )

    index_module = IndexConstructionModule(config.embedding_model, config.index_save_path)
    pc0_vectorstore = index_module.build_vector_index(children, write_manifest=False)
    pc0_child_retriever = ExperimentChildRetriever(pc0_vectorstore, children)
    pc0_retriever = ParentChildRetriever(pc0_child_retriever, parent_by_id, len(children))
    pc0_results = evaluate_ablation_cases(pc0_retriever, EVAL_CASES, args.top_k)

    baseline_label = "pc0"
    baseline_results = pc0_results
    candidate_label = args.experiment
    if candidate_label == "pc1":
        contextual_children = build_contextual_children(children)
        candidate_vectorstore = index_module.build_vector_index(
            contextual_children,
            write_manifest=False,
        )
        candidate_child_retriever = ExperimentChildRetriever(
            candidate_vectorstore,
            contextual_children,
        )
        candidate_retriever = ParentChildRetriever(
            candidate_child_retriever,
            parent_by_id,
            len(contextual_children),
        )
    elif candidate_label == "pc2":
        candidate_retriever = SourceAwareMaxRetriever(
            pc0_child_retriever,
            parent_by_id,
            len(children),
        )
    else:
        baseline_label = "pc2"
        baseline_retriever = SourceAwareMaxRetriever(
            pc0_child_retriever,
            parent_by_id,
            len(children),
        )
        baseline_results = evaluate_ablation_cases(
            baseline_retriever,
            EVAL_CASES,
            args.top_k,
        )
        candidate_retriever = SourceAwareTop2Retriever(
            pc0_child_retriever,
            parent_by_id,
            len(children),
        )
    candidate_results = evaluate_ablation_cases(
        candidate_retriever,
        EVAL_CASES,
        args.top_k,
    )

    baseline_evaluation = _evaluation_summary(baseline_results, args.top_k)
    candidate_evaluation = _evaluation_summary(candidate_results, args.top_k)
    baseline_cases = _case_report(baseline_results)
    candidate_cases = _case_report(candidate_results)
    changes = compare_case_reports(
        baseline_cases,
        candidate_cases,
        candidate_label=candidate_label,
        baseline_label=baseline_label,
    )
    source_rank_change_count = sum(
        change["evaluable"]
        and change[baseline_label]["source_rank"]
        != change[candidate_label]["source_rank"]
        for change in changes
    )
    source_rank_improvement_count = sum(
        change["evaluable"]
        and (
            change[candidate_label]["source_rank"]
            if change[candidate_label]["source_rank"] is not None
            else math.inf
        )
        < (
            change[baseline_label]["source_rank"]
            if change[baseline_label]["source_rank"] is not None
            else math.inf
        )
        for change in changes
    )
    source_rank_regression_count = sum(
        change["evaluable"] and change["change_type"] in {"regressed", "dropped"}
        for change in changes
    )
    source_drop_count = sum(
        change["evaluable"] and change["change_type"] == "dropped"
        for change in changes
    )
    metrics_gate_passed = _metrics_gate(
        baseline_evaluation,
        candidate_evaluation,
        args.top_k,
        source_rank_improvement_count if candidate_label == "h1" else source_rank_change_count,
    )
    if candidate_label == "h1" and source_drop_count:
        metrics_gate_passed = False
    length_bias_diagnostic = None
    if candidate_label == "h1":
        length_bias_diagnostic = _length_bias_diagnostic(
            changes,
            baseline_label,
            candidate_label,
            _source_size_stats(parent_by_id, children),
        )
    recommendation = "failed"
    if metrics_gate_passed:
        recommendation = (
            "manual_review"
            if length_bias_diagnostic and length_bias_diagnostic["manual_review"]
            else "passed"
        )

    report = {
        "parent_settings": {
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
        },
        "child_settings": {
            "chunk_size": args.child_size,
            "chunk_overlap": args.child_overlap,
        },
        "parent_stats": _length_stats(parents),
        "child_stats": _length_stats(children),
        "candidate_k": 80,
        f"{baseline_label}_evaluation": baseline_evaluation,
        f"{candidate_label}_evaluation": candidate_evaluation,
        f"{baseline_label}_retrieval_stats": _retrieval_stats(baseline_results),
        f"{candidate_label}_retrieval_stats": _retrieval_stats(candidate_results),
        "source_rank_change_count": source_rank_change_count,
        "source_rank_improvement_count": source_rank_improvement_count,
        "source_rank_regression_count": source_rank_regression_count,
        "source_drop_count": source_drop_count,
        "metrics_gate_passed": metrics_gate_passed,
        "recommendation": recommendation,
        "length_bias_diagnostic": length_bias_diagnostic,
        "case_changes": changes,
        f"{baseline_label}_cases": baseline_cases,
        f"{candidate_label}_cases": candidate_cases,
    }

    baseline_display = baseline_label.upper()
    candidate_display = candidate_label.upper()
    print(f"Parent-child {baseline_display}/{candidate_display} ablation evaluation")
    print(f"  parents: {report['parent_stats']['count']}")
    print(f"  children: {report['child_stats']['count']}")
    print(f"  average parent length: {report['parent_stats']['average_length']:.1f}")
    print(f"  maximum parent length: {report['parent_stats']['maximum_length']}")
    print(f"  average child length: {report['child_stats']['average_length']:.1f}")
    print(f"  maximum child length: {report['child_stats']['maximum_length']}")
    print(f"  candidate_k per retrieval route: {report['candidate_k']}")
    for label, evaluation in (
        (baseline_display, baseline_evaluation),
        (candidate_display, candidate_evaluation),
    ):
        print(f"  {label} top1 source accuracy: {evaluation['top1_source_accuracy']:.1%}")
        print(f"  {label} top{args.top_k} source accuracy: {evaluation[f'top{args.top_k}_source_accuracy']:.1%}")
        print(f"  {label} top{args.top_k} topic accuracy: {evaluation[f'top{args.top_k}_topic_accuracy']:.1%}")
        print(f"  {label} source MRR@{args.top_k}: {evaluation[f'source_mrr_at_{args.top_k}']:.4f}")
        print(f"  {label} topic MRR@{args.top_k}: {evaluation[f'topic_mrr_at_{args.top_k}']:.4f}")
        retrieval_stats = report[f"{label.lower()}_retrieval_stats"]
        print(f"  {label} average retrieval latency: {retrieval_stats['average_retrieval_ms']:.1f} ms")
        print(f"  {label} maximum retrieval latency: {retrieval_stats['maximum_retrieval_ms']} ms")
    print(f"  changed cases: {len(changes)}")
    print(f"  source-rank-changing evaluable cases: {source_rank_change_count}")
    print(f"  source-rank improvements: {source_rank_improvement_count}")
    print(f"  source-rank regressions: {source_rank_regression_count}")
    print(f"  sources dropped from top{args.top_k}: {source_drop_count}")
    print(f"  metrics gate passed: {metrics_gate_passed}")
    print(f"  recommendation: {recommendation}")
    if length_bias_diagnostic:
        print(
            "  improvements from highest child-count quartile: "
            f"{length_bias_diagnostic['high_quartile_improvement_count']}/"
            f"{length_bias_diagnostic['improvement_count']}"
        )
    if changes:
        print("  per-case changes:")
        for change in changes:
            baseline = change[baseline_label]
            candidate = change[candidate_label]
            print(f"    {change['name']}:")
            print(f"      change type: {change['change_type']}")
            print(
                "      "
                f"source rank {baseline['source_rank']} -> {candidate['source_rank']}; "
                f"topic rank {baseline['topic_rank']} -> {candidate['topic_rank']}"
            )
            print(f"      {baseline_display} paths: {baseline['paths']}")
            print(f"      {candidate_display} paths: {candidate['paths']}")

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  report: {args.report_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
