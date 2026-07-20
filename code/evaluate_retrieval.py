"""
Offline retrieval evaluation for the ML notes RAG system.

This script does not call the LLM. It evaluates whether the retriever can
return the expected source file/topic for a frozen corpus-grounded query set.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

sys.path.append(str(Path(__file__).parent))

from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import DataPreparationModule, IndexConstructionModule, RetrievalOptimizationModule
from retrieval_eval_cases import DATA_QUALITY_CASE_ROWS, PRIMARY_CASE_ROWS


@dataclass(frozen=True)
class EvalCase:
    name: str
    user_query: str
    retrieval_query: str
    expected_topic: str
    expected_path_contains: str


EVAL_CASES: Sequence[EvalCase] = tuple(EvalCase(*row) for row in PRIMARY_CASE_ROWS)
DATA_QUALITY_CASES: Sequence[EvalCase] = tuple(
    EvalCase(*row) for row in DATA_QUALITY_CASE_ROWS
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate offline retrieval accuracy.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved chunks to evaluate.")
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build and save the FAISS index if it does not already exist.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow Hugging Face online checks/downloads. By default offline cache mode is used.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    config = DEFAULT_CONFIG
    data_module = DataPreparationModule(
        config.data_path,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    data_module.load_documents()
    chunks = data_module.chunk_documents()

    index_module = IndexConstructionModule(config.embedding_model, config.index_save_path)
    vectorstore = index_module.load_index(chunks)
    if vectorstore is None:
        if not args.build_index:
            print(f"Index not found at: {config.index_save_path}")
            print("Run the app once to build it, or rerun with --build-index.")
            return 2
        vectorstore = index_module.build_vector_index(chunks)
        index_module.save_index()

    retriever = RetrievalOptimizationModule(vectorstore, chunks)
    results = evaluate_cases(retriever, EVAL_CASES, args.top_k)
    print_report(results, args.top_k, config)
    return 0


def evaluate_cases(
    retriever: RetrievalOptimizationModule,
    cases: Sequence[EvalCase],
    top_k: int,
) -> List[dict]:
    results = []
    data_root = Path(DEFAULT_CONFIG.data_path)
    for case in cases:
        docs = retriever.hybrid_search(case.retrieval_query, top_k=top_k)
        paths = [doc.metadata.get("relative_path", "") for doc in docs]
        topics = [doc.metadata.get("topic", "") for doc in docs]
        sections = [doc.metadata.get("section_path", "") for doc in docs]

        top1_path = paths[0] if paths else ""
        top1_topic = topics[0] if topics else ""
        expected_file = data_root / case.expected_path_contains
        # Empty placeholder Markdown files cannot generate a chunk, so they are
        # tracked as data-quality cases but excluded from source-recall metrics.
        evaluable = expected_file.is_dir() or (
            expected_file.is_file() and expected_file.stat().st_size >= 100
        )
        top1_hit = evaluable and case.expected_path_contains in top1_path
        topk_hit = evaluable and any(case.expected_path_contains in path for path in paths)
        topic_hit = case.expected_topic in topics
        results.append(
            {
                "case": case,
                "docs": docs,
                "paths": paths,
                "topics": topics,
                "sections": sections,
                "top1_hit": top1_hit,
                "topk_hit": topk_hit,
                "topic_hit": topic_hit,
                "top1_topic": top1_topic,
                "evaluable": evaluable,
            }
        )
    return results


def print_report(results: Sequence[dict], top_k: int, config: RAGConfig) -> None:
    total = len(results)
    evaluable_results = [result for result in results if result["evaluable"]]
    evaluable_total = len(evaluable_results)
    top1_hits = sum(1 for result in evaluable_results if result["top1_hit"])
    topk_hits = sum(1 for result in evaluable_results if result["topk_hit"])
    topic_hits = sum(1 for result in evaluable_results if result["topic_hit"])

    print("\nRetrieval Evaluation")
    print("=" * 80)
    print(f"Data path: {config.data_path}")
    print(f"Index path: {config.index_save_path}")
    print(f"Embedding model: {config.embedding_model}")
    print(f"Cases: {total} ({evaluable_total} source-evaluable, {total - evaluable_total} empty-source cases)")
    print(f"Top-1 source accuracy: {top1_hits}/{evaluable_total} = {top1_hits / evaluable_total:.1%}")
    print(f"Top-{top_k} source accuracy: {topk_hits}/{evaluable_total} = {topk_hits / evaluable_total:.1%}")
    print(f"Top-{top_k} topic accuracy: {topic_hits}/{evaluable_total} = {topic_hits / evaluable_total:.1%}")

    print("\nPer-case results")
    print("-" * 80)
    for result in results:
        case = result["case"]
        status = "SKIP" if not result["evaluable"] else ("PASS" if result["topk_hit"] else "FAIL")
        print(f"[{status}] {case.name}")
        print(f"  user query: {case.user_query}")
        print(f"  retrieval query: {case.retrieval_query}")
        print(f"  expected: topic={case.expected_topic}, path contains={case.expected_path_contains}")
        for rank, doc in enumerate(result["docs"], start=1):
            print(
                "  "
                f"{rank}. topic={doc.metadata.get('topic')} | "
                f"section={doc.metadata.get('section_path')} | "
                f"path={doc.metadata.get('relative_path')}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
