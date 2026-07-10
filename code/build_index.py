"""Offline index build, evaluation, and publish command."""

import argparse
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from config import DEFAULT_CONFIG
from evaluate_retrieval import EVAL_CASES, evaluate_cases
from rag_modules import DataPreparationModule, IndexConstructionModule, RetrievalOptimizationModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and publish the verified ML-notes FAISS index.")
    parser.add_argument("--publish", action="store_true", help="Save the index and write a passed manifest when evaluation succeeds.")
    parser.add_argument("--top-k", type=int, default=3, help="Evaluation top-k.")
    parser.add_argument("--min-top1", type=float, default=0.60, help="Minimum top-1 source accuracy for publish.")
    parser.add_argument("--min-topk", type=float, default=0.85, help="Minimum top-k source accuracy for publish.")
    parser.add_argument("--online", action="store_true", help="Allow Hugging Face network access.")
    return parser.parse_args()


def _evaluation_summary(results: list[dict], top_k: int, min_top1: float, min_topk: float) -> dict:
    evaluable = [result for result in results if result["evaluable"]]
    total = len(results)
    evaluable_total = len(evaluable)
    top1_hits = sum(1 for result in evaluable if result["top1_hit"])
    topk_hits = sum(1 for result in evaluable if result["topk_hit"])
    topic_hits = sum(1 for result in evaluable if result["topic_hit"])
    top1_accuracy = top1_hits / evaluable_total if evaluable_total else 0.0
    topk_accuracy = topk_hits / evaluable_total if evaluable_total else 0.0
    status = "passed" if top1_accuracy >= min_top1 and topk_accuracy >= min_topk else "failed"
    return {
        "status": status,
        "case_count": total,
        "source_evaluable_count": evaluable_total,
        "top_k": top_k,
        "top1_source_accuracy": top1_accuracy,
        f"top{top_k}_source_accuracy": topk_accuracy,
        f"top{top_k}_topic_accuracy": topic_hits / evaluable_total if evaluable_total else 0.0,
        "thresholds": {"min_top1": min_top1, "min_topk": min_topk},
    }


def main() -> int:
    args = parse_args()
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    config = DEFAULT_CONFIG
    data_module = DataPreparationModule(config.data_path)
    data_module.load_documents()
    chunks = data_module.chunk_documents()

    index_module = IndexConstructionModule(config.embedding_model, config.index_save_path)
    vectorstore = index_module.build_vector_index(chunks, skipped_files=data_module.skipped_files)
    retriever = RetrievalOptimizationModule(vectorstore, chunks)
    results = evaluate_cases(retriever, EVAL_CASES, args.top_k)
    evaluation = _evaluation_summary(results, args.top_k, args.min_top1, args.min_topk)

    print("Offline index evaluation")
    print(f"  status: {evaluation['status']}")
    print(f"  source-evaluable cases: {evaluation['source_evaluable_count']}/{evaluation['case_count']}")
    print(f"  top1 source accuracy: {evaluation['top1_source_accuracy']:.1%}")
    print(f"  top{args.top_k} source accuracy: {evaluation[f'top{args.top_k}_source_accuracy']:.1%}")

    if args.publish:
        if evaluation["status"] != "passed":
            print("Evaluation did not pass; index was not published.")
            return 1
        index_module.save_index()
        index_module.write_manifest_for_existing_index(chunks, skipped_files=data_module.skipped_files, evaluation=evaluation)
        print(f"Published verified index to: {config.index_save_path}")
    return 0 if evaluation["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
