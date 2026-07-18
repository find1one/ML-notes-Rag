"""Offline index build, evaluation, and publish command."""

import argparse
import json
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
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Recursive character chunk size; defaults to RAG_CHUNK_SIZE or 1200.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optionally write aggregate metrics and per-case retrieval outcomes as JSON.",
    )
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
    chunk_size = args.chunk_size if args.chunk_size is not None else config.chunk_size
    data_module = DataPreparationModule(
        config.data_path,
        chunk_size=chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    data_module.load_documents()
    chunks = data_module.chunk_documents()

    index_module = IndexConstructionModule(config.embedding_model, config.index_save_path)
    vectorstore = index_module.build_vector_index(
        chunks,
        skipped_files=data_module.skipped_files,
        write_manifest=False,
    )
    retriever = RetrievalOptimizationModule(vectorstore, chunks)
    results = evaluate_cases(retriever, EVAL_CASES, args.top_k)
    evaluation = _evaluation_summary(results, args.top_k, args.min_top1, args.min_topk)

    print("Offline index evaluation")
    chunk_lengths = [len(chunk.page_content) for chunk in chunks]
    print(f"  chunk size setting: {chunk_size} characters")
    print(f"  chunk overlap: {config.chunk_overlap} characters")
    print(f"  chunks: {len(chunks)}")
    print(f"  average chunk length: {sum(chunk_lengths) / len(chunk_lengths):.1f} characters")
    print(f"  maximum chunk length: {max(chunk_lengths)} characters")
    print(f"  status: {evaluation['status']}")
    print(f"  source-evaluable cases: {evaluation['source_evaluable_count']}/{evaluation['case_count']}")
    print(f"  top1 source accuracy: {evaluation['top1_source_accuracy']:.1%}")
    print(f"  top{args.top_k} source accuracy: {evaluation[f'top{args.top_k}_source_accuracy']:.1%}")
    print(f"  top{args.top_k} topic accuracy: {evaluation[f'top{args.top_k}_topic_accuracy']:.1%}")

    if args.report_json:
        report = {
            "chunk_size": chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "chunk_count": len(chunks),
            "average_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "maximum_chunk_length": max(chunk_lengths),
            "evaluation": evaluation,
            "cases": [
                {
                    "name": result["case"].name,
                    "evaluable": result["evaluable"],
                    "top1_hit": result["top1_hit"],
                    "topk_hit": result["topk_hit"],
                    "topic_hit": result["topic_hit"],
                    "paths": result["paths"],
                }
                for result in results
            ],
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  report: {args.report_json}")

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
