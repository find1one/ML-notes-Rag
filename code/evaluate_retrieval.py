"""
Offline retrieval evaluation for the ML notes RAG system.

This script does not call the LLM. It evaluates whether the retriever can
return the expected source file/topic for a small curated query set.
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


@dataclass(frozen=True)
class EvalCase:
    name: str
    user_query: str
    retrieval_query: str
    expected_topic: str
    expected_path_contains: str


EVAL_CASES: Sequence[EvalCase] = (
    EvalCase(
        name="linear_regression_variable_selection",
        user_query="线性回归怎么做变量选择？",
        retrieval_query="linear regression variable selection feature selection p-value backward elimination",
        expected_topic="Regression",
        expected_path_contains="01-Regression/01-LinearRegression.md",
    ),
    EvalCase(
        name="linear_regression_assumptions",
        user_query="线性回归有哪些假设？",
        retrieval_query="linear regression assumptions linearity homoscedasticity multivariate normality independence errors multicollinearity",
        expected_topic="Regression",
        expected_path_contains="01-Regression/01-LinearRegression.md",
    ),
    EvalCase(
        name="classification_methods",
        user_query="分类有哪些方法？",
        retrieval_query="classification logistic regression knn support vector machines naive bayes decision tree random forest",
        expected_topic="Classification",
        expected_path_contains="02-Classification",
    ),
    EvalCase(
        name="naive_bayes",
        user_query="什么是朴素贝叶斯？",
        retrieval_query="naive bayes classifier algorithm probability classification",
        expected_topic="Classification",
        expected_path_contains="02-Classification/04-NaiveBayes.md",
    ),
    EvalCase(
        name="decision_tree",
        user_query="决策树是什么？",
        retrieval_query="decision tree classification algorithm",
        expected_topic="Classification",
        expected_path_contains="02-Classification/05-DecisionTree.md",
    ),
    EvalCase(
        name="random_forest",
        user_query="随机森林是什么？",
        retrieval_query="random forest classification ensemble decision trees",
        expected_topic="Classification",
        expected_path_contains="02-Classification/06-RandomForest.md",
    ),
    EvalCase(
        name="kmeans_steps",
        user_query="K-means 聚类的步骤是什么？",
        retrieval_query="K-means clustering algorithm steps objective function soft k-means",
        expected_topic="Clustering",
        expected_path_contains="03-Clustering/01-K-meansClustering.md",
    ),
    EvalCase(
        name="hierarchical_clustering",
        user_query="层次聚类如何合并簇？",
        retrieval_query="hierarchical clustering euclidean distance clusters closest points farthest points centroids",
        expected_topic="Clustering",
        expected_path_contains="03-Clustering/02-HierarchicalClustering.md",
    ),
    EvalCase(
        name="gaussian_mixture",
        user_query="高斯混合模型是什么？",
        retrieval_query="gaussian mixture model clustering expectation maximization",
        expected_topic="Clustering",
        expected_path_contains="03-Clustering/03-GaussianMixtureModels.md",
    ),
    EvalCase(
        name="apriori",
        user_query="Apriori 算法是什么？",
        retrieval_query="Apriori association rule learning algorithm support confidence lift",
        expected_topic="Association Rule Learning",
        expected_path_contains="04-AssociationRuleLearning/01-Apriori.md",
    ),
    EvalCase(
        name="pca",
        user_query="PCA 是什么？",
        retrieval_query="principal component analysis PCA dimensionality reduction",
        expected_topic="Dimensionality Reduction",
        expected_path_contains="08-DimensionalityReduction/01-PrincipalComponentAnalysis.md",
    ),
    EvalCase(
        name="time_series_state_space",
        user_query="状态空间模型是什么？",
        retrieval_query="state space models time series",
        expected_topic="Time Series",
        expected_path_contains="11-TimeSeries/StateSpaceModels.md",
    ),
    EvalCase(
        name="dummy_variable_trap",
        user_query="dummy variable trap 是什么？",
        retrieval_query="dummy variable trap one-hot encoding linear regression categorical variables",
        expected_topic="Regression",
        expected_path_contains="01-Regression/01-LinearRegression.md",
    ),
    EvalCase(
        name="backward_elimination_code",
        user_query="backward elimination 用哪个函数？",
        retrieval_query="backward elimination statsmodels OLS add column of ones",
        expected_topic="Regression",
        expected_path_contains="01-Regression/01-LinearRegression.md",
    ),
    EvalCase(
        name="numpy_matrix",
        user_query="numpy matrix 怎么用？",
        retrieval_query="numpy matrix tutorial Python",
        expected_topic="Prerequisites",
        expected_path_contains="00-Prerequisites/numpyMatrixTutorial.md",
    ),
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
    data_module = DataPreparationModule(config.data_path)
    data_module.load_documents()
    chunks = data_module.chunk_documents()

    index_module = IndexConstructionModule(config.embedding_model, config.index_save_path)
    vectorstore = index_module.load_index()
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
    for case in cases:
        docs = retriever.hybrid_search(case.retrieval_query, top_k=top_k)
        paths = [doc.metadata.get("relative_path", "") for doc in docs]
        topics = [doc.metadata.get("topic", "") for doc in docs]
        sections = [doc.metadata.get("section_path", "") for doc in docs]

        top1_path = paths[0] if paths else ""
        top1_topic = topics[0] if topics else ""
        top1_hit = case.expected_path_contains in top1_path
        topk_hit = any(case.expected_path_contains in path for path in paths)
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
            }
        )
    return results


def print_report(results: Sequence[dict], top_k: int, config: RAGConfig) -> None:
    total = len(results)
    top1_hits = sum(1 for result in results if result["top1_hit"])
    topk_hits = sum(1 for result in results if result["topk_hit"])
    topic_hits = sum(1 for result in results if result["topic_hit"])

    print("\nRetrieval Evaluation")
    print("=" * 80)
    print(f"Data path: {config.data_path}")
    print(f"Index path: {config.index_save_path}")
    print(f"Embedding model: {config.embedding_model}")
    print(f"Cases: {total}")
    print(f"Top-1 source accuracy: {top1_hits}/{total} = {top1_hits / total:.1%}")
    print(f"Top-{top_k} source accuracy: {topk_hits}/{total} = {topk_hits / total:.1%}")
    print(f"Top-{top_k} topic accuracy: {topic_hits}/{total} = {topic_hits / total:.1%}")

    print("\nPer-case results")
    print("-" * 80)
    for result in results:
        case = result["case"]
        status = "PASS" if result["topk_hit"] else "FAIL"
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
