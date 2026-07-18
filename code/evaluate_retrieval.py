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

# Keep the original hand-written cases above and broaden coverage with stable
# source expectations across every topic. These cases intentionally do not use
# the LLM rewrite path so retrieval regressions remain reproducible.
_ADDITIONAL_CASES = (
    ("polynomial_regression", "多项式回归是什么", "polynomial regression", "Regression", "01-Regression/02-PolynomialRegression.md"),
    ("support_vector_regression", "支持向量回归是什么", "support vector regression SVR", "Regression", "01-Regression/03-SupportVectorRegression.md"),
    ("regression_comparison", "回归模型如何比较", "regression models comparison", "Regression", "01-Regression/00-RegressionModelsComparison.md"),
    ("logistic_regression", "逻辑回归是什么", "logistic regression classification", "Classification", "02-Classification/01-LogisticRegression.md"),
    ("knn", "KNN 分类如何工作", "k nearest neighbors KNN classification", "Classification", "02-Classification/02-knn.md"),
    ("svm", "支持向量机是什么", "support vector machines SVM classification", "Classification", "02-Classification/03-SupportVectorMachines.md"),
    ("hidden_markov", "隐马尔可夫模型是什么", "hidden markov models HMM", "Classification", "02-Classification/07-HiddenMarkovModels.md"),
    ("eclat", "Eclat 算法是什么", "Eclat association rule learning", "Association Rule Learning", "04-AssociationRuleLearning/02-Eclat.md"),
    ("reinforcement_overview", "强化学习是什么", "reinforcement learning", "Reinforcement Learning", "05-ReinforcementLearning/README.md"),
    ("nlp_overview", "自然语言处理是什么", "natural language processing NLP", "Natural Language Processing", "06-NaturalLanguageProcessing/README.md"),
    ("deep_learning_overview", "深度学习是什么", "deep learning neural networks", "Deep Learning", "07-DeepLearning/README.md"),
    ("pca_steps", "PCA 如何降维", "principal component analysis PCA dimensionality reduction", "Dimensionality Reduction", "08-DimensionalityReduction/01-PrincipalComponentAnalysis.md"),
    ("recommendation_overview", "推荐系统是什么", "recommendation engines", "Recommendation Engines", "09-RecommendationEngines/README.md"),
    ("boosting_overview", "Boosting 是什么", "model selection boosting", "Model Selection and Boosting", "10-ModelSelectionAndBoosting/README.md"),
    ("time_series_overview", "时间序列是什么", "time series introduction", "Time Series", "11-TimeSeries/01-Introduction.md"),
    ("time_series_r", "R 如何处理时间序列", "time series in R", "Time Series", "11-TimeSeries/01-TimeSeriesInR.md"),
    ("csp_overview", "约束满足问题是什么", "constraint satisfaction problems CSP", "Constraint Satisfaction Problems", "12-ConstraintSatisfactionProblems/README.md"),
    ("types_of_data", "数据有哪些类型", "types of data statistics", "Prerequisites", "00-Prerequisites/TypesOfData.md"),
    ("computer_science", "计算机科学基础", "computer science prerequisites", "Prerequisites", "00-Prerequisites/ComputerScience.md"),
    ("statistics", "统计学基础", "statistics prerequisites", "Prerequisites", "00-Prerequisites/Statistics.md"),
    ("linear_algebra", "线性代数基础", "linear algebra prerequisites", "Prerequisites", "00-Prerequisites/LinearAlgebra.md"),
    ("numpy", "NumPy 怎么用", "NumPy Python arrays", "Appendix", "13-Appendix/01-Programming/02-Python/01-Numpy.md"),
    ("scipy", "SciPy 怎么用", "SciPy Python", "Appendix", "13-Appendix/01-Programming/02-Python/04-SciPy.md"),
    ("pandas", "Pandas 怎么用", "Pandas Python dataframes", "Appendix", "13-Appendix/01-Programming/02-Python/03-Pandas.md"),
    ("matplotlib", "Matplotlib 怎么画图", "Matplotlib Python plotting", "Appendix", "13-Appendix/01-Programming/02-Python/02-MatPlotLib.md"),
    ("urllib", "urllib 怎么请求网页", "urllib Python", "Appendix", "13-Appendix/01-Programming/02-Python/05-urllib.md"),
    ("dplyr", "dplyr 怎么处理数据", "dplyr R tutorial", "Appendix", "13-Appendix/01-Programming/01-R/01-DplyrTutorial.md"),
    ("application_intro", "机器学习应用领域", "machine learning application areas introduction", "Appendix", "13-Appendix/02-ApplicationAreas/01-Introduction.md"),
    ("classification_overview", "分类问题概览", "classification machine learning", "Classification", "02-Classification/README.md"),
    ("clustering_overview", "聚类问题概览", "clustering machine learning", "Clustering", "03-Clustering/README.md"),
    ("association_overview", "关联规则概览", "association rule learning", "Association Rule Learning", "04-AssociationRuleLearning/README.md"),
    ("prerequisites_overview", "机器学习先修知识", "machine learning prerequisites", "Prerequisites", "00-Prerequisites/README.md"),
    ("appendix_python", "Python 编程附录", "Python programming appendix", "Appendix", "13-Appendix/01-Programming/02-Python/README.md"),
    ("appendix_r", "R 编程附录", "R programming appendix", "Appendix", "13-Appendix/01-Programming/01-R/README.md"),
    ("programming_overview", "编程附录", "programming appendix", "Appendix", "13-Appendix/01-Programming/README.md"),
)

EVAL_CASES = EVAL_CASES + tuple(
    EvalCase(name, user_query, retrieval_query, expected_topic, expected_path)
    for name, user_query, retrieval_query, expected_topic, expected_path in _ADDITIONAL_CASES
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
