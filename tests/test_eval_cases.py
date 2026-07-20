import re
from collections import Counter
from pathlib import Path

from config import DEFAULT_CONFIG
from evaluate_retrieval import DATA_QUALITY_CASES, EVAL_CASES
from rag_modules import DataPreparationModule


def _normalized(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def _indexable_documents():
    module = DataPreparationModule(DEFAULT_CONFIG.data_path)
    documents = module.load_documents()
    return {document.metadata["relative_path"]: document for document in documents}


def test_primary_eval_set_has_120_source_evaluable_cases():
    data_root = Path(DEFAULT_CONFIG.data_path)

    assert len(EVAL_CASES) == 120
    assert all((data_root / case.expected_path_contains).is_file() for case in EVAL_CASES)
    assert all(
        (data_root / case.expected_path_contains).stat().st_size >= 100
        for case in EVAL_CASES
    )


def test_every_indexable_source_has_at_least_three_cases():
    documents = _indexable_documents()
    cases_by_path = Counter(case.expected_path_contains for case in EVAL_CASES)

    assert set(cases_by_path) == set(documents)
    assert min(cases_by_path.values()) >= 3


def test_primary_case_topics_match_source_metadata():
    documents = _indexable_documents()

    assert all(
        case.expected_topic
        == documents[case.expected_path_contains].metadata["topic"]
        for case in EVAL_CASES
    )


def test_primary_case_names_and_queries_are_unique():
    names = [case.name for case in EVAL_CASES]
    user_queries = [_normalized(case.user_query) for case in EVAL_CASES]
    retrieval_queries = [_normalized(case.retrieval_query) for case in EVAL_CASES]

    assert len(names) == len(set(names))
    assert len(user_queries) == len(set(user_queries))
    assert len(retrieval_queries) == len(set(retrieval_queries))


def test_data_quality_cases_remain_outside_primary_metrics():
    data_root = Path(DEFAULT_CONFIG.data_path)
    primary_names = {case.name for case in EVAL_CASES}

    assert len(DATA_QUALITY_CASES) == 15
    assert primary_names.isdisjoint(case.name for case in DATA_QUALITY_CASES)
    assert all(
        not (data_root / case.expected_path_contains).is_file()
        or (data_root / case.expected_path_contains).stat().st_size < 100
        for case in DATA_QUALITY_CASES
    )
