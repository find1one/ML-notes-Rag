from types import SimpleNamespace

from langchain_core.documents import Document
import pytest

from rag_service import RAGPrepared, RAGService, RAGUnavailableError


class FakeGeneration:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.called = 0

    def generate_step_by_step_answer(self, question, docs):
        self.called += 1
        return next(self.answers)


def test_generation_calls_llm_once():
    system = SimpleNamespace(
        config=SimpleNamespace(),
        generation_module=FakeGeneration(["usable answer"]),
        data_module=SimpleNamespace(get_parent_documents=lambda docs: docs),
    )
    service = RAGService(system)
    prepared = RAGPrepared("question", "detail", None, "question", [], [], {}, "passed")
    assert service.generate_answer(prepared) == "usable answer"
    assert system.generation_module.called == 1


def test_empty_generation_raises_without_retrying():
    system = SimpleNamespace(
        config=SimpleNamespace(),
        generation_module=FakeGeneration([""]),
        data_module=SimpleNamespace(get_parent_documents=lambda docs: docs),
    )
    service = RAGService(system)
    prepared = RAGPrepared("question", "detail", None, "question", [], [], {}, "passed")
    with pytest.raises(RAGUnavailableError, match="empty response"):
        service.generate_answer(prepared)
    assert system.generation_module.called == 1


def test_route_and_expansion_are_deterministic_without_llm():
    service = RAGService(SimpleNamespace())
    outputs = [(service._route("线性回归怎么做变量选择？"), service._expand_query("线性回归怎么做变量选择？")) for _ in range(5)]
    assert len(set(outputs)) == 1
    route, expanded = outputs[0]
    assert route == "detail"
    assert "linear regression" in expanded
    assert "variable selection" in expanded


def test_gate_rejects_empty_evidence_without_generation():
    generation = FakeGeneration(["should not be used"])
    system = SimpleNamespace(
        config=SimpleNamespace(top_k=1, candidate_k=1),
        generation_module=generation,
        data_module=SimpleNamespace(),
        retrieval_module=SimpleNamespace(
            hybrid_search=lambda *args, **kwargs: [Document(page_content=" ", metadata={"topic": "Regression"})],
            metadata_filtered_search=lambda *args, **kwargs: [
                Document(page_content=" ", metadata={"topic": "Regression"})
            ],
            last_metrics={},
        ),
    )
    service = RAGService(system)
    execution = service.execute("线性回归是什么")
    assert execution.terminal_event == "rejected"
    assert generation.called == 0


def test_gate_passes_on_topic_or_english_term():
    service = RAGService(SimpleNamespace())
    doc = Document(
        page_content="Ordinary least squares estimates linear regression coefficients.",
        metadata={"topic": "Regression", "title": "Linear Regression", "relative_path": "regression.md"},
    )
    sources = service.format_sources([doc])
    assert service._gate_evidence("linear regression", "Regression", [doc], sources)[0] == "passed"
    assert service._gate_evidence("ordinary least squares", None, [doc], sources)[0] == "passed"
