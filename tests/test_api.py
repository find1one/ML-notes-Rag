from contextlib import asynccontextmanager
import time

from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

import api
from api import app
from rag_service import RAGExecution, RAGPrepared


def _source():
    return {
        "id": "S1",
        "title": "Linear Regression",
        "topic": "Regression",
        "section": "Overview",
        "path": "regression.md",
        "score": 0.1,
        "excerpt": "linear regression content",
    }


class FakeService:
    ready = True

    def __init__(self, gate_decision="passed", delay_first_token=False, empty_stream=False):
        self.calls = 0
        self.gate_decision = gate_decision
        self.delay_first_token = delay_first_token
        self.empty_stream = empty_stream

    def execute(self, question):
        prepared = self.prepare(question)
        if prepared.gate_decision != "passed":
            return RAGExecution(
                answer="当前笔记中没有找到足够依据。",
                route_type=prepared.route_type,
                topic=prepared.topic,
                retrieval_query=prepared.retrieval_query,
                sources=prepared.sources,
                metrics=prepared.metrics,
                gate_decision=prepared.gate_decision,
                terminal_event="rejected",
                rejection_reason=prepared.rejection_reason,
            )
        return RAGExecution(
            answer="answer [S1]",
            route_type="detail",
            topic="Regression",
            retrieval_query="linear regression",
            sources=[_source()],
            metrics={"generation_ms": 1},
        )

    def prepare(self, question):
        self.calls += 1
        return RAGPrepared(
            question=question,
            route_type="detail",
            topic="Regression",
            retrieval_query="linear regression",
            docs=[],
            sources=[_source()],
            metrics={"total_retrieval_ms": 1},
            gate_decision=self.gate_decision,
            rejection_reason=None if self.gate_decision == "passed" else "no_non_empty_chunks",
        )

    def stream_answer(self, prepared):
        if self.empty_stream:
            return
        if self.delay_first_token:
            time.sleep(0.2)
        yield "answer "
        yield "[S1]"


class FakeDatabase:
    def __init__(self, fail=False):
        self.records = []
        self.fail = fail

    def create_query(self, **values):
        if self.fail:
            raise SQLAlchemyError("mysql down")
        self.records.append(values)
        return type("Query", (), {"id": len(self.records)})()

    def create_feedback(self, query_id, rating, comment):
        if query_id > len(self.records):
            raise LookupError("query_id does not exist")


@asynccontextmanager
async def no_lifespan(application):
    yield


def _events(text):
    parsed = []
    for raw in text.strip().split("\n\n"):
        lines = raw.splitlines()
        event = lines[0].removeprefix("event: ")
        parsed.append(event)
    return parsed


def _client(service=None, database=None):
    old_lifespan = app.router.lifespan_context
    app.router.lifespan_context = no_lifespan
    app.state.rag_service = service or FakeService()
    app.state.database = database if database is not None else FakeDatabase()
    return old_lifespan, TestClient(app)


def test_stream_contract_done_and_database_failure_degrades_persistence(monkeypatch):
    monkeypatch.setattr("api.get_exact_cache", lambda question: None)
    monkeypatch.setattr("api.set_exact_cache", lambda question, payload: True)
    old_lifespan, client = _client(database=FakeDatabase(fail=True))
    try:
        response = client.post("/v1/chat/stream", json={"question": "线性回归是什么"})
    finally:
        client.close()
        app.router.lifespan_context = old_lifespan
    assert response.status_code == 200
    events = _events(response.text)
    assert events == ["accepted", "retrieval_started", "retrieval_done", "generation_started", "token", "token", "done"]


def test_stream_rejected_when_gate_fails(monkeypatch):
    monkeypatch.setattr("api.get_exact_cache", lambda question: None)
    old_lifespan, client = _client(service=FakeService(gate_decision="rejected"))
    try:
        response = client.post("/v1/chat/stream", json={"question": "未知问题"})
    finally:
        client.close()
        app.router.lifespan_context = old_lifespan
    assert _events(response.text)[-1] == "rejected"


def test_stream_cache_hit_and_fresh_bypass(monkeypatch):
    payload = {
        "answer": "cached answer [S1]",
        "sources": [_source()],
        "route_type": "detail",
        "topic": "Regression",
        "retrieval_query": "linear regression",
    }
    service = FakeService()
    monkeypatch.setattr("api.get_exact_cache", lambda question: payload)
    old_lifespan, client = _client(service=service)
    try:
        hit = client.post("/v1/chat/stream", json={"question": "线性回归是什么"})
        fresh = client.post("/v1/chat/stream", json={"question": "线性回归是什么", "cache_mode": "fresh"})
    finally:
        client.close()
        app.router.lifespan_context = old_lifespan
    assert "cached answer" in hit.text
    assert service.calls == 1
    assert _events(fresh.text)[1] == "retrieval_started"


def test_stream_first_token_timeout_degrades(monkeypatch):
    monkeypatch.setattr("api.get_exact_cache", lambda question: None)
    monkeypatch.setattr(api.DEFAULT_CONFIG, "first_token_wait_seconds", 0)
    monkeypatch.setattr(api.DEFAULT_CONFIG, "first_token_timeout_seconds", 0.05)
    old_lifespan, client = _client(service=FakeService(delay_first_token=True))
    try:
        response = client.post("/v1/chat/stream", json={"question": "线性回归是什么"})
    finally:
        client.close()
        app.router.lifespan_context = old_lifespan
    events = _events(response.text)
    assert "waiting_for_first_token" in events
    assert events[-1] == "degraded"


def test_stream_empty_llm_stream_degrades(monkeypatch):
    monkeypatch.setattr("api.get_exact_cache", lambda question: None)
    old_lifespan, client = _client(service=FakeService(empty_stream=True))
    try:
        response = client.post("/v1/chat/stream", json={"question": "线性回归是什么"})
    finally:
        client.close()
        app.router.lifespan_context = old_lifespan
    assert _events(response.text)[-1] == "degraded"


def test_feedback_accepts_existing_query_and_rejects_unknown_query(monkeypatch):
    monkeypatch.setattr("api.get_exact_cache", lambda question: None)
    monkeypatch.setattr("api.set_exact_cache", lambda question, payload: True)
    old_lifespan, client = _client()
    try:
        stream = client.post("/v1/chat/stream", json={"question": "线性回归是什么"})
        feedback = client.post("/feedback", json={"query_id": 1, "rating": "helpful"})
        invalid = client.post("/feedback", json={"query_id": 99, "rating": "helpful"})
    finally:
        client.close()
        app.router.lifespan_context = old_lifespan
    assert stream.status_code == 200
    assert feedback.status_code == 200
    assert invalid.status_code == 404


def test_health_ready_and_removed_compatibility_routes():
    old_lifespan, client = _client()
    try:
        health = client.get("/health")
        ready = client.get("/ready")
        chat = client.post("/chat", json={"question": "问题"})
        debug = client.post("/chat/debug", json={"question": "问题"})
        query = client.post("/query", json={"question": "问题"})
    finally:
        client.close()
        app.router.lifespan_context = old_lifespan
    assert health.json() == {"status": "ok"}
    assert ready.json()["ready"] is True
    assert chat.status_code == 404
    assert debug.status_code == 404
    assert query.status_code == 404


def test_openapi_exposes_only_business_routes():
    paths = set(app.openapi()["paths"])
    assert paths == {"/v1/chat/stream", "/feedback"}
