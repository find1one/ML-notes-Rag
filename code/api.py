"""FastAPI application for the ML Notes RAG service."""

from contextlib import asynccontextmanager
import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from cache import get_exact_cache, set_exact_cache
from config import DEFAULT_CONFIG
from database import Database
from main import MLNotesRAGSystem
from rag_logger import log_persistence_fallback, log_rag_query
from rag_service import RAGExecution, RAGPrepared, RAGService

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class StreamChatRequest(ChatRequest):
    cache_mode: Literal["default", "fresh"] = "default"
    debug: bool = False


class FeedbackRequest(BaseModel):
    query_id: int = Field(gt=0)
    rating: Literal["helpful", "not_helpful"]
    comment: Optional[str] = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    ok: bool = True


TERMINAL_EVENTS = {"done", "rejected", "degraded", "cancelled"}


def _get_service(request: Request) -> RAGService:
    service = getattr(request.app.state, "rag_service", None)
    if service is None or not service.ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RAG service is not ready")
    return service


def _get_database(request: Request) -> Database:
    database = getattr(request.app.state, "database", None)
    if database is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database is not ready")
    return database


def _write_query_best_effort(
    request: Request,
    trace_id: str,
    question: str,
    execution: RAGExecution,
    latency_ms: int,
    cached: bool,
    debug: bool = False,
) -> Optional[int]:
    database = getattr(request.app.state, "database", None)
    if database is None:
        log_persistence_fallback(trace_id, question, execution.answer, execution.sources, "database_not_ready", debug=debug)
        return None
    try:
        record = database.create_query(
            question=question if debug else "",
            retrieval_query=execution.retrieval_query,
            route_type=execution.route_type,
            topic=execution.topic,
            answer=execution.answer if debug else "",
            sources_json=json.dumps(execution.sources, ensure_ascii=False),
            latency_ms=latency_ms,
            cached=cached,
            cache_type="exact" if cached else None,
            similarity=None,
        )
        return record.id
    except SQLAlchemyError as exc:
        logger.warning("Failed to persist query log; continuing", exc_info=True)
        log_persistence_fallback(trace_id, question, execution.answer, execution.sources, exc, debug=debug)
        return None


def _sse(event: str, payload: Optional[dict] = None) -> str:
    data = json.dumps(payload or {}, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


def _cache_execution(payload: dict) -> RAGExecution:
    return RAGExecution(
        answer=payload["answer"],
        route_type=payload["route_type"],
        topic=payload.get("topic"),
        retrieval_query=payload.get("retrieval_query"),
        sources=payload["sources"],
        metrics={"cache_hit": 1},
        gate_decision=payload.get("gate_decision", "passed"),
        terminal_event="done",
    )


def _cache_payload(execution: RAGExecution) -> dict:
    return {
        "answer": execution.answer,
        "sources": execution.sources,
        "route_type": execution.route_type,
        "topic": execution.topic,
        "retrieval_query": execution.retrieval_query,
        "gate_decision": execution.gate_decision,
    }


def _token_chunks(answer: str, size: int = 24):
    for start in range(0, len(answer), size):
        yield answer[start:start + size]


def _stream_answer_with_first_token_timeout(service: RAGService, prepared: RAGPrepared):
    output: queue.Queue[tuple[str, Optional[str]]] = queue.Queue()

    def worker():
        try:
            for chunk in service.stream_answer(prepared):
                if chunk:
                    output.put(("token", str(chunk)))
            output.put(("done", None))
        except Exception as exc:  # pragma: no cover - exercised through API tests with fakes
            output.put(("error", str(exc)))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    start = time.perf_counter()
    waiting_sent = False
    first_token_seen = False
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= DEFAULT_CONFIG.orphan_request_timeout_seconds:
            yield ("degraded", {
                "reason": "orphan_request_timeout",
                "sources": prepared.sources,
                "excerpts": [source.get("excerpt") for source in prepared.sources],
            })
            return
        if not first_token_seen and not waiting_sent and elapsed >= DEFAULT_CONFIG.first_token_wait_seconds:
            waiting_sent = True
            yield ("waiting_for_first_token", {"elapsed_ms": int(elapsed * 1000)})
        if not first_token_seen and elapsed >= DEFAULT_CONFIG.first_token_timeout_seconds:
            yield ("degraded", {
                "reason": "first_token_timeout",
                "sources": prepared.sources,
                "excerpts": [source.get("excerpt") for source in prepared.sources],
            })
            return
        timeout = 0.25 if not first_token_seen else DEFAULT_CONFIG.stream_idle_timeout_seconds
        try:
            kind, value = output.get(timeout=timeout)
        except queue.Empty:
            if first_token_seen:
                yield ("degraded", {
                    "reason": "stream_idle_timeout",
                    "sources": prepared.sources,
                    "excerpts": [source.get("excerpt") for source in prepared.sources],
                })
                return
            continue
        if kind == "token":
            first_token_seen = True
            yield ("token", {"text": value})
        elif kind == "done":
            return
        elif kind == "error":
            yield ("degraded", {
                "reason": "llm_stream_error",
                "error": value,
                "sources": prepared.sources,
                "excerpts": [source.get("excerpt") for source in prepared.sources],
            })
            return


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_service = None
    app.state.database = None
    try:
        rag = MLNotesRAGSystem(DEFAULT_CONFIG)
        rag.initialize_system()
        rag.load_knowledge_base()
        app.state.rag_service = RAGService(rag)
    except Exception:
        logger.exception("RAG initialization failed")
    try:
        database = Database(DEFAULT_CONFIG.database_url, DEFAULT_CONFIG.database_connect_timeout)
        database.initialize()
        app.state.database = database
    except Exception:
        logger.exception("Database initialization failed")
    yield


app = FastAPI(title="ML Notes RAG API", version="2.0.0", lifespan=lifespan)


@app.get("/health", include_in_schema=False)
def health_check():
    return {"status": "ok"}


@app.get("/ready", include_in_schema=False)
def readiness_check(request: Request):
    service = getattr(request.app.state, "rag_service", None)
    database = getattr(request.app.state, "database", None)
    rag_ready = bool(service and service.ready)
    return {"ready": rag_ready, "rag_ready": rag_ready, "database_ready": bool(database)}


@app.post("/v1/chat/stream")
async def chat_stream(request: StreamChatRequest, http_request: Request):
    trace_id = uuid.uuid4().hex
    start = time.perf_counter()
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    service = _get_service(http_request)

    async def events():
        answer_parts: list[str] = []
        cached = False
        execution: Optional[RAGExecution] = None
        terminal_event = "cancelled"
        prepared: Optional[RAGPrepared] = None
        try:
            yield _sse("accepted", {"trace_id": trace_id})
            cached_payload = None if request.cache_mode == "fresh" else get_exact_cache(question)
            if cached_payload:
                cached = True
                execution = _cache_execution(cached_payload)
                yield _sse("retrieval_started", {"cached": True})
                yield _sse("retrieval_done", {"sources": execution.sources, "cached": True})
                yield _sse("generation_started", {"cached": True})
                for chunk in _token_chunks(execution.answer):
                    answer_parts.append(chunk)
                    yield _sse("token", {"text": chunk})
                    await asyncio.sleep(0)
                terminal_event = "done"
                latency_ms = int((time.perf_counter() - start) * 1000)
                query_id = _write_query_best_effort(http_request, trace_id, question, execution, latency_ms, cached, request.debug)
                yield _sse("done", {"query_id": query_id, "cached": True})
                return

            yield _sse("retrieval_started", {"cached": False})
            prepared = service.prepare(question)
            yield _sse("retrieval_done", {
                "sources": prepared.sources,
                "route_type": prepared.route_type,
                "retrieval_query": prepared.retrieval_query,
                "gate_decision": prepared.gate_decision,
            })
            if prepared.gate_decision != "passed":
                execution = RAGExecution(
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
                terminal_event = "rejected"
                latency_ms = int((time.perf_counter() - start) * 1000)
                _write_query_best_effort(http_request, trace_id, question, execution, latency_ms, cached, request.debug)
                yield _sse("rejected", {
                    "reason": prepared.rejection_reason,
                    "sources": prepared.sources,
                })
                return

            yield _sse("generation_started", {"cached": False})
            degraded_payload = None
            for event_name, payload in _stream_answer_with_first_token_timeout(service, prepared):
                if event_name == "token":
                    answer_parts.append(payload["text"])
                elif event_name == "degraded":
                    degraded_payload = payload
                    terminal_event = "degraded"
                    yield _sse("degraded", payload)
                    break
                yield _sse(event_name, payload)
                await asyncio.sleep(0)
            if degraded_payload is not None:
                execution = RAGExecution(
                    answer="",
                    route_type=prepared.route_type,
                    topic=prepared.topic,
                    retrieval_query=prepared.retrieval_query,
                    sources=prepared.sources,
                    metrics=prepared.metrics,
                    gate_decision=prepared.gate_decision,
                    terminal_event="degraded",
                    rejection_reason=degraded_payload.get("reason"),
                )
                _write_query_best_effort(
                    http_request,
                    trace_id,
                    question,
                    execution,
                    int((time.perf_counter() - start) * 1000),
                    cached,
                    request.debug,
                )
                return
            answer = "".join(answer_parts).strip()
            if not answer:
                terminal_event = "degraded"
                execution = RAGExecution(
                    answer="",
                    route_type=prepared.route_type,
                    topic=prepared.topic,
                    retrieval_query=prepared.retrieval_query,
                    sources=prepared.sources,
                    metrics=prepared.metrics,
                    gate_decision=prepared.gate_decision,
                    terminal_event="degraded",
                    rejection_reason="llm_empty_stream",
                )
                _write_query_best_effort(
                    http_request,
                    trace_id,
                    question,
                    execution,
                    int((time.perf_counter() - start) * 1000),
                    cached,
                    request.debug,
                )
                yield _sse("degraded", {
                    "reason": "llm_empty_stream",
                    "sources": prepared.sources,
                    "excerpts": [source.get("excerpt") for source in prepared.sources],
                })
                return
            execution = RAGExecution(
                answer=answer,
                route_type=prepared.route_type,
                topic=prepared.topic,
                retrieval_query=prepared.retrieval_query,
                sources=prepared.sources,
                metrics=prepared.metrics,
                gate_decision=prepared.gate_decision,
                terminal_event="done",
            )
            if answer:
                set_exact_cache(question, _cache_payload(execution))
            terminal_event = "done"
            latency_ms = int((time.perf_counter() - start) * 1000)
            query_id = _write_query_best_effort(http_request, trace_id, question, execution, latency_ms, cached, request.debug)
            yield _sse("done", {"query_id": query_id, "cached": False})
        except asyncio.CancelledError:
            terminal_event = "cancelled"
            if prepared is not None:
                execution = RAGExecution(
                    answer="".join(answer_parts),
                    route_type=prepared.route_type,
                    topic=prepared.topic,
                    retrieval_query=prepared.retrieval_query,
                    sources=prepared.sources,
                    metrics=prepared.metrics,
                    gate_decision=prepared.gate_decision,
                    terminal_event="cancelled",
                )
                _write_query_best_effort(http_request, trace_id, question, execution, int((time.perf_counter() - start) * 1000), cached, request.debug)
            raise
        except Exception as exc:
            terminal_event = "degraded"
            sources = prepared.sources if prepared is not None else []
            metrics = prepared.metrics if prepared is not None else {}
            execution = RAGExecution(
                answer="",
                route_type=prepared.route_type if prepared is not None else "unknown",
                topic=prepared.topic if prepared is not None else None,
                retrieval_query=prepared.retrieval_query if prepared is not None else None,
                sources=sources,
                metrics=metrics,
                gate_decision=prepared.gate_decision if prepared is not None else "unknown",
                terminal_event="degraded",
                rejection_reason=str(exc),
            )
            _write_query_best_effort(http_request, trace_id, question, execution, int((time.perf_counter() - start) * 1000), cached, request.debug)
            yield _sse("degraded", {
                "reason": "service_error",
                "error": str(exc),
                "sources": sources,
                "excerpts": [source.get("excerpt") for source in sources],
            })
        finally:
            if execution is not None:
                log_rag_query(
                    question=question,
                    retrieval_query=execution.retrieval_query,
                    top_sources=execution.sources,
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    stage="response",
                    error=execution.rejection_reason,
                    retrieval_metrics=execution.metrics,
                    trace_id=trace_id,
                    route_type=execution.route_type,
                    gate_decision=execution.gate_decision,
                    terminal_event=terminal_event,
                    answer=execution.answer,
                    debug=request.debug,
                )

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest, http_request: Request) -> FeedbackResponse:
    try:
        _get_database(http_request).create_feedback(request.query_id, request.rating, request.comment)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        logger.exception("Failed to persist feedback")
        raise HTTPException(status_code=503, detail="Failed to persist feedback") from exc
    return FeedbackResponse()
