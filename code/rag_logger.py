import hashlib
import json
import logging
from pathlib import Path

class JsonFormatter(logging.Formatter):
    def format(self,record):
        payload = {
            "level": record.levelname,
            "message": record.getMessage(),
        }
        for field in [
            "trace_id",
            "question",
            "question_hash",
            "answer",
            "retrieval_query",
            "retrieval_query_hash",
            "top_sources",
            "latency_ms",
            "stage",
            "terminal_event",
            "error",
            "retrieval_metrics",
            "route_type",
            "gate_decision",
            "excerpts",
        ]:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        return json.dumps(payload, ensure_ascii=False)
    
LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "rag_queries.jsonl"

logger = logging.getLogger("rag_logger")
logger.setLevel(logging.INFO)

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

if not logger.handlers:
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

## 调试
# logger.info(
#     "RAG logger",
#     extra={
#         "question": "线性回归是什么？",
#         "retrieval_query": "linear regression",
#         "top_sources": ["01-Regression/01-LinearRegression.md"],
#         "latency_ms": 820,
#         "stage": "debug",
#         "error": None,
#         },
# )

def _hash_text(text):
    if text is None:
        return None
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _safe_sources(top_sources):
    safe = []
    for source in top_sources or []:
        safe.append({
            key: source.get(key)
            for key in ["id", "title", "topic", "section", "path", "score"]
            if key in source
        })
    return safe


def log_rag_query(
    question,
    retrieval_query,
    top_sources,
    latency_ms,
    stage,
    error=None,
    retrieval_metrics=None,
    trace_id=None,
    route_type=None,
    gate_decision=None,
    terminal_event=None,
    answer=None,
    debug=False,
):
    extra = {
        "trace_id": trace_id,
        "question_hash": _hash_text(question),
        "retrieval_query_hash": _hash_text(retrieval_query),
        "top_sources": _safe_sources(top_sources),
        "latency_ms": latency_ms,
        "stage": stage,
        "terminal_event": terminal_event,
        "error": error,
        "retrieval_metrics": retrieval_metrics,
        "route_type": route_type,
        "gate_decision": gate_decision,
    }
    if debug:
        extra.update({
            "question": question,
            "retrieval_query": retrieval_query,
            "answer": answer,
            "excerpts": [source.get("excerpt") for source in top_sources or []],
        })
    logger.info(
        "RAG logger",
        extra=extra,
    )


def log_persistence_fallback(trace_id, question, answer, sources, error, debug=False):
    log_rag_query(
        question=question,
        retrieval_query=None,
        top_sources=sources,
        latency_ms=0,
        stage="persistence_fallback",
        error=str(error),
        retrieval_metrics=None,
        trace_id=trace_id,
        terminal_event="persistence_failed",
        answer=answer,
        debug=debug,
    )
