import json
import logging
from pathlib import Path

class JsonFormatter(logging.Formatter):
    def format(self,record):
        payload = {
            "level": record.levelname,
            "message": record.getMessage(),
        }
        for field in ["question", "retrieval_query", "top_sources", "latency_ms", "stage", "error"]:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        return json.dumps(payload, ensure_ascii=False)
    
LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "rag_queries.jsonl"

logger = logging.getLogger("rag_logger")
logger.setLevel(logging.INFO)

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

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

def log_rag_query(question, retrieval_query, top_sources, latency_ms, stage, error=None):
    logger.info(
        "RAG logger",
        extra={
            "question": question,
            "retrieval_query": retrieval_query,
            "top_sources": top_sources,
            "latency_ms": latency_ms,
            "stage": stage,
            "error": error,
        },
    )