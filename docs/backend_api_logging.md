# FastAPI SSE 与 JSONL 日志说明

本文档记录当前面试展示型 RAG 后端的接口和日志约定。主目标是让一次问答的检索、证据门控、生成和降级结果都可观察，并且不让 Redis、MySQL 或 LLM 故障破坏主链路。

## 接口

主要入口：

```text
POST /v1/chat/stream
POST /feedback
```

`GET /health` 和 `GET /ready` 是从 OpenAPI schema 隐藏的运维探针。

`/v1/chat/stream` 是用户入口，返回 Server-Sent Events：

```text
accepted
retrieval_started
retrieval_done {sources}
generation_started
waiting_for_first_token
token
done | rejected | degraded | cancelled
```

调试信息通过 `/v1/chat/stream` 请求体的 `debug=true` 开启，不再提供单独的同步或 debug 接口。

## 降级策略

- 证据门控失败时不调用 LLM，直接返回 `rejected`，并携带最相关 sources。
- 进入 LLM 流式生成后，10 秒无首 token 发送 `waiting_for_first_token`。
- 30 秒仍无首 token 时返回 `degraded`，携带 sources 和 excerpts。
- Redis 读写失败跳过 cache。
- MySQL 写入失败时 `query_id=null`，并追加 JSONL fallback。
- 客户端断开时记录 `cancelled`。

## 日志字段

日志文件：

```text
logs/rag_queries.jsonl
```

默认日志只记录 trace 元数据和来源，不保存原始问题、答案和 excerpts。`debug=true` 时才会写入调试内容。

核心字段：

```text
trace_id
question_hash
retrieval_query_hash
top_sources
latency_ms
stage
terminal_event
error
retrieval_metrics
route_type
gate_decision
```

`retrieval_metrics` 常见字段：

```text
route_ms
expand_ms
faiss_ms
bm25_ms
rrf_ms
total_retrieval_ms
generation_ms
source_count
```

示例：

```json
{
  "level": "INFO",
  "message": "RAG logger",
  "trace_id": "abc",
  "question_hash": "...",
  "retrieval_query_hash": "...",
  "top_sources": [
    {
      "id": "S1",
      "title": "Linear Regression",
      "topic": "Regression",
      "section": "Linear Regression",
      "path": "01-Regression/01-LinearRegression.md",
      "score": 0.04
    }
  ],
  "latency_ms": 1234,
  "stage": "response",
  "terminal_event": "done",
  "error": null,
  "route_type": "detail",
  "gate_decision": "passed",
  "retrieval_metrics": {
    "route_ms": 0,
    "expand_ms": 0,
    "faiss_ms": 24,
    "bm25_ms": 3,
    "rrf_ms": 1,
    "total_retrieval_ms": 29,
    "generation_ms": 1180,
    "source_count": 4
  }
}
```
