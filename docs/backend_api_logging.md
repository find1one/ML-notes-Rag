# FastAPI 与 JSONL 日志改进说明

本文档总结本项目在原有 RAG 问答链路基础上新增的 FastAPI 后端接口、错误处理和 JSONL 请求日志能力。这部分改进的目标不是改变检索算法本身，而是让系统更接近一个可调用、可排查、可评估的后端服务。

## 改进背景

原项目已经具备 CLI 和 Streamlit 入口，可以完成：

- 中文问题输入
- query router 判断问题类型
- 中文问题改写为英文 retrieval query
- FAISS + BM25 混合检索
- RRF 融合重排
- 基于检索上下文生成中文回答

但如果只看最终回答，很难判断一次问答失败到底发生在哪个环节。例如：

- query rewrite 是否改偏
- retrieval 是否召回正确 source
- generation 是否返回空回答
- 一次请求具体耗时多久
- 失败时系统已经执行到哪个阶段

因此新增了 FastAPI 接口和结构化日志，用于把 RAG 的中间链路暴露出来。

## FastAPI 接口

后端入口位于：

```text
code/api.py
```

当前主要接口包括：

```text
GET  /health
GET  /ready
POST /chat
POST /chat/debug
```

### /health

用于简单健康检查：

```python
@app.get("/health")
def health_check():
    return {"status": "ok"}
```

它表示服务进程是否能正常响应。

### /ready

用于检查 RAG 系统是否完成初始化：

```python
@app.get("/ready")
def readiness_check():
    return {
        "ready": bool(
            rag_system
            and rag_system.retrieval_module
            and rag_system.generation_module
        )
    }
```

相比 `/health`，`/ready` 更关注知识库、检索模块和生成模块是否已经准备好。

### /chat

普通问答接口，只返回最终答案：

```text
request:  {"question": "..."}
response: {"answer": "..."}
```

它适合前端或外部服务直接调用。

### /chat/debug

调试问答接口，除了答案外，还返回：

- `route_type`
- `retrieval_query`
- `sources`

示例响应结构：

```json
{
  "answer": "...",
  "route_type": "basic",
  "retrieval_query": "linear regression definition overview OLS ordinary least squares",
  "sources": [
    {
      "title": "Linear Regression",
      "topic": "Regression",
      "section": "Linear Regression",
      "path": "01-Regression/01-LinearRegression.md"
    }
  ]
}
```

这个接口可以帮助观察 RAG 中间结果，而不是只看最终生成文本。

## 错误处理

当前后端使用 FastAPI 的 `HTTPException` 处理明确错误。

### 空问题

如果用户传入空问题：

```python
if not question:
    raise HTTPException(status_code=400, detail="Question cannot be empty")
```

返回：

```text
400 Bad Request
```

含义是请求参数不合法。

### LLM 空回答

如果 LLM 多次返回空回答：

```python
def _raise_empty_answer() -> None:
    raise HTTPException(status_code=502, detail="LLM returned an empty response")
```

返回：

```text
502 Bad Gateway
```

含义是后端调用下游 LLM 服务后没有拿到有效结果。

### 异常日志与重新抛出

在 `/chat/debug` 中，异常会先被记录到 JSONL 日志，再继续抛给 FastAPI：

```python
except Exception as exc:
    log_rag_query(
        question=question,
        retrieval_query=retrieval_query,
        top_sources=_format_top_sources(retrieved_docs),
        latency_ms=latency_ms,
        stage=stage,
        error=str(exc),
        retrieval_metrics=retrieval_metrics,
    )
    raise
```

这里的 `raise` 用于保留原始异常处理流程，避免记录日志后把错误吞掉。

## JSONL 日志

日志模块位于：

```text
code/rag_logger.py
```

日志文件输出到：

```text
logs/rag_queries.jsonl
```

JSONL 的特点是：

```text
一行一条 JSON 记录
```

适合请求日志，因为每次请求完成后只需要追加一行，不需要读写整个 JSON 数组。

## 日志字段

当前每条 RAG 请求日志记录以下字段：

```text
level
message
question
retrieval_query
top_sources
latency_ms
stage
error
retrieval_metrics
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `level` | 日志级别，例如 `INFO` |
| `message` | 日志消息，例如 `RAG logger` |
| `question` | 用户原始问题 |
| `retrieval_query` | query rewrite 后实际用于检索的 query |
| `top_sources` | 检索返回的 top source 元数据 |
| `latency_ms` | 本次请求总耗时，单位毫秒 |
| `stage` | 当前请求执行到的阶段 |
| `error` | 错误信息；成功时为 `null` |
| `retrieval_metrics` | 检索阶段分项耗时；未进入检索阶段时为 `null` |

示例：

```jsonl
{"level": "INFO", "message": "RAG logger", "question": "线性回归是什么", "retrieval_query": "linear regression definition overview OLS ordinary least squares", "top_sources": [{"title": "Linear Regression", "topic": "Regression", "section": "Linear Regression", "path": "01-Regression/01-LinearRegression.md"}], "latency_ms": 138806, "stage": "response", "error": null, "retrieval_metrics": {"faiss_ms": 201, "bm25_ms": 0, "rrf_ms": 0, "total_retrieval_ms": 202}}
```

这条日志可以直接说明：

- query rewrite 成功
- retrieval 成功，并且召回了线性回归相关 source
- FAISS 向量检索耗时约 201ms，是这次检索阶段的主要开销
- BM25 和 RRF 融合耗时小于 1ms，取整后显示为 0
- 整个请求耗时约 138 秒，因此端到端瓶颈不在检索，而更可能在 LLM rewrite 或 generation

失败日志示例：

```jsonl
{"level": "INFO", "message": "RAG logger", "question": "", "retrieval_query": null, "top_sources": [], "latency_ms": 0, "stage": "validate", "error": "400: Question cannot be empty", "retrieval_metrics": null}
```

这说明请求还没有进入 route、rewrite 和 retrieval 阶段，就在输入校验阶段失败了。

## stage 字段

为了定位问题发生在哪个环节，日志新增了 `stage` 字段。

当前阶段包括：

```text
validate
route
rewrite
retrieval
generation
response
```

排查时可以这样理解：

| stage | 说明 |
| --- | --- |
| `validate` | 输入校验阶段 |
| `route` | 问题分类阶段 |
| `rewrite` | 中文问题改写英文检索 query 阶段 |
| `retrieval` | FAISS/BM25/RRF 检索阶段 |
| `generation` | LLM 生成答案阶段 |
| `response` | 已准备返回响应 |

例如：

```json
{"stage": "retrieval", "error": "..."}
```

说明问题大概率出在检索模块。

```json
{"stage": "generation", "error": "502: LLM returned an empty response"}
```

说明检索已经完成，问题出在答案生成阶段。

## retrieval_metrics 字段

为了进一步分析混合检索性能，检索模块会统计 FAISS、BM25、RRF 三个阶段的耗时，并将最近一次检索指标保存到 `RetrievalOptimizationModule.last_metrics`。

当前记录的字段包括：

```text
faiss_ms
bm25_ms
rrf_ms
total_retrieval_ms
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `faiss_ms` | FAISS 向量检索耗时 |
| `bm25_ms` | BM25 关键词检索耗时 |
| `rrf_ms` | RRF 融合重排耗时 |
| `total_retrieval_ms` | 一次 hybrid search 的总检索耗时 |

示例：

```json
"retrieval_metrics": {
  "faiss_ms": 201,
  "bm25_ms": 0,
  "rrf_ms": 0,
  "total_retrieval_ms": 202
}
```

这说明在该次检索中，检索阶段主要开销来自 FAISS；但如果整体 `latency_ms` 远大于 `total_retrieval_ms`，则端到端瓶颈通常不在检索，而在 LLM query rewrite 或 answer generation。

`retrieval_metrics` 和普通 `logger.info(...)` 的区别是：

```text
logger.info: 输出到终端，方便开发时观察
retrieval_metrics: 写入 JSONL 业务日志，方便后续统计、复盘和面试展示
```

## top_sources 的处理

检索返回的是 LangChain `Document` 对象，不适合直接写入日志。因此在 `api.py` 中新增了内部辅助函数：

```python
def _format_top_sources(retrieved_docs: list) -> list[dict]:
    sources = []
    for doc in retrieved_docs:
        metadata = doc.metadata
        sources.append({
            "title": metadata.get("title", "Untitled"),
            "topic": metadata.get("topic", "Unknown topic"),
            "section": metadata.get("section_path", "Unknown section"),
            "path": metadata.get("relative_path", metadata.get("source", ""))
        })
    return sources
```

这样日志只保存可读、可分析的 source 元数据，而不是完整文档内容。

## 错误案例验证

目前已经验证过典型错误可以进入 JSONL 日志。

### 空问题

请求：

```json
{"question": ""}
```

日志：

```json
{"question": "", "retrieval_query": null, "top_sources": [], "stage": "validate", "error": "400: Question cannot be empty", "retrieval_metrics": null}
```

说明输入校验失败时，系统可以记录原始问题、当前阶段和错误原因。

### LLM 空回答

当检索成功但 LLM 多次返回空回答时，日志会记录：

```json
{"stage": "generation", "error": "502: LLM returned an empty response"}
```

这说明 query rewrite 和 retrieval 已经完成，问题发生在答案生成阶段。

这类验证的价值在于：日志不仅能记录成功请求，也能在异常场景下说明失败发生在哪个环节。

## 对原项目的价值

这部分改进让项目从一个 RAG demo 更接近后端服务，主要价值包括：

### 1. 支持外部调用

通过 FastAPI，RAG 系统不再只能通过 CLI 或 Streamlit 使用，也可以被前端、脚本或其他服务通过 HTTP 调用。

### 2. 增强可观测性

JSONL 日志记录了用户问题、检索 query、top sources、端到端耗时、错误阶段和检索分项耗时。这样当回答质量不好或请求变慢时，可以定位是 rewrite、FAISS、BM25、RRF 还是 generation 的问题。

### 3. 支持失败 case 复盘

日志中的失败请求可以沉淀为离线评估样本。例如：

- query rewrite 改错的问题
- top sources 召回错误的问题
- generation 空回答的问题

这些样本可以继续加入 `evaluate_retrieval.py` 或后续评估集。

### 4. 改善错误语义

通过 `HTTPException`，后端可以区分：

- `400`：用户输入错误
- `502`：下游 LLM 返回异常或空回答
- `500`：未捕获的系统内部错误

比直接让程序崩溃更适合 API 服务。

### 5. 便于性能分析

`latency_ms` 可以发现请求耗时异常，`retrieval_metrics` 可以进一步拆解检索阶段。例如某次请求总耗时 138 秒，但 `total_retrieval_ms` 只有 202ms，就能判断检索不是主要瓶颈，应该优先排查 LLM 调用、空回答重试或 query rewrite。

## 后续可继续优化

当前日志已经能定位主要阶段，但还可以继续细化：

- 继续记录 `route_ms`、`rewrite_ms`、`generation_ms`
- 将 `/chat` 也接入同样的日志逻辑
- 避免多个入口重复实现 RAG 流程，抽象统一的 `RAGService`
- 将 `log_path`、`max retries`、`top_k` 配置化
- 对 `query_rewrite`、`retrieval`、`generation` 分别增加更明确的异常处理
- 避免日志记录敏感信息，尤其是接入用户私有文档后

## 总结

本次 FastAPI 与 logging 改进主要解决了两个问题：

```text
1. RAG 系统如何作为后端服务被调用
2. RAG 问答失败时如何定位具体环节
```

通过 `/chat/debug` 和 JSONL 日志，系统现在不仅能返回答案，也能记录一次问答从 question 到 retrieval_query、top_sources、retrieval_metrics、generation 的关键过程。这为后续优化检索质量、排查 LLM 生成异常和构建评估闭环提供了基础。
