# ML Notes RAG 系统架构概览

本文说明当前代码库实际运行的系统边界。安装、环境变量和命令以根目录 [README](../../README.md) 为准；接口终态、日志字段和降级细节见 [API 与可观测性](api-observability.md)。

## 1. 系统定位

项目将英文机器学习 Markdown 笔记构建为可检索知识库，支持中文 query 检索英文资料，并基于检索证据生成带来源的中文回答。

系统由两条边界清楚的链路组成：

- **离线平面**：读取语料、分块、构建索引、运行检索评测、显式发布验证通过的索引；
- **在线平面**：加载已发布索引，执行规则路由、混合检索、证据门控和流式生成。

API 不在启动时构建索引。Streamlit 也不直接初始化模型、索引、Redis 或 MySQL。

## 2. 离线索引生命周期

```text
English Markdown
  → metadata extraction
  → Markdown heading-aware chunking
  → recursive split for long sections
  → multilingual embeddings
  → FAISS index
  → frozen retrieval evaluation
  → passed manifest
  → explicit publish
```

`DataPreparationModule` 负责读取 Markdown、提取标题和目录元数据，并生成带稳定 ID 的 chunk。主要元数据包括：

- `title`
- `topic`
- `chapter`
- `relative_path`
- `section_path`
- `parent_id`
- `chunk_id`

`IndexConstructionModule` 使用多语 embedding 构建 FAISS 索引。发布命令只有在离线检索门槛通过后才保存索引并写入 `passed` manifest：

```bash
python code/build_index.py --publish
```

在线服务加载索引时会校验语料、embedding 模型、chunk 状态和 manifest。FAISS 文件能够打开，不代表它与当前语料和配置匹配。

## 3. 在线问答链路

```text
Chinese question
  → deterministic route: list / detail / general
  → deterministic term expansion and topic detection
  → FAISS + BM25 retrieval or topic document lookup
  → stable chunk ID RRF and metadata signals
  → evidence gate
  → Moonshot generation
  → SSE answer, sources and optional query_id
```

### 3.1 路由与 query 处理

在线链路使用确定性规则识别 list、detail 和 general query，并对已知中文术语补充英文检索词。路由和 query expansion 不额外调用 LLM。

列表类问题可根据 topic 元数据直接列出父文档；detail/general query 进入混合检索。

### 3.2 混合检索

`RetrievalOptimizationModule` 组合：

- FAISS 语义检索；
- BM25 关键词检索；
- 基于稳定 `chunk_id` 的 RRF；
- 标题、路径和 topic 元数据匹配；
- topic 优先和候选不足回退。

RRF 使用排名而不是直接混合 FAISS 与 BM25 的原始分数，避免不同分数尺度造成不稳定融合。

### 3.3 Evidence gate 与生成

`RAGService` 在生成前判断检索证据是否足够：

- 证据不足：不调用 LLM，以 `rejected` 终止并返回来源；
- 证据充分：调用 Moonshot 流式生成，并用 `[S1]` 等来源编号组织引用。

检索排序提升不等同于最终答案质量提升。当前离线 benchmark 主要测量 source/topic 排名；答案正确性、evidence coverage 和 citation faithfulness 需要单独标签和评测。

## 4. API 与客户端边界

唯一在线问答业务入口是：

```text
POST /v1/chat/stream
```

反馈入口是：

```text
POST /feedback
```

`GET /health` 与 `GET /ready` 仅作为运维探针。

Streamlit 是纯 FastAPI SSE 客户端：

- 只读取 `RAG_API_URL`；
- 不直接加载 RAG、embedding 或索引；
- 不直接访问 Redis 或 MySQL；
- 只有获得有效 `query_id` 时才允许提交反馈；
- 控件 rerun 不会重复执行已经完成的问答请求。

## 5. 缓存、持久化与日志

在线主链路之外还有三类 best-effort 支撑：

- Redis：exact cache；不可用时跳过缓存；
- MySQL：`query_logs` 与 `feedback`；写入失败不阻断回答；
- JSONL：记录 trace、终态、来源和检索/生成耗时。

默认日志不保存原始问题、答案或 excerpts；请求设置 `debug=true` 时才记录这些调试内容。详细字段见 [API 与可观测性](api-observability.md)。

## 6. 代码职责

| 位置 | 当前职责 |
| --- | --- |
| `code/api.py` | FastAPI、SSE、缓存与反馈入口 |
| `code/api_client.py` | Streamlit 使用的 HTTP/SSE 客户端 |
| `code/streamlit_app.py` | 纯 API 客户端 UI |
| `code/rag_service.py` | 路由、检索、证据门控与生成编排 |
| `code/rag_modules/data_preparation.py` | 文档加载、元数据和分块 |
| `code/rag_modules/index_construction.py` | embedding、FAISS 与 manifest |
| `code/rag_modules/retrieval_optimization.py` | FAISS、BM25、RRF 和过滤 |
| `code/rag_modules/generation_integration.py` | Moonshot prompt 与流式生成 |
| `code/build_index.py` | 离线构建、评测与显式发布 |
| `code/evaluate_retrieval.py` | 单层检索评测入口 |
| `code/evaluate_parent_child.py` | Parent-Child 离线消融入口 |

这些路径是当前演示和运行入口。评测代码后续可能独立到顶层 `evaluation/`，但当前阶段没有执行该迁移。

## 7. 当前实验边界

当前权威检索结果记录在 [检索实验结果](../experiments/retrieval/results.md)：

- 120 条 source-evaluable 主 case；
- 15 条数据质量诊断；
- 已完成 chunk size、PC0、PC1、PC2 与 H1 消融；
- PC2 通过离线门槛，但尚未发布生产索引；
- H1 提升 Top-3 覆盖但降低 Source MRR@3，因此未采用。

任何历史设计中的 35-case 或 50-case 数字都只是当时快照，不代表当前评测口径。

## 8. 继续阅读

- [项目安装与运行](../../README.md)
- [API 与可观测性](api-observability.md)
- [检索实验结果](../experiments/retrieval/results.md)
- [PARADE 与项目对照案例](../research/parade/case-study.md)
