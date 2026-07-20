# Streamlit 与 FastAPI 联调设计

> 状态：已于 2026-07-16 实施，代码提交为 `2d67b93`。

## 背景

改造前，Streamlit 直接初始化 `MLNotesRAGSystem` 和 `RAGService`，没有调用 FastAPI。其结果是 Streamlit 请求绕过了 `/v1/chat/stream` 中的 Redis exact cache、请求级 debug、MySQL 查询记录和 `query_id`，也无法通过 `/feedback` 提交反馈。即使 Redis 已有同一问题的答案，Streamlit 仍会重新检索和生成。

## 目标

- Streamlit 成为纯 FastAPI 客户端，不再加载本地模型、索引或 RAG 服务。
- 所有问答统一通过 `POST /v1/chat/stream` 完成。
- 用户可以选择优先使用缓存或强制重新生成，并可开启请求级 debug。
- Streamlit 实时展示 SSE 回答、检索来源、终态和缓存命中状态。
- 成功取得 `query_id` 后，用户可以通过 `POST /feedback` 提交评价和可选评论。
- Streamlit rerun 不会重复提交已经完成的问答请求。

## 非目标

- 不增加新的 FastAPI 业务路由。
- 不让 Streamlit 直接读取 Redis 或 MySQL。
- 不在 FastAPI 不可用时回退到本地 RAG。
- 不改变现有 Redis key、缓存 payload 或 SSE 事件合同。
- 不实现鉴权、多用户隔离或线上部署。

## 方案选择

采用“Streamlit 纯 API 客户端”方案。FastAPI 是唯一业务入口，统一负责 Redis 缓存、检索、生成、MySQL 持久化和反馈。相比混合模式，该方案不会形成两套行为；相比前端直读 Redis，该方案不会让 UI 耦合缓存内部格式。

## 架构与组件

### Streamlit UI

负责收集输入、展示过程和结果、维护页面会话状态以及提交反馈。它不导入或实例化 `MLNotesRAGSystem`、`RAGService` 或生成模块，也不检查 `MOONSHOT_API_KEY`。

侧边栏提供：

- FastAPI 地址：读取 `RAG_API_URL`，默认 `http://127.0.0.1:8000`。
- 缓存策略：“优先使用缓存”映射为 `cache_mode=default`；“强制重新生成”映射为 `cache_mode=fresh`。
- Debug 开关：映射为请求体中的 `debug`。

### API 客户端

新增一个与 Streamlit 渲染解耦的轻量 Python 模块，负责：

- 检查 `/ready`。
- 向 `/v1/chat/stream` 发送 JSON 请求并流式读取响应。
- 按 SSE 规范解析 `event` 和 JSON `data` 字段。
- 向 `/feedback` 提交评价。
- 将 HTTP、连接、超时和协议错误转换为清晰的客户端异常。

客户端以事件迭代器形式向 UI 返回数据，以便独立测试解析和请求参数。

### FastAPI

保持现有业务接口：

- `POST /v1/chat/stream`
- `POST /feedback`

保留运维探针 `/health` 和 `/ready`。缓存读取、缓存写入、debug 日志和数据库记录继续由后端完成。

## 问答数据流

1. 用户输入问题，选择缓存策略和 Debug，然后点击提交。
2. Streamlit 清理上一问题对应的临时结果，但保留独立问题已完成的会话状态。
3. API 客户端向 `/v1/chat/stream` 发送 `question`、`cache_mode` 和 `debug`。
4. Streamlit 消费 SSE 事件：
   - `accepted`：记录 `trace_id`。
   - `retrieval_started`：记录是否命中缓存。
   - `retrieval_done`：保存并展示 sources、路由类型、检索查询和 evidence gate；缓存命中时允许这些调试字段为空。
   - `generation_started`：进入回答生成阶段。
   - `waiting_for_first_token`：提示后端仍在等待模型首 token。
   - `token`：追加回答文本并增量刷新页面。
   - `done`：保存 `query_id`、`cached` 和成功终态。
   - `rejected`：保存拒绝原因和 sources，不显示空白回答。
   - `degraded`：保留已收到的回答、sources 和错误原因。
5. 收到终态或流结束后，将完整结果写入 `st.session_state`。
6. 页面因反馈输入等控件 rerun 时，直接渲染会话中的结果，不重新请求问答。

## 反馈数据流

仅当成功终态返回非空 `query_id` 时显示反馈区。用户选择 `helpful` 或 `not_helpful`，可附加最多 2000 字评论，然后提交到 `/feedback`。

反馈成功后在当前会话记录已提交状态并禁用重复提交。若回答成功但数据库不可用导致 `query_id` 为空，页面说明当前无法提交反馈，不影响答案展示。

## 状态模型

每次提交生成一个页面结果对象，至少包含：

- `question`
- `answer`
- `sources`
- `trace_id`
- `query_id`
- `cached`
- `route_type`
- `retrieval_query`
- `gate_decision`
- `terminal_event`
- `reason`
- `feedback_submitted`

只有点击问答提交按钮才发起新请求。缓存策略或 Debug 控件自身变化不会触发请求。

## 异常处理

- FastAPI 不可连接：显示 API 地址和后端启动提示，不尝试本地 RAG。
- `/ready` 返回未就绪：展示 `rag_ready` 和 `database_ready`；RAG 未就绪时阻止问答，数据库未就绪时允许问答但提示反馈可能不可用。
- HTTP 非 2xx：提取 FastAPI `detail`；无法提取时显示状态码和简短响应内容。
- SSE JSON 或事件格式错误：终止本次消费，保留已收到内容并报告协议错误。
- 连接中断或读取超时：保留已收到的回答和来源，标记为客户端错误，并允许用户重新提交。
- `rejected`：展示后端原因和来源，不把它当作网络异常。
- `degraded`：展示后端原因、已生成文本和来源。
- Redis 不可用：沿用 FastAPI 的 best-effort 降级，前端无需特殊分支。
- MySQL 不可用：答案仍正常显示；无 `query_id` 时禁用反馈。

## 测试设计

### API 客户端单元测试

- 请求体正确映射 `default`、`fresh` 和 `debug`。
- 正确解析被拆分为任意网络 chunk 的 SSE 帧。
- 正确处理 `done`、`rejected`、`degraded` 和 `waiting_for_first_token`。
- 正确处理多个 `token`、空 data、非法 JSON、HTTP 错误和连接异常。
- feedback 请求包含 `query_id`、`rating` 和可选 `comment`。

### Streamlit 状态逻辑测试

- 缓存命中结果显示 `cached=true`，并使用后端返回答案。
- `fresh` 请求不会被错误映射为默认缓存模式。
- rerun 渲染已有结果时不再次调用问答 API。
- 只有非空 `query_id` 才允许反馈。
- 成功反馈后禁止同一会话重复提交。

### 回归测试

- 保留现有 FastAPI 和缓存测试。
- 运行完整 pytest 套件，确保 API 合同与 RAG 行为未改变。

## 文档与运行方式

README 增加明确的启动顺序：先启动基础设施和 FastAPI，确认 `/ready`，再启动 Streamlit。示例配置：

```bash
export RAG_API_URL=http://127.0.0.1:8000
```

Moonshot、Redis、MySQL 和索引配置只由 FastAPI 进程使用。Streamlit 只需要能够访问 `RAG_API_URL`。

## 验收标准

- Redis 已缓存某问题时，Streamlit 使用默认缓存策略可直接显示缓存答案，后端不调用检索或生成。
- 强制重新生成时，后端绕过 Redis 读取并执行完整 RAG。
- Debug 开关值准确传到 FastAPI。
- Streamlit 能实时展示 token、来源、终态和缓存命中状态。
- 返回 `query_id` 后可提交一次反馈，并能看到成功或明确失败提示。
- Streamlit rerun 不会重复运行同一已完成请求。
- FastAPI 不可用时，前端提供可操作的错误信息。
- 新增测试和现有测试全部通过。
