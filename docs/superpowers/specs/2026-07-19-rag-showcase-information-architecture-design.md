# RAG 展示网站信息架构调整设计

> 状态说明（2026-07-19）：本文是展示页的信息架构设计，文中的 35-case 指标是设计时快照。当前 120-case 指标和 Parent–Child 消融结果以 `README.md` 与 `docs/retrieval_experiment_log.md` 为准。

## 目标

将项目展示网站从“指标、混合链路、工程能力”的并列介绍，调整为围绕系统生命周期展开的四段式叙事：

1. 保留现有模拟问答展示；
2. 介绍离线平面如何构建、评估并发布索引；
3. 介绍在线平面如何消费 verified index 并返回可追溯回答；
4. 展示当前 Retrieval Evaluation 数据，并为后续补充检索性能优化过程保留清晰扩展点。

网站继续同时面向技术开发者与潜在雇主。信息应足够准确以支持技术追问，也应让非项目参与者快速理解系统价值与工程边界。

## 页面结构

### Part 1：模拟问答展示

现有双栏 Hero、三组模拟问题、检索 Trace、生成答案与来源展示保持不变。Header 导航调整为：

- `Offline Plane`
- `Online Plane`
- `Evaluation`

`SYSTEM READY` 继续定位到首屏模拟终端。页内导航使用浏览器原生锚点滚动，不由框架导航接管。

### Part 2：Offline Plane

用一条构建流水线表达离线知识库生命周期：

```text
English Markdown
  -> metadata extraction + Markdown header-aware chunking
  -> 255 chunks with stable IDs and traceable metadata
  -> multilingual MiniLM embeddings
  -> FAISS index
  -> 50-case offline retrieval evaluation
  -> publish verified index with passed manifest
```

这一部分重点说明：索引不是构建完成就直接上线。`python code/build_index.py --publish` 会先构建索引并执行离线评估，只有达到发布阈值才保存可供在线服务加载的索引，并将 manifest 标记为 `passed`。

需要展示的真实项目信息包括：

- 知识源为英文 Markdown 笔记；
- 先按 Markdown 标题切分，长块再递归切分；
- chunk 保留 `topic`、`chapter`、`section_path`、`relative_path`、`parent_id` 和稳定 `chunk_id`；
- 当前 verified index 包含 255 个 chunk；
- Embedding 模型为 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`；
- 评估集共 50 个 case，其中 35 个对应可索引 source。

### Offline–Online Bridge：Verified Index Contract

Offline Plane 与 Online Plane 之间增加醒目的 `VERIFIED INDEX CONTRACT` 连接区，展示在线服务安全消费离线产物所依赖的三个合同：

1. **Same chunking logic**：在线加载时依据当前语料与分块逻辑重新计算期望状态；
2. **Same embedding model**：文档向量与查询向量必须位于相同向量空间；
3. **Passed manifest + matching corpus fingerprint**：索引必须通过评估，且 chunk 数、topic、语料指纹、模型和 schema 与当前运行环境一致。

该桥梁需要明确表达：“FAISS 文件可以打开”不等于“索引可以安全上线”。API 只加载验证通过且匹配当前语料与配置的索引。

### Part 3：Online Plane

用请求时间线表达在线问答主链路：

```text
Chinese question
  -> deterministic route + term expansion
  -> topic detection
  -> FAISS + BM25 retrieval / topic document lookup
  -> stable chunk ID deduplication + RRF + metadata boost
  -> evidence gate
  -> Moonshot generation
  -> SSE answer + sources + query_id
```

这一部分必须遵守项目真实边界：

- 唯一在线问答入口是 `POST /v1/chat/stream`；
- Streamlit 是纯 FastAPI 客户端，不直接初始化 RAG、模型、索引或访问 Redis/MySQL；
- 路由和术语扩展使用确定性规则，不调用 LLM；
- 列表类 topic 查询可直接返回父文档，detail/general 查询走混合检索；
- evidence gate 在生成前执行，证据不足时返回 `rejected`，不调用 Moonshot；
- SSE 可返回 `accepted`、`retrieval_started`、`retrieval_done`、`generation_started`、`token` 和单一终态；
- 成功终态返回来源和可选 `query_id`。

Redis exact cache、MySQL query log 与 JSONL 日志作为主链路旁路支撑展示：

- Redis 与 MySQL 写入均为 best-effort，不阻塞回答；
- JSONL 使用 `trace_id` 串联路由、检索、生成与终态；
- 没有有效 `query_id` 时不能提交反馈。

### Part 4：Retrieval Evaluation

当前版本只展示已有数据：

- Top-1 Source Accuracy：88.6%；
- Top-3 Source Accuracy：91.4%；
- Top-3 Topic Accuracy：94.3%。

指标说明必须明确：

- 评估不调用 LLM；
- 50 个评估 case 中当前有 35 个可索引 source case；
- 这些指标衡量本地 Retrieval 质量，不等同于端到端答案质量或 API 延迟。

本区块保留稳定标题和独立结构，以便后续加入检索性能提升过程、实验时间线、消融对比和错误案例，但本次不新增推测性实验内容或占位图表。

## 视觉与响应式设计

- 延续现有近黑背景、青蓝主状态色、酸性黄门禁状态色和等宽技术标签；
- 不新增插图，也不切换到卡片堆叠式视觉；
- Offline Plane 采用横向构建流水线；
- Verified Index Contract 使用窄而醒目的横向桥梁；
- Online Plane 使用主请求时间线，并将 best-effort 基础设施放在旁路；
- Evaluation 延续大数字指标表现；
- 桌面端保持横向信息流，移动端折叠为纵向步骤，禁止横向溢出；
- 支持 `prefers-reduced-motion`；
- Header 锚点保持浏览器原生滚动行为。

## 状态与错误表达

网站仍是项目展示站，不执行真实后端请求。本次不增加新持久状态或外部连接。

架构内容需要准确表达在线系统的错误与退化边界：

- evidence 不足：`rejected`，不调用 LLM；
- 首 token 超时、流空闲超时或生成异常：`degraded`；
- Redis/MySQL 不可用：主回答链路继续执行；
- MySQL 未生成有效 `query_id`：前端不得提交反馈。

## 验证

实施完成后验证：

1. 网站构建成功；
2. Part 1 的视觉、三组模拟查询与答案切换保持不变；
3. Header 三个导航项分别定位到 Offline Plane、Online Plane 和 Evaluation；
4. 锚点跳转后仍可继续上下滚动；
5. Offline Plane、Verified Index Contract 和 Online Plane 的文案与 `architecture.md`、`AGENTS.md` 一致；
6. Evaluation 数据及适用范围准确；
7. GitHub Pages 子路径下的样式、字体、脚本和分享图片能够加载；
8. 桌面和移动布局无重叠或横向溢出。

## 非目标

- 不在展示网站中接入真实 `/v1/chat/stream` 后端；
- 不部署 Python、FAISS、Embedding 模型、Redis 或 MySQL；
- 不修改 RAG 检索、生成、日志或反馈实现；
- 不新增未经项目实验支持的性能数据；
- 不在本次实现 Retrieval 优化过程、消融实验或错误案例可视化。
