# RAG 项目待办

## 已完成

- [x] FastAPI `/health`、`/ready`、`/chat`、`/chat/debug`、`/query`、`/feedback`
- [x] FastAPI SSE 主入口 `/v1/chat/stream`
- [x] 统一 RAGService 检索、证据门控、生成与分阶段耗时日志
- [x] Redis exact cache，故障自动降级
- [x] MySQL `query_logs` best-effort 写入、`feedback` 持久化与 Docker healthcheck
- [x] 离线索引 manifest、稳定 RRF 去重、元数据加分和候选回退
- [x] 50 条离线检索评估集及 SSE/服务层 pytest 基础覆盖

## 运行与验收

1. `python code/build_index.py --publish`，发布带 passed manifest 的离线索引。
2. `cp .env.example .env`，填入密钥和数据库密码。
3. `docker compose up -d`，确认基础设施健康。
4. `cd code && uvicorn api:app --reload`，检查 `/ready`。
5. 用 `/v1/chat/stream` 验证 `done`、`rejected`、`degraded` 三类终态。
6. 运行 `pytest` 与 `python code/evaluate_retrieval.py --top-k 3`，记录 Top-1、Top-3、本地检索 P95 和首 token 降级结果。

## 后续阶段

- [ ] React/Vite 前端、CORS 与部署鉴权。
- [ ] 将端到端延迟和缓存命中率汇总成展示报告。
