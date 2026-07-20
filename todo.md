# RAG 项目待办

## 已完成

- [x] FastAPI 仅保留 `/v1/chat/stream`、`/feedback` 两个业务接口
- [x] `/health`、`/ready` 运维探针从 OpenAPI schema 隐藏
- [x] 统一 RAGService 检索、证据门控、生成与分阶段耗时日志
- [x] Redis exact cache，故障自动降级
- [x] MySQL `query_logs` best-effort 写入、`feedback` 持久化与 Docker healthcheck
- [x] 离线索引 manifest、稳定 RRF 去重、元数据加分和候选回退
- [x] 120 条主离线检索 case、15 条数据质量诊断及 SSE/服务层 pytest 覆盖
- [x] Parent–Child PC1、PC2、H1 消融；PC2 通过离线门槛，实验未发布生产索引
- [x] Streamlit 通过 FastAPI SSE 接入 Redis cache、debug、query id 和 feedback

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
- [ ] 未排期：项目进入真实产品验证或 RAG Evaluation 岗位准备时，先标注 20～30 条高质量 evidence case，再评估 section、evidence span、reference claims 和引用忠实度。
