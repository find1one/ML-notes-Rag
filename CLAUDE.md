# ML Notes RAG 项目规则

## 边界

- 在线问答的唯一业务入口是 `POST /v1/chat/stream`；反馈入口是 `POST /feedback`。
- Streamlit 是纯 FastAPI 客户端，不得直接初始化 RAG、模型、索引或访问 Redis/MySQL。
- API 不在启动时构建索引；先运行 `python code/build_index.py --publish` 发布通过评估的索引。
- Redis 与 query log 写入采用 best-effort；没有有效 `query_id` 时不得提交反馈。
- `.env` 和密钥不得提交。配置项以 `.env.example` 和 `code/config.py` 为准。

## 常用命令

```bash
docker compose up -d
cd code && uvicorn api:app --reload
RAG_API_URL=http://127.0.0.1:8000 python -m streamlit run code/streamlit_app.py
.venv/bin/python -m pytest -q
```

## 深入文档

- `README.md`：安装、运行、接口示例与评估结果。
- `docs/README.md`：文档地图、权威顺序与归档约定。
- `docs/architecture/system-overview.md`：当前系统组件、数据流与运行边界。
- `docs/architecture/api-observability.md`：后端接入、终态、降级与日志字段。
- `docs/experiments/retrieval/results.md`：检索 benchmark、Parent–Child 消融、逐 case 结果与阶段边界。
