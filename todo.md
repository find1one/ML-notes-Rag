# RAG 项目待办

## 当前状态

- [x] 已有 FastAPI `/chat`、`/chat/debug`、`/health`、`/ready` 接口
- [x] 已有 RAG 路由、查询改写、混合检索、回答生成和 JSONL 日志
- [x] 新增 `docker-compose.yml`
- [x] Docker Compose 配置 MySQL 8.4
- [x] Docker Compose 配置 Redis Stack
- [x] Python 依赖加入 `redis`
- [ ] `code/cache.py` 尚未实现
- [ ] `/query` 和 `/feedback` 尚未实现
- [ ] README 中缓存、MySQL、`/query`、`/feedback` 的描述尚未与实际代码对齐

## 第一阶段：运行基础设施

- [ ] 启动 MySQL 和 Redis

  ```bash
  docker compose up -d
  ```

- [ ] 检查容器状态

  ```bash
  docker compose ps
  ```

- [ ] 验证 Redis 连接

  ```bash
  docker compose exec redis redis-cli ping
  ```

- [ ] 将 MySQL 和 Redis 的连接参数移到 `.env`，避免在代码中硬编码
- [ ] 为两个服务增加 healthcheck

## 第二阶段：实现 Exact Redis Cache

- [ ] 在 `code/cache.py` 中创建 Redis 客户端
- [ ] 实现 `normalize_question(question) -> str`
- [ ] 使用归一化问题的 SHA-256 生成 key

  ```text
  rag:exact:{question_hash}
  ```

- [ ] 实现 `get_exact_cache(question)`
- [ ] 实现 `set_exact_cache(question, payload, ttl_seconds)`
- [ ] Redis 不可用时记录 warning，并让请求继续走 RAG
- [ ] 缓存 payload 使用 JSON 序列化，暂定字段：

  ```python
  payload = {
      "answer": answer,
      "sources": sources,
      "route_type": route_type,
      "topic": topic,
      "query_id": query_id,
  }
  ```

## 第三阶段：新增 `/query`

- [ ] 保留现有 `/chat` 接口，避免破坏已有调用
- [ ] 定义 `QueryResponse`：
  - `query_id`
  - `answer`
  - `sources`
  - `latency_ms`
  - `cached`
  - `cache_type`
  - `similarity`
- [ ] 抽取 `/chat/debug` 中可复用的 RAG 执行逻辑，避免复制整段流程
- [ ] 在 `/query` 中先接入 exact cache：

  ```text
  验证问题
  -> 查询 exact cache
  -> 命中则返回缓存结果
  -> 未命中则执行 RAG
  -> 保存查询记录
  -> 写入 exact cache
  -> 返回结果
  ```

- [ ] 每次 exact cache 命中也创建一条新的查询日志，返回本次请求的新 `query_id`
- [ ] 只缓存成功且 answer 非空的结果

## 第四阶段：验证 Exact Cache

- [ ] 第一次请求相同问题：
  - `cached=false`
  - `cache_type=null`
- [ ] 第二次请求相同问题：
  - `cached=true`
  - `cache_type="exact"`
  - `similarity=null`
- [ ] 验证问题首尾空格不影响 exact cache 命中
- [ ] 验证空问题仍返回 HTTP 400
- [ ] 验证 Redis 停止后 `/query` 仍可走完整 RAG
- [ ] 为问题归一化、缓存命中、缓存未命中和 Redis 故障降级添加测试

## 第五阶段：MySQL 查询记录与反馈

- [ ] 增加 MySQL Python 驱动和 ORM/数据库访问层
- [ ] 创建 `query_logs` 表
- [ ] `query_logs` 记录 fresh、exact、semantic 三种请求
- [ ] 创建 `feedback` 表，并关联 `query_id`
- [ ] 实现 `/feedback`
- [ ] 校验 rating，只允许 `helpful` 和 `not_helpful`
- [ ] 数据库写入失败时明确处理，避免静默丢失

## 第六阶段：Redis Semantic Cache

> Exact cache 验证通过后再开始，避免同时调试两套缓存。

- [ ] 确认现有 embedding 模型的实际向量维度
- [ ] 在 Redis Stack 中创建向量索引
- [ ] 复用现有 RAG embedding 模型生成 question embedding
- [ ] semantic cache 保存：
  - question
  - normalized_question
  - embedding
  - answer
  - sources
  - route_type
  - topic
  - query_id
  - created_at
  - not_helpful
- [ ] 实现 `ensure_semantic_cache_index()`
- [ ] 实现 `get_semantic_cache(question_embedding, route_type, topic)`
- [ ] 实现 `set_semantic_cache(question, question_embedding, payload, ttl_seconds)`
- [ ] 默认相似度阈值设为 `0.95`，并通过环境变量配置
- [ ] semantic 命中必须满足：
  - similarity >= threshold
  - route_type 一致
  - 已识别 topic 时 topic 一致
  - cached answer 非空
  - 缓存记录未被标记为 `not_helpful`
- [ ] `/feedback` 收到 `not_helpful` 后禁止后续 semantic cache 复用该答案
- [ ] v1 暂不删除 exact cache，在 README 说明此项权衡

## 第七阶段：Semantic Cache 测试

- [ ] 相似问法达到阈值时返回：
  - `cached=true`
  - `cache_type="semantic"`
  - `similarity` 为实际数值
- [ ] “线性回归是什么”不能错误命中“线性回归变量选择”
- [ ] “逻辑回归是什么”不能错误命中“线性回归是什么”
- [ ] route_type 不同时不能命中
- [ ] topic 不同时不能命中
- [ ] `not_helpful` 的缓存答案不能再次语义命中
- [ ] semantic cache entry 具有 TTL 或清理策略

## 第八阶段：前端与文档

- [ ] 创建最小 React + Vite 单页前端
- [ ] 支持输入问题、loading、error、answer、sources 和 latency
- [ ] 展示 fresh、exact、semantic 缓存状态
- [ ] 支持 helpful / not helpful 反馈
- [ ] 配置 FastAPI CORS
- [ ] 更新 README，使接口、存储方案和实际实现一致
- [ ] 补充 Docker 启动、环境变量、接口调用和测试步骤
- [ ] 记录缓存命中前后的延迟对比，作为项目展示数据

## 当前下一步

1. 运行并验证 Docker Compose 中的 MySQL 和 Redis。
2. 完成 `code/cache.py` 的 exact cache。
3. 新增 `/query`，先只接入 exact cache。
4. 连续请求同一问题，验证 miss 和 hit。
5. Exact cache 稳定后，再开始 MySQL 和 semantic cache。
