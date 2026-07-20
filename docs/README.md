# 项目文档导航

本目录按“当前架构、实验事实、研究材料、历史设计”组织。第一次接触项目时，先阅读根目录 [README](../README.md) 完成安装和运行，再根据任务进入下列文档。

## 当前权威文档

| 想了解什么 | 文档 | 说明 |
| --- | --- | --- |
| 安装、启动、接口示例和当前指标 | [项目 README](../README.md) | 对外总入口 |
| 系统组件、离线与在线数据流 | [系统架构概览](architecture/system-overview.md) | 当前实现边界 |
| FastAPI、SSE、日志和降级 | [API 与可观测性](architecture/api-observability.md) | 后端接入与运维参考 |
| 检索 benchmark、消融和采用结论 | [检索实验结果](experiments/retrieval/results.md) | 当前实验事实来源 |
| PARADE 与项目的公开对照 | [PARADE 案例](research/parade/case-study.md) | 面向项目展示与求职 |

若文档之间出现口径差异，按以下顺序判断当前事实：

1. 根目录 `README.md`：安装、命令、接口和对外概览；
2. `docs/architecture/`：当前系统机制与边界；
3. `docs/experiments/`：已经执行的实验结果与决策；
4. `docs/research/`：基于当前事实形成的研究与求职材料；
5. `docs/archive/`：历史设计，只解释当时为什么这样设计。

## 按读者选择

### 新开发者

1. [项目 README](../README.md)
2. [系统架构概览](architecture/system-overview.md)
3. [API 与可观测性](architecture/api-observability.md)
4. [检索实验结果](experiments/retrieval/results.md)

### RAG/LLM 应用算法面试准备

1. [PARADE 项目案例](research/parade/case-study.md)
2. [面试指南](research/parade/interview-guide.md)
3. 遇到陌生术语时查 [精读笔记](research/parade/reading-notes.md)
4. 需要核实数字时回到 [检索实验结果](experiments/retrieval/results.md)

### 追溯设计过程

`archive/` 保留已经实施、失败、停止或被后续事实取代的设计：

- `archive/integration/`：Streamlit 与 FastAPI 集成设计；
- `archive/retrieval/`：chunk、Parent-Child、评测集与聚合消融设计；
- `archive/research/`：研究材料的写作设计；
- `archive/showcase/`：独立展示网站的信息架构设计；
- `archive/maintenance/`：仓库与文档维护设计。

历史设计中的指标是当时快照。当前检索数字和采用结论始终以 [检索实验结果](experiments/retrieval/results.md) 为准。

## 目录约定

```text
docs/
├── architecture/     # 当前系统如何工作
├── experiments/      # 实际执行结果与复现方式
├── research/         # 论文阅读、项目对照与求职材料
├── archive/          # 已完成或过期设计历史
└── images/           # 文档静态图片
```

新增文档应优先进入上述主题目录，不再按生成工具或 agent 名称创建目录。
