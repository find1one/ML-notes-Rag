# 仓库文档整理与评测代码分阶段迁移设计

## 1. 背景

当前仓库的运行实现、离线评测、实验数据和文档都已经形成较完整内容，但目录边界仍沿用早期开发阶段的组织方式：

- `code/` 同时包含在线运行代码、索引构建命令、离线评测入口、实验实现和评测数据；
- `code/evaluate_parent_child.py` 已超过 900 行，混合 corpus 构建、聚合检索器、指标、报告与 CLI 编排；
- `docs/superpowers/specs/` 按生成工具和过程分类，混合系统设计、检索实验、展示网站与研究写作设计；
- `docs/` 没有索引页，当前架构、实验事实、研究材料和历史设计缺少统一导航；
- 根目录 `architecture.md` 被 `.gitignore` 忽略，却仍被历史设计稿引用，不能作为公开仓库的稳定事实来源。

本设计采用分阶段整理。2026-07-20 确认只实施阶段一的文档整理；任何运行代码和评测代码迁移均推迟到演示结束后。

## 2. 目标

阶段一完成后，公开文档按读者和用途分类：

- 当前系统如何工作；
- 已完成实验得到什么结果；
- 论文与项目研究材料；
- 已执行设计的历史档案。

同时保证明天演示使用的 FastAPI、Streamlit、CLI、索引发布、评测和测试入口完全不变。

## 3. 非目标

阶段一不执行以下工作：

- 不修改或移动 `code/**`；
- 不修改 `tests/**`、`pytest.ini`、`.vscode/**`；
- 不改变 Python import、模块入口或环境配置；
- 不移动 `build_index.py`、`evaluate_retrieval.py`、`evaluate_parent_child.py` 或 `retrieval_eval_cases.py`；
- 不修改 `site/`、`面试/`、`data/`、索引文件、日志或 `.env`；
- 不整理根目录 `todo.md`；
- 不执行完整的 `src/ml_notes_rag/` package 迁移。

## 4. 当前审计结论

### 4.1 已符合的项目规则

- `AGENTS.md` 是指向 `CLAUDE.md` 的软链接，没有规则分叉；
- `CLAUDE.md` 为 25 行、1154 字节，没有历史叙事膨胀；
- `.gitignore` 已忽略 `.env`、日志、缓存、生成索引、`site/` 和私人 `面试/`；
- API、反馈、Streamlit 和索引发布边界在 README、CLAUDE 与当前实现中保持一致。

### 4.2 需要整理的问题

- `docs/superpowers/specs/` 的目录名暴露内部生成流程，不能表达文档面向的读者；
- 已完成设计与当前事实并列，历史 35-case 数字容易干扰当前 120-case 口径；
- PARADE 的公开案例、精读笔记和面试指南缺少主题目录；
- API/日志说明与检索实验日志位于同一层，缺乏架构和实验边界；
- 没有 `docs/README.md` 说明权威来源和阅读顺序；
- 被忽略的根目录 `architecture.md` 混合架构说明与面试问答，不适合直接作为公开架构文档。

## 5. 阶段一目标结构

```text
docs/
├── README.md
├── architecture/
│   ├── system-overview.md
│   └── api-observability.md
├── experiments/
│   └── retrieval/
│       └── results.md
├── research/
│   └── parade/
│       ├── case-study.md
│       ├── reading-notes.md
│       └── interview-guide.md
├── archive/
│   ├── integration/
│   ├── maintenance/
│   ├── research/
│   ├── retrieval/
│   └── showcase/
└── images/
```

目录职责：

- `architecture/`：当前有效的系统机制和接口边界；
- `experiments/`：实际执行结果、指标和复现方式；
- `research/`：论文阅读、项目对照和求职解释材料；
- `archive/`：已经实施、失败、停止或被当前事实取代的设计历史；
- `images/`：公开文档使用的静态图片。

## 6. 文件迁移映射

### 6.1 当前有效文档

| 当前路径 | 目标路径 |
| --- | --- |
| `docs/backend_api_logging.md` | `docs/architecture/api-observability.md` |
| `docs/retrieval_experiment_log.md` | `docs/experiments/retrieval/results.md` |
| `docs/parade_parent_child_case_study.md` | `docs/research/parade/case-study.md` |
| `docs/parade_reading_notes.md` | `docs/research/parade/reading-notes.md` |
| `docs/parade_interview_guide.md` | `docs/research/parade/interview-guide.md` |

新增：

- `docs/README.md`：文档地图、权威来源和推荐阅读顺序；
- `docs/architecture/system-overview.md`：从当前 README、CLAUDE 和实际代码边界提炼的简洁公开架构说明。

根目录 `architecture.md` 继续作为被忽略的本地笔记保留。本轮不删除、不移动，也不整篇复制；活动文档不得继续依赖该文件。

### 6.2 历史设计档案

| 当前文件 | 目标目录 |
| --- | --- |
| `2026-07-16-streamlit-fastapi-integration-design.md` | `docs/archive/integration/` |
| `2026-07-18-chunk-size-ab-evaluation-design.md` | `docs/archive/retrieval/` |
| `2026-07-18-parent-child-retrieval-experiment-design.md` | `docs/archive/retrieval/` |
| `2026-07-18-parent-child-retrieval-ablation-design.md` | `docs/archive/retrieval/` |
| `2026-07-19-retrieval-eval-case-expansion-design.md` | `docs/archive/retrieval/` |
| `2026-07-19-parent-child-h1-top2-aggregation-design.md` | `docs/archive/retrieval/` |
| `2026-07-19-rag-showcase-information-architecture-design.md` | `docs/archive/showcase/` |
| `2026-07-20-parade-project-case-study-design.md` | `docs/archive/research/` |
| 本设计文档 | `docs/archive/maintenance/` |

迁移完成并确认 `docs/superpowers/` 为空后，删除空目录。Git 不跟踪空目录，不删除任何文档内容。

## 7. 文档权威关系

迁移后按以下顺序解释冲突：

1. `README.md`：安装、启动、接口示例和对外项目概览；
2. `docs/architecture/`：当前系统机制、边界和降级策略；
3. `docs/experiments/retrieval/results.md`：当前检索 benchmark、指标和采用结论；
4. `docs/research/parade/case-study.md`：公开研究案例；
5. `docs/archive/`：历史设计，只解释当时为什么这样设计，不作为当前事实。

`docs/README.md` 必须明确以上关系，避免读者从 archive 中读取过期数字后误判系统现状。

## 8. 链接更新范围

阶段一允许修改的活动文件仅限：

- `README.md` 中的文档链接和目录树；
- `CLAUDE.md` 的深入文档指针；
- 所有被迁移 Markdown 文件中的相对链接；
- archive 内仍需跳转到当前实验结果或当前架构的链接。

不得借链接更新顺便修改命令、指标、API 合同或运行说明。

## 9. 演示保护与验证

### 9.1 运行面冻结检查

整理前后执行：

```bash
git diff --name-only -- code tests pytest.ini .vscode
```

阶段一新增的差异必须为空。由于工作区当前已有既存改动，实际核验使用开始整理前保存的路径与 diff 快照，确认阶段一没有进一步改变这些文件。

### 9.2 文档检查

- `git diff --check` 无空白错误；
- 所有活动 Markdown 相对链接指向存在的文件；
- 活动文档中不存在 `docs/superpowers/specs` 引用；
- 活动文档中不存在依赖根目录 ignored `architecture.md` 的链接；
- README 与 CLAUDE 中所有运行命令逐字保持不变；
- archive 顶层通过 `docs/README.md` 明确标识为历史材料。

### 9.3 回归检查

运行现有 pytest。阶段一不运行新的索引构建或发布，不调用外部模型，也不执行会改变索引的命令。

## 10. 工作区安全

当前仓库存在多项未提交的检索实验、测试和文档变化。物理迁移前必须建立清晰 checkpoint，避免把已有内容修改与目录整理混入同一提交。

实施时遵守：

- 不重置、不覆盖、不丢弃已有改动；
- 只使用精确文件路径移动已确认文档；
- 如果待迁移文件同时包含未提交内容，先停止并向用户确认 checkpoint 方式；
- 文档整理使用独立提交，便于单独 revert；
- 不把用户的其他工作区修改加入文档整理提交。

## 11. 后续阶段（本轮不实施）

演示结束后可以建立顶层 `evaluation/`：

```text
evaluation/
├── cases.py
├── retrieval.py
└── parent_child/
    ├── corpus.py
    ├── retrievers.py
    ├── reporting.py
    └── cli.py
```

旧的 `code/evaluate_*.py` 暂时作为兼容入口，`code/build_index.py` 的实现后续迁入 `scripts/`。该阶段需要单独设计、完整测试和新旧入口一致性验证，不属于阶段一。

## 12. 验收标准

阶段一完成需同时满足：

1. 文档目录与第 5 节结构一致；
2. 当前事实、实验结果、研究材料和历史设计边界清楚；
3. `docs/README.md` 可以在五分钟内引导新人找到运行、架构、实验和研究材料；
4. 所有活动链接有效；
5. 运行代码、测试、配置和命令没有因整理发生变化；
6. 现有 pytest 通过；
7. 文档整理形成独立、可回滚提交。
