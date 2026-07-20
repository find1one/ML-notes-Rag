# 检索实验记录

最后更新：2026-07-19

## 1. 文档用途

本文记录当前已经实际完成的离线检索实验及其结果。设计方案说明“准备怎么做”，本文说明“实际做了什么、得到什么结果、是否采用”。后续如设计文档与本文的结果口径不同，以对应实验当次的原始报告和本文为准。

所有实验均只构建内存索引，没有调用 `save_index`，没有写入 manifest，也没有发布或替换生产索引。通过某项离线实验只表示该机制值得进入下一阶段，不表示已经进入线上检索链路。

## 2. 统一口径

### 2.1 名称

- **B0**：当前单层检索基线。约 1200 字符的标题感知 chunk 直接进入 FAISS + BM25，经 chunk-level RRF 后返回 chunk。
- **PC0**：原始 Parent–Child 基线。B0 chunk 作为 parent，再切成 `child_size=200`、`overlap=20` 的 child；raw child 参与检索，child-level RRF 后按 first-hit parent 返回。
- **PC1**：只把 child 的检索文本改为确定性上下文文本，其他变量与 PC0 相同。
- **PC2**：只把 PC0 的结果聚合改为 source-aware max，其他变量与 PC0 相同。
- **H1**: PC2的对照实验，只把两级max改为两级top2加权。

### 2.2 指标

- **Top-1 Source Accuracy**：第 1 个结果是否来自预期文件。
- **Top-3 Source Accuracy**：前 3 个结果是否至少有一个来自预期文件。
- **Top-3 Topic Accuracy**：前 3 个结果是否至少有一个属于预期主题。
- **Source MRR@3**：预期文件首次出现在第 1、2、3 名时分别记 `1`、`1/2`、`1/3`，未出现记 `0`，再对全部 case 取平均。本轮 Parent–Child 消融的主指标。
- **Topic MRR@3**：对预期主题采用相同计算方法，是辅助指标，不等同于答案质量。

延迟是同一台开发机上的单次离线运行结果，只用于观察明显回归，不代表 FastAPI、网络或 LLM 的端到端延迟。

### 2.3 评测集

项目先使用过 35 条 source 可评估 case。由于一条 case 的名次变化就可能显著改变总体结论，2026-07-19 将主评测集扩充并冻结为 120 条：

- 覆盖当前全部 34 个可索引 source，每个 source 至少 3 条；
- 覆盖 12 个实际进入索引的 topic 标签；
- 每条 case 使用精确 Markdown 相对路径；
- query 从语料正文、标题和元数据生成，不查看 PC0、PC1 或其他候选的检索结果；
- 原有 15 条空文件、缺失文件或过短文件 case 保留为数据质量诊断，不进入 120 条主评测分母。

除特别标注为“历史 35-case”外，本文的当前结论均以冻结的 120-case 基准为准。

## 3. 实验总览

| 日期 | 实验 | 评测集 | 关键结果 | 结论 |
| --- | --- | ---: | --- | --- |
| 2026-07-18 | chunk size 1200 vs 1000 | 历史 35-case | 三项 accuracy 完全相同 | 保留 1200，不发布 |
| 2026-07-18 | B0 vs PC0 | 历史 35-case | PC0 Top-1 source 下降，Topic Top-3 上升 | PC0 不替换 B0 |
| 2026-07-19 | 评测集扩充 | 120-case + 15 条诊断 | 34 个可索引 source 均至少 3 条 case | 冻结为当前主基准 |
| 2026-07-19 | PC0 vs PC1 | 120-case | Source MRR 小幅上升，但 Topic MRR 和 Topic Top-3 下降 | PC1 失败 |
| 2026-07-19 | PC0 vs PC2 | 120-case | Source MRR、Source Top-3、Topic MRR 均上升，无 source 名次回退 | PC2 通过离线门槛 |
| 2026-07-19 | PC2 vs H1 Top-2 加权聚合 | 120-case | Source Top-3 上升，但 Source MRR 和 Top-1 下降，2 条 case 漏出 Top-3 | H1 失败 |

## 4. 已完成实验

### 4.1 Chunk size：1200 vs 1000

目的：只改变长章节递归切分的 chunk size，判断从 1200 缩短到 1000 是否改善召回。

| 指标 | 1200 | 1000 |
| --- | ---: | ---: |
| chunks | 255 | 266 |
| 平均长度 | 555.1 | 534.4 |
| 最大长度 | 1375 | 1375 |
| Top-1 source | 31/35（88.6%） | 31/35（88.6%） |
| Top-3 source | 32/35（91.4%） | 32/35（91.4%） |
| Top-3 topic | 33/35（94.3%） | 33/35（94.3%） |

共有 5 条 case 的具体路径排序发生变化，但总体命中结果没有改善；`logistic_regression` 的 source 不可评估，topic 结果反而回退。

结论：缩短到 1000 没有提供可量化收益，保留 1200。两个候选均未发布。

### 4.2 原始 Parent–Child：B0 vs PC0

PC0 将 B0 的 255 个 chunk 作为 parent，共切出 1180 个 child。child 平均长度为 120.0，最大长度为 200。

PC0 固定使用：

```text
raw child content
→ FAISS + BM25
→ child-level RRF
→ first-hit parent
→ Top-k parents
```

它没有 source 聚合，同一个 source 的多个 parent 可以占据多个结果位。

| 指标 | B0 | PC0 |
| --- | ---: | ---: |
| Top-1 source | 31/35（88.6%） | 30/35（85.7%） |
| Top-3 source | 32/35（91.4%） | 32/35（91.4%） |
| Top-3 topic | 33/35（94.3%） | 34/35（97.1%） |

共有 12 条 case 的 source 排序变化，其中 10 条只改变具体顺序，2 条改变命中结果：

- `clustering_overview`：获得 Top-3 topic 命中，但仍未命中预期 source；
- `appendix_python`：预期 Python README 从第 1 名降到第 2 名，Numpy source 排到前面。

结论：PC0 的 topic 覆盖改善，但 Top-1 source 回退，不满足生产替换条件。

### 4.3 评测集从 35 条扩充到 120 条

扩充前只有 35 条有效 case，单条 case 从第 2 名升到第 1 名就会使 Source MRR@3 变化约 0.0143，证据过弱。扩充后：

- 主评测集正好 120 条，全部 source 可评估；
- 34 个可索引 source 每个有 3～5 条 case；
- case 名、用户问题和 retrieval query 均唯一；
- expected topic 与实际语料元数据一致；
- 15 条旧的数据质量 case 单独保存，不参与主指标。

当前已发布的 B0 索引在 120-case 基准上只读复评结果如下：

| 指标 | B0（120-case） |
| --- | ---: |
| Top-1 source | 98/120（81.7%） |
| Top-3 source | 106/120（88.3%） |
| Top-3 topic | 113/120（94.2%） |

这次复评没有修改索引。生产 manifest 仍保存发布时的旧 35-case 结果，下一次正式发布前需要用当前基准重新执行发布门禁。

### 4.4 PC1：确定性 Contextual Child

直接对照：PC0。

唯一变量是 child 的检索文本。PC0 使用原始 child；PC1 使用：

```text
Document: {title}
Section: {section_path}
Content: {child content}
```

该文本同时用于 FAISS embedding 和 BM25。parent store 仍保留纯原文，最终返回的 parent 不包含人工前缀。child size、overlap、embedding、BM25、RRF、`candidate_k=80`、child-level RRF、first-hit parent 等均保持不变。

| 指标 | PC0 | PC1 | 变化 |
| --- | ---: | ---: | ---: |
| Top-1 source | 95/120（79.2%） | 96/120（80.0%） | +1 case |
| Top-3 source | 106/120（88.3%） | 106/120（88.3%） | 0 |
| Top-3 topic | 116/120（96.7%） | 115/120（95.8%） | -1 case |
| Source MRR@3 | 0.8333 | 0.8389 | +0.0056 |
| Topic MRR@3 | 0.9250 | 0.9167 | -0.0083 |
| 平均检索延迟 | 6.2 ms | 6.1 ms | -0.1 ms |

33 条 case 的完整返回路径列表发生变化，其中 8 条改变了预期 source 的名次：

| Case | PC0 source rank | PC1 source rank | 判断 |
| --- | ---: | ---: | --- |
| `types_of_data` | 2 | 1 | 改善 |
| `cross_sectional_data` | 1 | 2 | 回退 |
| `time_series_data_type` | 2 | 3 | 回退 |
| `classification_algorithm_index` | 未进 Top-3 | 1 | 改善 |
| `feature_selection_vs_extraction` | 1 | 2 | 回退 |
| `time_series_forecasting` | 2 | 1 | 改善 |
| `python_data_types` | 3 | 2 | 改善 |
| `financial_ml_decisions` | 3 | 未进 Top-3 | 回退 |

结论：Source MRR@3 小幅提高，Top-3 source 持平，但 Topic MRR@3 和 Top-3 topic 均下降；收益和回退相互抵消，PC1 未通过门槛。历史 35-case 上 PC1 只靠 `appendix_python` 一条 case 改善，更说明小评测集不足以支持采用结论。

### 4.5 PC2：Source-Aware Max 两级聚合

直接对照：PC0。

PC2 保留 raw child、child-level RRF 和 `candidate_k=80`，只改变排序聚合单位：

```text
parent_score = max(child RRF score)
source_score = max(parent_score)
```

同一个 source 只占一个结果位；每个入选 source 返回得分最高的 parent。这里的“两级”指 child 先聚合到 parent，再由 parent 聚合到 source。

| 指标 | PC0 | PC2 | 变化 |
| --- | ---: | ---: | ---: |
| Top-1 source | 95/120（79.2%） | 95/120（79.2%） | 0 |
| Top-3 source | 106/120（88.3%） | 111/120（92.5%） | +5 cases |
| Top-3 topic | 116/120（96.7%） | 117/120（97.5%） | +1 case |
| Source MRR@3 | 0.8333 | 0.8500 | +0.0167 |
| Topic MRR@3 | 0.9250 | 0.9306 | +0.0056 |
| 平均检索延迟 | 5.7 ms | 5.4 ms | -0.3 ms |

105 条 case 的完整路径列表发生变化，主要原因是同一 source 不再重复占位。6 条 case 改变了预期 source 的名次，全部是改善：

| Case | PC0 source rank | PC2 source rank | Topic rank 变化 |
| --- | ---: | ---: | ---: |
| `classification_methods` | 未进 Top-3 | 3 | 不变 |
| `association_algorithms` | 未进 Top-3 | 3 | 不变 |
| `numpy_linear_system` | 未进 Top-3 | 2 | 未进 Top-3 → 2 |
| `time_series_chapter_topics` | 未进 Top-3 | 3 | 不变 |
| `time_series_implementation_index` | 未进 Top-3 | 3 | 不变 |
| `python_data_types` | 3 | 2 | 3 → 2 |

结论：PC2 的 Source MRR@3、Top-3 source、Topic MRR@3 和 Top-3 topic 均提高，Top-1 source 持平，且没有 source 名次回退。PC2 通过当前离线消融门槛。

诊断中也发现固定 `candidate_k=80` 有时只能聚合出少于 3 个不同 source。后续可以单独验证“保证 source 覆盖的自适应候选池”，但本轮没有改变 `candidate_k`，也没有把该想法混入 PC2。

### 4.6 H1：Top-2 Child 与 Parent 加权聚合

直接对照：PC2。

H1 保留 PC2 的 raw child、child-level RRF、`candidate_k=80` 和每个 source 只返回一个最佳 parent，只把两级 max 改为固定 Top-2 加权：

```text
parent_score = best_child_score + 0.5 × second_child_score
source_score = best_parent_score + 0.3 × second_parent_score
```

不足两个 child 或 parent 时，第二项按 `0` 计算。权重在实验前固定，没有根据结果调参。

| 指标 | PC2 | H1 | 变化 |
| --- | ---: | ---: | ---: |
| Top-1 source | 95/120（79.2%） | 91/120（75.8%） | -4 cases |
| Top-3 source | 111/120（92.5%） | 113/120（94.2%） | +2 cases |
| Top-3 topic | 117/120（97.5%） | 119/120（99.2%） | +2 cases |
| Source MRR@3 | 0.8500 | 0.8361 | -0.0139 |
| Topic MRR@3 | 0.9306 | 0.9403 | +0.0097 |
| 平均检索延迟 | 5.7 ms | 5.7 ms | 持平 |

H1 平均每条 query 有 17.43 个 parent 获得第二 child 支持，占召回 parent 的 43.5%；平均有 8.18 个 source 获得第二 parent 支持，占召回 source 的 58.8%。说明 Top-2 项在多数 query 中实际参与了排序，而不是退化为 PC2 的 max。

共有 71 条 case 的完整路径列表变化，24 条改变预期 source 名次，其中 12 条改善、12 条回退：

| Case | PC2 source rank | H1 source rank | 判断 |
| --- | ---: | ---: | --- |
| `linear_algebra_dot_outer_product` | 2 | 1 | 改善 |
| `types_of_data` | 2 | 1 | 改善 |
| `cross_sectional_data` | 1 | 2 | 回退 |
| `time_series_data_type` | 2 | 3 | 回退 |
| `regression_overview` | 1 | 未进 Top-3 | 漏出 |
| `regression_error_assumptions` | 1 | 2 | 回退 |
| `classification_methods` | 3 | 2 | 改善 |
| `classification_algorithm_index` | 未进 Top-3 | 3 | 进入 |
| `clustering_overview` | 1 | 3 | 回退 |
| `clustering_density_estimation` | 未进 Top-3 | 2 | 进入 |
| `clustering_latent_variables` | 未进 Top-3 | 3 | 进入 |
| `association_overview` | 1 | 2 | 回退 |
| `association_hidden_rules` | 1 | 2 | 回退 |
| `pca_steps` | 2 | 未进 Top-3 | 漏出 |
| `feature_selection_vs_extraction` | 1 | 2 | 回退 |
| `dimensionality_methods` | 1 | 2 | 回退 |
| `time_series_seasonality` | 2 | 1 | 改善 |
| `time_series_forecasting` | 2 | 1 | 改善 |
| `time_series_chapter` | 1 | 3 | 回退 |
| `pandas_time_series` | 未进 Top-3 | 3 | 进入 |
| `appendix_python` | 1 | 3 | 回退 |
| `python_data_types` | 2 | 1 | 改善 |
| `financial_data_sources` | 3 | 2 | 改善 |
| `financial_ml_decisions` | 3 | 1 | 改善 |

12 条改善 case 中有 4 条来自 child 数量最高四分位的 source，没有触发“超过一半改善来自长 source”的人工复核条件。因此，失败原因不是预设的长文档集中偏置门槛，而是加权重复支持改变排序后，Top-1 回退和两条 Top-3 漏出造成 Source MRR@3 明显下降。

结论：H1 虽然提高了 Top-3 source、Top-3 topic 和 Topic MRR@3，但主指标 Source MRR@3 下降，且出现无法接受的 source 漏出，因此未通过门槛。固定 `0.5/0.3` 权重不进入默认方案，也不根据本轮失败 case 继续调参。

#### 实验意义

PC2 和 H1 研究的是 RAG 链路中“候选召回之后、LLM 上下文构建之前”的检索后处理：FAISS、BM25 和 child-level RRF 已经给出 child 排名，实验只比较这些 child 应该如何聚合成 parent/source 排名。它不改变 embedding、query expansion、候选召回或 LLM 生成。

PC2 的正向结果证明，聚合单位与评测和消费单位对齐具有实际价值：同一 source 不再由多个 parent 重复占位，有限的 Top-k 可以覆盖更多不同来源。H1 则是有意义的负结果。它证明“同一 parent/source 获得多条证据支持”确实是有效排序信号，因为 Top-3 source 和 topic 覆盖均提高；但固定 `0.5/0.3` 加权会牺牲第一名稳定性，并使两条原本命中的 source 漏出 Top-3。因此，重复证据不能在当前公式下直接替代 PC2 的 max 作为默认 source 排序策略。

H1 未通过的是以 Source MRR@3 为主的排序门槛，这不等同于证明它给 LLM 的最终上下文更差。如果生成阶段会完整使用 Top-3 source，那么 Top-3 覆盖可能比第一名的小幅波动更重要；H1 从 111/120 提升到 113/120，意味着有两条额外 case 的预期 source 进入前三。但 source 命中只证明文件路径正确，不证明返回的最佳 parent 包含可回答文本，也不证明它最终完整进入 prompt。

当前生成模块按排名顺序拼接文档，basic/general 上下文上限为 2400 字符，detail 上限为 3200 字符；预算耗尽后，后排文档会被截断或不再加入。因此 Top-1 仍然比后排 source 更容易完整进入上下文。线上默认 `top_k=4` 也与本轮 Top-3 排序评测不完全相同。

由此得到的更重要结论是：检索排序优化不等于最终 RAG 效果优化。下一步若要判断 PC2 与 H1 哪个更适合生成，应保持真实上下文预算和拼接顺序，评估预期 source、parent/section 和支持答案的文本是否实际进入 prompt，再比较答案正确性、依据充分性和引用忠实度。继续围绕当前 120 条 case 微调 `0.5/0.3` 权重的意义较低，也容易过拟合。

## 5. 阶段收尾决策

本阶段在 retrieval 聚合排序层收尾，不继续构建完整的“最终上下文证据覆盖”评测。这不是因为 context-level evaluation 没有价值，而是因为当前继续投入的边际收益已经低于标签质量和复杂度风险。

当前工作已经形成完整的检索优化闭环：

```text
发现 35 条有效 case 不足以支持稳定结论
→ 扩充并冻结 120-case benchmark
→ 增加 Source/Topic MRR@3 和 Top-k 护栏
→ 控制变量验证 PC1、PC2、H1
→ 记录聚合指标和逐 case 变化
→ 采用通过门槛的 PC2，保留 PC1/H1 负结果
→ 明确实验只覆盖检索后处理，不发布生产索引
```

因此，本阶段不是未完成的端到端 RAG 评测，而是边界明确的 retrieval optimization project。PC2 给出了可解释的正向结果，H1 则暴露了“更广 Top-3 覆盖”和“更稳定第一名排序”之间的取舍。保留这一负结果比围绕当前 case 继续调整权重更有价值。

### 5.1 为什么暂不继续证据级标注

现有 120 条 case 只有 `expected_topic` 和 `expected_path`。要严谨评估最终上下文和答案，还需要新增：

- `expected_section`：能够回答问题的目标章节；
- `evidence_span`：source 中直接支持答案的文本范围；
- `reference_claims`：参考答案必须包含的原子结论；
- 引用关系标签：每条回答结论应由哪个 source/evidence 支持。

这些标签比 source path 更主观，也需要更高质量的人工复核。当前继续标注容易受到疲劳影响，标签噪声可能超过 PC2/H1 的真实差异。如果改用未经校准的 LLM Judge，还需要额外解释 Judge 偏差、参考答案质量和评分稳定性。一个证据可靠、边界清楚的 retrieval 实验，比一个标签质量可疑但看似完整的端到端分数更可信。

### 5.2 面试表述

可以将本阶段工作表述为：

> 本轮工作的范围是候选召回后的 Parent–Child 聚合与 source 排序。我先发现原有评测只有 35 条有效 case，单条变化足以左右消融结论，因此将 benchmark 扩充并冻结为覆盖全部可索引 source 的 120 条，同时加入 Source MRR@3、Topic MRR@3 和 Top-k 护栏。随后分别验证 contextual child、source-aware max 和 Top-2 多证据加权。Source-aware max 将 PC0 的 Top-3 Source Accuracy 从 88.3% 提升到 92.5%，Source MRR@3 从 0.8333 提升到 0.8500，且没有 source 名次回退；Top-2 加权虽然提高 Top-3 覆盖，却降低第一名稳定性，因此没有围绕失败 case 继续调参。
>
> 我也明确区分了检索排序与最终 RAG 答案质量。下一阶段需要标注 expected section、evidence span、reference claims，并在真实上下文预算下评估 evidence coverage 和 citation faithfulness。由于当前没有这些可靠标签，我没有用简单词面匹配或未经校准的 LLM Judge 制造一个看似完整但不可靠的端到端分数。

这种表述能够说明本阶段测量了什么、没有测量什么、为什么在可靠证据的边界上停止，以及未来如何继续，而不把离线 source 指标夸大为端到端答案质量。

### 5.3 未来恢复条件

只有在项目进入真实产品验证，或目标工作明确需要 RAG Evaluation 能力时，再恢复 context-level evaluation。建议先建立 20～30 条高质量、分层抽样、人工复核的 evidence benchmark，而不是立即为全部 120 条补标签。届时依次评估：

1. 真实 prompt 预算下预期 source 是否完整进入；
2. 返回 parent 是否属于正确 section；
3. prompt 是否包含标注的 evidence span；
4. 答案是否覆盖 reference claims；
5. 每个引用是否实际支持对应结论。

在恢复之前，只把这些内容记录为 future work，不新增正式指标，也不继续调 H1 权重。

## 6. 当前结论与边界

1. 35 条有效 case 不足以稳定判断小幅变化；当前主结果必须使用冻结的 120-case 基准。
2. 单纯缩短 parent chunk 没有收益。
3. 原始 Parent–Child PC0 不能直接替换 B0。
4. 确定性 contextual child（PC1）没有通过当前门槛，不进入默认组合。
5. source-aware max 两级聚合（PC2）在当前基准上提供了明确且无 source 回退的 Top-3 改善，值得进入单独的生产替换设计，但目前仍只是离线实验实现。
6. Top-2 child/parent 加权聚合（H1）提高覆盖但损害 Source MRR@3，不采用，也不围绕当前 case 调整权重。
7. PC3 的进入条件是 PC1 和 PC2 都通过。由于 PC1 失败，按既定设计不运行 PC3。
8. 生产索引、API、在线 RAG 默认路径均未因这些实验改变。

## 7. 验证与复现

运行当前单层索引的 120-case 只读评估：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python code/evaluate_retrieval.py --top-k 3
```

运行 PC1：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python code/evaluate_parent_child.py --experiment pc1
```

运行 PC2：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python code/evaluate_parent_child.py --experiment pc2
```

运行 H1：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python code/evaluate_parent_child.py --experiment h1
```

运行完整自动化测试：

```bash
.venv/bin/python -m pytest -q
```

H1 完成后的复验结果是 `56 passed in 1.47s`。它表示 56 个 pytest 自动化测试全部通过，用于验证代码合同和回归保护；它不是“56 个检索 case 通过”。检索质量由上面的 120 条离线 case 和 Source/Topic 指标单独衡量。

## 8. 相关文件

- 评测数据：`code/retrieval_eval_cases.py`
- 单层评测：`code/evaluate_retrieval.py`
- Parent–Child 消融：`code/evaluate_parent_child.py`
- 评测集校验：`tests/test_eval_cases.py`
- Parent–Child 实验测试：`tests/test_parent_child_experiment.py`
- 消融设计：`docs/archive/retrieval/2026-07-18-parent-child-retrieval-ablation-design.md`
- 评测集扩充设计：`docs/archive/retrieval/2026-07-19-retrieval-eval-case-expansion-design.md`
