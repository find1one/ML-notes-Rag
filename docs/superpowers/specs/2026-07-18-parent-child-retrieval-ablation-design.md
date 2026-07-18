# Parent–Child 检索消融实验设计

## 1. 目标

本设计通过分阶段消融实验，判断 Parent–Child 检索中每项机制对检索质量的独立贡献，并避免一次叠加过多规则后无法归因或针对小评估集过拟合。

长期目标是提升有依据的回答质量，但项目目前没有可靠的回答质量评估器。因此，本轮以预期 source 的排序质量为主，以 topic 排序作为低成本相关性代理，并对所有结果变化进行人工复核。

本设计只覆盖离线实验。任何候选方案通过评估后，仍需单独设计生产替换方案；实验本身不修改 API、不发布索引，也不改变当前运行时默认值。

## 2. 当前事实与统一命名

### 2.1 B0：当前单层检索基线

```text
Markdown 标题感知切分
→ 约 1200 字符的检索 chunk
→ FAISS + BM25
→ chunk-level RRF
→ 返回 Top-k chunks
```

B0 没有真正的 Parent–Child 回溯。当前结果为：

| 指标 | B0 |
| --- | ---: |
| 索引 chunks | 255 |
| Top-1 source accuracy | 31/35（88.6%） |
| Top-3 source accuracy | 32/35（91.4%） |
| Top-3 topic accuracy | 33/35（94.3%） |

### 2.2 PC0：已完成的原始 Parent–Child 实验

```text
B0 的 chunks 作为 parents
→ 每个 parent 切成 child_size=200、overlap=20 的 children
→ raw child content 进入 FAISS + BM25
→ child-level RRF
→ 按 child 排名第一次出现的位置去重 parent
→ 返回 Top-k parents
```

PC0 的 parent 排序等价于：

```text
parent_score = max(child_score)
```

代码没有显式计算 parent score，而是顺序扫描已排序 children；首次看到某个 `parent_chunk_id` 时加入该 parent，后续属于同一 parent 的 children 被忽略。PC0 不做 source 聚合，同一 source 的多个 parents 可以同时占据 Top-k。

PC0 已完成结果为：

| 指标 | B0 | PC0 |
| --- | ---: | ---: |
| Parent chunks | 255 | 255 |
| Indexed children | — | 1180 |
| Average child length | — | 120.0 |
| Maximum child length | — | 200 |
| Top-1 source accuracy | 88.6% | 85.7% |
| Top-3 source accuracy | 91.4% | 91.4% |
| Top-3 topic accuracy | 94.3% | 97.1% |

PC0 提升了 topic 命中，但 `appendix_python` 中 `01-Numpy.md` 超过预期的 Python `README.md`，导致 Top-1 source 下降。PC0 不满足生产替换条件。

## 3. 固定变量

除非某项实验明确声明正在测试对应变量，否则所有实验固定：

- 相同语料版本和相同的 50 条 eval cases。
- 其中 35 条 source 可评估 case 进入 source 指标分母。
- Parent 使用 B0 当前生成的 255 个 chunks。
- Child size 为 200 字符，overlap 为 20 字符。
- 相同 embedding 模型。
- 相同 FAISS、BM25 分词方式和 RRF 常数。
- 必跑序列固定每路 `candidate_k=80`；自适应候选只在 H3 中测试。
- 离线内存构建，不调用 `save_index`，不写 manifest，不发布。
- Top-3 source 作为检索评估深度。
- 每次实验只改变明确列出的变量。

## 4. 统一评测口径

### 4.1 主指标：Source MRR@3

沿用现有 `expected_path_contains`，不新增相关文档列表：

```text
预期 source 排名 1：RR = 1
预期 source 排名 2：RR = 1/2
预期 source 排名 3：RR = 1/3
Top-3 未出现：RR = 0
Source MRR@3 = 所有可评估 cases 的 RR 平均值
```

Source MRR@3 能区分“从第 1 降到第 2”和“完全漏出 Top-3”，作为本轮主指标。

### 4.2 辅助指标：Topic MRR@3

沿用现有 `expected_topic`：

```text
预期 topic 第一次出现在排名 1：RR = 1
第一次出现在排名 2：RR = 1/2
第一次出现在排名 3：RR = 1/3
Top-3 未出现：RR = 0
```

Topic MRR@3 是片段相关性的低成本代理，不证明 child 足以回答问题。

### 4.3 护栏与诊断指标

- Top-3 source accuracy 不得下降。
- Top-1 source accuracy 继续记录，作为严格回归提示，但不单独决定采用与否。
- Top-3 topic accuracy 继续记录。
- 记录每个 outcome-changing case 的 source 排名、topic 排名和具体命中文档。
- 记录索引规模、候选数、聚合后 parent/source 数和检索延迟。

### 4.4 单项实验通过规则

候选相对其直接对照组满足以下条件，才可进入组合实验：

1. Source MRR@3 提升。
2. Top-3 source accuracy 不下降。
3. Topic MRR@3 不下降；如果下降，必须有明确且可接受的 case-level 原因，否则失败。
4. 没有无法解释的 source 完全漏召回。
5. 改进不能只来自一条 case；若只改变一条 case，将结果视为弱证据，不直接进入生产设计。

若 Source MRR@3 持平，则该机制不进入默认组合，但可以保留为诊断工具。实验不通过时不调整权重去追逐单个失败 case。

## 5. 必跑消融序列

### 5.1 PC1：只测试低成本 Contextual Child

**直接对照：** PC0。

**唯一变量：** child 的检索文本。

PC0 使用：

```text
{child_page_content}
```

PC1 使用：

```text
Document: {title}
Section: {section_path}
Content: {child_page_content}
```

Contextual text 同时用于 FAISS embedding 和 BM25 索引；parent store 仍保存纯原文，生成阶段只返回纯 parent 内容。向量文本不加入完整路径，标题不重复，prefix 保持简短。

PC1 继续使用 PC0 的 child-level RRF 和 first-hit parent 排序，不增加 source 聚合。PC0→PC1 只回答：结构化上下文是否改善 child 检索。

### 5.2 PC2：只测试 Source-Aware Max Aggregation

**直接对照：** PC0。

**唯一变量：** 从 parent 排序改为 source-aware 排序。

PC2 仍使用 raw child content 和 child-level RRF：

```text
parent_score = max(child_score)
source_score = max(parent_score)
```

同一 source 只占一个 source 排名。Source 排序后，每个 Top-3 source 暂时只返回其最佳 parent。PC0→PC2 只回答：把排序单位与 source 评测单位对齐是否改善结果。

### 5.3 PC3：组合已验证的 Contextual Child 与 Source Aggregation

**进入条件：** PC1 与 PC2 均通过各自的单项实验规则。

```text
contextual child
→ child-level RRF
→ parent_score = max(child_score)
→ source_score = max(parent_score)
```

PC3 用于检查两项机制是否有正向或负向交互。若 PC1、PC2 单独有效而 PC3 变差，必须检查 contextual prefix 是否过度同质化，以及 source max 是否放大单个 child 异常值。

### 5.4 PC4a / PC4b：隔离测试 Metadata Boost 的位置与形式

**直接对照：** PC3；若 PC3 未进入但 PC2 通过，则使用 PC2。若 PC2 未通过，PC4a/PC4b 不运行，因为尚无通过验证的 source score 路径可供 metadata 消融。

为避免同时“移除 child boost”和“增加 source boost”导致无法归因，本项分两步：

```text
PC4a：移除 child-level metadata boost，不增加新的 metadata boost
PC4b：PC4a + source-level multiplicative metadata boost
```

PC4a 回答：现有 child-level metadata boost 是否有益。PC4b 回答：metadata 只在 source 层计算一次并采用乘法校正是否有益。

```text
title_coverage = source title 命中的 query 有效词比例
topic_match = query 推导 topic 与 source topic 匹配时为 1，否则为 0
path_match = query 命中规范化 source path 组件时为 1，否则为 0

metadata_boost = min(
    0.15,
    0.08 * title_coverage
    + 0.05 * topic_match
    + 0.02 * path_match
)

final_source_score = source_evidence_score * (1 + metadata_boost)
```

Metadata boost 最多影响 15%，不能把没有检索证据的 source 抬入结果。不得使用 `expected_path_contains` 或 `expected_topic` 计算线上 boost。第一轮不添加 README 或 overview 特判。

### 5.5 PC5：比较 Early Child RRF 与 Late Source RRF

**直接对照：** 使用 PC1–PC4b 中表现最好的固定 child 表征、聚合方式和 metadata 策略，只切换融合层级。

Early child fusion：

```text
FAISS child ranking ─┐
                     ├→ child RRF → parent/source aggregation
BM25 child ranking ──┘
```

Late source fusion：

```text
FAISS children → FAISS parent/source ranking ─┐
                                              ├→ source RRF
BM25 children → BM25 parent/source ranking ──┘
```

两种 RRF 路径二选一，不叠加，避免重复计算相同检索信号。PC5 只回答：同一 child 与两路共同命中更重要，还是同一 source 被两路不同 children 支持更重要。

## 6. 可选基线实验

Source-level RRF 不依赖 Parent–Child。若需要判断当前 B0 是否也存在融合单位错位，可单独运行：

```text
B0：1200 chunks + chunk-level RRF
B1：1200 chunks 分别生成 FAISS/BM25 source ranking + source-level RRF
```

B0→B1 属于单层检索优化，不与 PC1–PC5 混为一个变量。除非 PC5 显示 source-level RRF 明显有效，否则 B1 不进入近期必跑序列。

## 7. 条件触发假设池

以下机制保留为候选，不在第一轮同时实现。只有日志或前序消融暴露对应问题时才触发。

### 7.1 H1：Top-2 Child 与 Parent 加权聚合

**触发条件：** max/first-hit 排序持续出现单个异常 child 抬高整个 source，或者多个中等相关证据无法超过单个局部强匹配。

初始候选公式：

```text
parent_score = best_child_score + 0.5 * second_child_score
source_score = best_parent_score + 0.3 * second_parent_score
```

`0.5` 和 `0.3` 是实验参数，不根据单个 case 调整。固定 Top-2 用于限制长文档的票数优势。

### 7.2 H2：Top-3 Source 的 Parent 上下文分配

**触发条件：** 检索排序已经稳定，需要比较真实生成上下文覆盖，而不仅是 source 指标。

分配规则：

1. Top-3 sources 各返回最佳 parent，先占三个位置。
2. 第四个位置优先给 Top-1 source 的第二 parent。
3. 第二 parent 分数必须达到该 source 最佳 parent 的 50%，并优先选择不同 `section_path`。
4. 不满足门槛时，第四个位置给所有已排序 sources 中最高分的未选 parent。

该规则不影响 Top-3 source 排名，只影响返回给生成阶段的四个 parents。

### 7.3 H3：有界自适应 Candidate Coverage

**触发条件：** 固定候选池导致不足三个 unique sources、Top-3 边界不稳定，或者候选扩展能稳定改变 Source MRR@3。

```text
initial_k = min(child_count, round_up_to_10(max(80, 0.10 * child_count)))
maximum_k = min(
    child_count,
    max(initial_k, round_up_to_10(min(300, 0.25 * child_count)))
)
```

当前 1180 children 对应 120→240→最多 300。`candidate_k` 分别作用于 FAISS 和 BM25。

扩展触发条件：

- 聚合后不足三个 unique sources。
- Source 第 3 与第 4 名的相对分差小于 5%。
- 在 H2 已启用时，Top-1 source 的第二 parent 比例位于 45%–55% 的不确定区间。

日志记录初始/最终 k、扩展原因、unique parents/sources、Top-3 是否变化及增量延迟。

### 7.4 H4：多样性感知的第二票

**触发条件：** 日志证明 Top-2 聚合经常把 overlap、近重复 child 或同一语义段落重复计票。

候选规则：

- 第二 child 与第一 child 的实际文本重叠不超过较短 child 的 20%。
- 规范化文本高度重复时第二票权重为 0。
- 第二 parent 来自不同 `section_path` 时使用完整第二票权重。
- 同一 `section_path` 但内容不同，第二 parent 权重减半。

该规则必须在观测到重复投票问题后才实现，不能仅因理论上合理而加入默认路径。

### 7.5 H5：生成式 Contextual Retrieval

**触发条件：** 确定性的 `title + section_path` contextual child 已验证无明显收益，且案例分析表明缺失的是代词指向、跨段背景或文档级语义。

生成式 context 会增加索引成本、不可重复性和噪声风险。对于 200 字符 child，生成 context 还可能比原始内容更长。因此它不进入近期实验，只保留为结构化 context 失效后的候选。

## 8. 推荐运行顺序与停止条件

```text
B0 / PC0（已有结果）
  ├→ PC1：contextual child
  └→ PC2：source-aware max

PC1 和 PC2 均通过
  → PC3：组合验证

PC2 通过
  → PC4a：移除 child metadata boost
  → PC4b：增加 source 乘法 metadata boost
  → PC5：child RRF vs source RRF

只有诊断触发
  → H1/H2/H3/H4/H5
```

停止条件：

- PC1 未通过：contextual child 不进入 PC3，但不阻止独立的 PC2 source 分支。
- PC2 未通过：PC3、PC4a、PC4b 和 PC5 均不运行，因为这些实验依赖通过验证的 source 排序路径。
- PC1 和 PC2 均未提升 Source MRR@3：保留 PC0 结论，不继续堆叠规则。
- 某机制提升只来自单个 case：标记为弱证据，不据此调参或替换生产实现。
- Top-3 source accuracy 下降：该实验失败，除非确认是 eval 标签错误并先修订评估集。
- 延迟显著增加但 Source MRR@3 无提升：停止该成本方向。
- 任何阶段都不得自动发布索引。

## 9. 结果记录格式

每个实验输出统一 JSON 和 Markdown 摘要：

```text
experiment_id
direct_baseline_id
single_changed_variable
parent_count
child_count
candidate_k_start
candidate_k_final
source_mrr_at_3
topic_mrr_at_3
top1_source_accuracy
top3_source_accuracy
top3_topic_accuracy
retrieval_latency_ms
outcome_changed_cases
ranking_only_changed_cases
decision
```

Case-level 变化至少记录：

```text
case name
expected source/topic
对照组 Top-3 sources
候选组 Top-3 sources
Source RR 变化
Topic RR 变化
变化原因分析
```

## 10. 防止过拟合与解释边界

- 不用 35 条有效 case 中的单条失败反推专用规则。
- 不根据 eval case 的 expected labels 参与检索或 metadata boost。
- 不在一次消融中同时修改 child 表征、聚合、融合层级和 candidate budget。
- 权重先使用预先声明值，再做少量敏感性分析，不进行大规模网格搜索。
- Topic MRR@3 只是相关性代理，不能替代 answer-quality evaluation。
- Source MRR@3 优化的是人工指定核心来源的排序，不代表具体文档一定比其他相关来源更适合回答。
- 未来建立回答质量评估后，应将检索指标降级为诊断与回归护栏。

## 11. 测试与安全要求

- 为 contextual retrieval text、source 去重、MRR 计算和 RRF 融合层级添加单元测试。
- 每种实验配置必须生成稳定、可复现的 experiment ID。
- 同一轮消融必须使用相同代码版本、语料、eval cases 和 embedding 模型。
- 实验索引只在内存或独立临时目录中构建。
- 运行前后验证 `code/vector_index_ml_notes/manifest.json` 和 FAISS 文件没有变化。
- 完整单元测试通过后才能运行离线评估。
- 实验通过只代表值得进入生产替换设计，不代表自动修改或发布生产索引。
