# Parent–Child H1 Top-2 加权聚合实验设计

> 状态（2026-07-19）：已执行。H1 的 Top-3 source/topic 覆盖提高，但 Source MRR@3 下降，且两条预期 source 漏出 Top-3，因此未通过预设门槛。完整结果见 `docs/retrieval_experiment_log.md`。

## 1. 目标

本实验判断：相较于 PC2 只采用单条最高分证据，有限地奖励同一 parent 或 source 内的第二条检索证据，能否改善 source 排序质量。

H1 只改变聚合公式。它不改变 child 表征、候选召回、RRF、返回配额或生产检索链路，也不测试 Top-1 source 返回两个 parents 的上下文分配方案。

实验只构建内存索引，不调用 `save_index`，不写 manifest，不发布或替换生产索引。

## 2. 直接对照与假设

### 2.1 直接对照：PC2

PC2 使用 raw child、child-level RRF 和两级 max 聚合：

```text
parent_score = max(child_score)
source_score = max(parent_score)
```

同一 source 只占一个 source 排名，并返回该 source 中得分最高的 parent。

### 2.2 H1 假设

PC2 可能被单个异常高分 child 主导，也可能忽略一个 parent/source 被多条中等相关证据共同支持的情况。H1 用固定 Top-2 加权聚合奖励有限的重复支持：

```text
parent_score = best_child_score + 0.5 * second_child_score
source_score = best_parent_score + 0.3 * second_parent_score
```

固定 Top-2 而不是累计全部命中，是为了限制长文档因 child 或 parent 数量更多而获得的天然票数优势。`0.5` 和 `0.3` 在实验前固定，不做网格搜索，也不根据当前 120 条 case 的结果调参。

## 3. 固定变量

PC2 与 H1 必须固定以下变量：

- 相同语料版本；
- 相同的冻结 120-case 主评测集；
- B0 当前生成的 parent chunks；
- `child_size=200`、`child_overlap=20`；
- raw child content，不使用 PC1 contextual prefix；
- 相同 embedding 模型；
- 相同 FAISS、BM25 分词方式和 RRF 常数；
- child-level RRF；
- 每路固定 `candidate_k=80`；
- Top-3 source 评测深度；
- 每个入选 source 最终只返回一个最佳 parent；
- 相同的延迟测量、评测和逐 case 报告路径。

本实验不得同时加入 metadata 聚合位置变化、自适应候选池、section diversity、第二 parent 返回配额或其他排序规则。

## 4. H1 聚合算法

### 4.1 Child 到 Parent

Child 检索器完成 FAISS + BM25 + child-level RRF 后，按 `parent_chunk_id` 分组。每个 parent 选择 RRF 分数最高的两个不同 child：

```text
child_1 = 最高分 child
child_2 = 第二高分 child；不存在时分数为 0

parent_score = child_1.score + 0.5 * child_2.score
```

第一轮不要求两个 child 来自不同 section，也不对 overlap child 去重。这样可以确保 PC2→H1 只改变 Top-2 聚合公式。报告必须保留两个 child 的 ID、原始名次和分数，以便识别重叠片段是否造成虚假重复支持。

### 4.2 Parent 到 Source

使用 parent 的 `relative_path` 作为 source ID；缺失时沿用 PC2 的 `source`、`parent_id` 回退顺序。每个 source 选择聚合分数最高的两个不同 parent：

```text
parent_1 = 最高分 parent
parent_2 = 第二高分 parent；不存在时分数为 0

source_score = parent_1.score + 0.3 * parent_2.score
```

每个 source 最终返回 `parent_1`，不会因为 H1 额外返回 `parent_2`。

### 4.3 排序稳定性

Source 首先按 `source_score` 降序排列。分数相同时，使用该 source 最佳 child 在原始 child RRF 列表中的名次升序排列；仍相同时使用 source ID 字符串升序排列，保证重复运行结果确定。

Parent 和 child 的 Top-2 选择采用同样原则：先按分数降序，再按原始 child 名次和稳定 ID 排序。不得依赖字典插入顺序决定并列结果。

## 5. 评测指标与通过规则

### 5.1 聚合指标

H1 直接对照 PC2，报告：

- Top-1 Source Accuracy；
- Top-3 Source Accuracy；
- Top-3 Topic Accuracy；
- Source MRR@3；
- Topic MRR@3；
- 平均和最大检索延迟；
- 聚合得到的 unique parents 和 unique sources 数量。

### 5.2 单项通过规则

H1 同时满足以下条件才通过离线门槛：

1. Source MRR@3 高于 PC2；
2. Top-3 Source Accuracy 不低于 PC2；
3. Topic MRR@3 不低于 PC2；
4. 没有无法解释的预期 source 完全漏出 Top-3；
5. source 名次改善发生在至少两条可评估 case；
6. 不针对单个失败 case 调整 `0.5`、`0.3` 或其他权重。

Top-1 Source Accuracy 和 Top-3 Topic Accuracy 继续记录为回归提示，但不单独决定通过。

### 5.3 长文档偏置诊断

报告每个 source 的 parent 总数和 child 总数，并按 child 总数划分 source 四分位。最高四分位固定取按 child 总数降序排列后的前 `ceil(source_count / 4)` 个 source；child 数相同时按 source ID 升序打破并列。若超过一半的 source-rank 改善 case 来自该最高四分位，则即使聚合门槛通过，最终建议也标记为 `manual_review`，不能直接建议进入生产替换设计。

该诊断只判断结果是否可能受文档长度驱动，不参与 H1 排序公式。

## 6. 逐 Case 报告

报告 PC2 与 H1 的全部返回路径，并至少区分：

- **改善**：预期 source 名次上升；
- **回退**：预期 source 名次下降；
- **进入**：预期 source 从 Top-3 外进入 Top-3；
- **漏出**：预期 source 从 Top-3 内掉出；
- **排序变化**：返回路径变化，但预期 source/topic 结果不变。

对每条 source 或 topic 名次变化 case，记录：

- PC2 与 H1 的 source rank、topic rank 和 Top-3 paths；
- H1 入选 source 的 `source_score`；
- 最佳两个 parents 的 ID、分数和原始最佳 child 名次；
- 每个 parent 最佳两个 children 的 ID、分数和原始名次；
- 第二 child 和第二 parent 对最终分数的加权贡献；
- source 的 parent/child 总数。

## 7. 测试设计

新增或扩展实验测试，至少验证：

1. 单 parent 只有一个 child 时，第二 child 分数按 `0` 计算；
2. 多个 child 按分数选择真正的两个不同 Top-2 children；
3. 单 source 只有一个 parent 时，第二 parent 分数按 `0` 计算；
4. 多个 parent 按聚合后的 `parent_score` 选择真正的两个不同 Top-2 parents；
5. source 最终返回最高分 parent，而不是第二 parent；
6. 并列分数按明确 tie-break 规则稳定排序；
7. H1 使用 raw child，且 child-level RRF 和 `candidate_k=80` 保持不变；
8. PC2 与 H1 通过同一评测和逐 case 报告路径；
9. 完整 pytest 测试套件通过；
10. 生产索引文件的大小和修改时间在实验前后保持不变。

## 8. 实现边界

预期只修改实验范围文件：

- `code/evaluate_parent_child.py`；
- `tests/test_parent_child_experiment.py`。

实验完成并确认结果后，才更新 `docs/retrieval_experiment_log.md`。不得修改：

- `code/rag_modules/retrieval_optimization.py`；
- API、RAGService、Streamlit 或配置；
- 生产索引目录及 manifest；
- 评测 case 数据；
- PC0、PC1、PC2 的既有算法语义。

CLI 应显式选择 H1，并输出 PC2→H1，而不是把 H1 与 PC0 比较。可选 JSON 报告必须包含聚合指标、逐 case 变化、Top-2 证据和长文档偏置诊断。

## 9. 完成条件

H1 实验完成需同时满足：

1. 固定公式按本文实现且有针对性测试；
2. PC2 与 H1 在同一冻结 120-case 上运行；
3. 输出聚合指标、逐 case 变化和 Top-2 聚合诊断；
4. 根据统一门槛给出 `passed`、`failed` 或因长文档偏置触发的 `manual_review` 建议；
5. 完整测试套件通过；
6. 除实验范围文件和实验记录外，不修改其他代码；
7. 不修改或发布生产索引；
8. H1 结果汇报后停止，不继续 H2 或其他实验。
