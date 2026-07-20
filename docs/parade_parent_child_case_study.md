# 从 PARADE 到 Parent-Child RAG：分层检索中的证据聚合实验

## 摘要

本项目的 Parent-Child 检索先用细粒度 child 召回，再返回较完整的 parent 作为生成上下文。离线分析发现，child-level RRF 排名与最终按 source 评测、交给 LLM 的单位并不一致，同一 source 的多个 parent 会重复占据有限的 Top-k。项目将这一问题对应到信息检索中的 passage-to-document evidence aggregation，并参考 PARADE 对 MaxP 与多 passage 表示聚合适用条件的分析。在冻结语料、child 表示、FAISS/BM25、RRF 和候选池后，source-aware max 将 Top-3 Source Accuracy 从 88.3% 提升到 92.5%，Source MRR@3 从 0.8333 提升到 0.8500，且没有 source 名次回退；Top-2 加权虽然进一步提高覆盖，却降低了首位排序稳定性，因此未采用。

## 1. 问题不是“召回更多”，而是“如何合并证据”

Parent-Child RAG 同时使用三种粒度：

- **child**：短文本，适合与 query 做精细匹配；
- **parent**：包含 child 的较完整段落，适合交给 LLM；
- **source**：原始 Markdown 文件，是当前评测和知识来源的单位。

项目的混合检索路径是：

```text
query
  → FAISS child ranking ─┐
                         ├→ child-level RRF
  → BM25 child ranking ──┘
                         → parent
                         → source
                         → LLM context
```

RRF（Reciprocal Rank Fusion）只根据名次融合 FAISS 与 BM25，避免直接比较两种检索器不同尺度的原始分数。问题发生在 RRF 之后：一条 query 可能命中同一 source 中多个 child 和 parent。如果直接按 first-hit parent 返回，同一 source 可以占据多个结果位。

这产生了三个单位的不一致：

```text
检索单位：child
排序返回单位：parent
评测与知识来源单位：source
```

即使 child 排名本身合理，有限的 Top-3 也可能被同一 source 重复占用，从而挤掉其他相关来源。此时继续扩大召回数量或修改 embedding 并没有直接回答问题；真正需要决定的是：多个局部命中应如何聚合成 parent 和 source 分数。

## 2. PARADE 提供的问题框架

[PARADE](https://arxiv.org/abs/2008.09093) 研究长文档 reranking：完整 document 无法一次放入 BERT 时，先将其切成多个 passage，再将 passage 信号聚合成 document relevance score。

论文区分两条路线：

### 2.1 Score aggregation

先给每个 passage 一个标量相关分数，再使用 max、sum、average 或 k-max 等规则聚合。

其中 MaxP 使用最高 passage score 作为 document score：

```text
document_score = max(passage_scores)
```

它隐含“一个足够相关的 passage 就能代表 document”的假设。

### 2.2 Representation aggregation

PARADE 的主要方法不是直接聚合 passage 分数，而是聚合 query-passage 的 `[CLS]` 向量，再预测 document score。CNN 或 Transformer 聚合器可以学习 passage 之间的关系、顺序和多处证据组合。

这里必须区分两个容易混淆的名称：

- 传统 **MaxP** 对 passage score 取最大值；
- **PARADE-Max** 对 passage representation 逐维 max pooling，再预测分数。

因此，本项目 PC2 的 max score 与传统 MaxP 的聚合假设相近，但不能称为实现了 PARADE-Max，更没有实现 PARADE-Transformer。

## 3. 论文最重要的条件性结论

PARADE 没有得到“复杂聚合永远优于 max”的结论。

在 Robust04 和 GOV2 上，PARADE-CNN、PARADE-Transformer 等层次化表示聚合总体更强。作者认为，这类 collection 的信息需求更宽，相关证据可能分散在文档多个 passage 中。

在 TREC DL 与 Genomics 上，PARADE-Max 反而优于 PARADE-Transformer。论文进一步统计了相关 document 中高相关 passage 的数量：MS MARCO 中 98%-99% 的 document 只有一个高相关 passage，DL 与 Genomics 中大多数 document 也只有一个或两个；GOV2 则有 40% 的 document 包含至少三个相关 passage。作者将前一种数据倾向称为 `maximum passage bias`。

由此可以得到一个比“max 或 sum 哪个更好”更有用的问题：

> 当前 query 的相关性是集中在一个局部 passage，还是需要多个独立 passage 共同支持？

这为项目的 PC2/H1 提供了分析框架，但论文结论不能代替项目自己的离线评测。PARADE 是有监督 document reranker；本项目是混合 child 召回和 RRF 之后的无训练两级聚合，任务、标签和计算预算均不同。

## 4. 从论文问题映射到项目实验

| PARADE 研究对象 | 当前项目 | 相似点 | 关键差异 |
| --- | --- | --- | --- |
| passage | child | 都承载局部相关证据 | child 先经过 FAISS、BM25 和 RRF |
| document | parent/source | 都需要从局部信号形成更高层排序 | 项目存在 parent 与 source 两级 |
| MaxP | PC2 两级 max | 都保留最强局部证据 | PC2 还完成 source 去重 |
| k-max / 多 passage score | H1 Top-2 加权 | 都有限地奖励多次支持 | H1 使用固定的两级权重 |
| representation aggregation | 未实现 | 可以学习多证据关系 | 当前没有训练标签和学习式 reranker |

项目没有照搬论文模型，而是借用论文的研究问题来组织可控实验：

```text
单条最强证据是否更稳定？
           vs
多次命中是否应获得额外奖励？
```

## 5. 先修复评测，再比较聚合策略

项目最初只有 35 条可评估 case。一条 query 从第 2 名变为第 1 名，就会使 Source MRR@3 变化约 0.0143，足以左右小幅消融结论。

因此在比较聚合方式之前，项目先将 benchmark 扩充并冻结为 120 条：

- 覆盖全部 34 个可索引 source，每个 source 至少 3 条；
- 覆盖实际进入索引的 12 个 topic；
- query 在不查看候选实验结果的条件下，从正文、标题和元数据生成；
- 15 条空文件、缺失文件或过短文件 case 作为数据质量诊断，不计入分母。

主要指标包括：

- **Top-3 Source Accuracy**：预期 source 是否进入前三；
- **Source MRR@3**：预期 source 越靠前得分越高，是本轮主指标；
- **Top-3 Topic Accuracy / Topic MRR@3**：辅助观察主题覆盖；
- **逐 case 回退**：避免平均指标掩盖关键结果漏出。

## 6. 受控消融

### 6.1 PC0：Parent-Child 基线

```text
raw child
→ FAISS + BM25
→ child-level RRF
→ first-hit parent
→ Top-k parents
```

PC0 不做 source 聚合，同一 source 的多个 parent 可以重复占位。

### 6.2 PC1：Contextual Child

PC1 只将用于检索的 child 文本改为：

```text
Document: {title}
Section: {section_path}
Content: {child content}
```

它检验结构化上下文是否改善 child 召回，不改变聚合方法。结果是 Source MRR@3 小幅提高，但 Topic MRR@3 和 Top-3 topic 回退，因此未通过门槛。

### 6.3 PC2：Source-Aware Max

PC2 保留 raw child、child-level RRF 和 `candidate_k=80`，只改变聚合与去重单位：

```text
parent_score = max(child RRF score)
source_score = max(parent_score)
```

每个 source 只占一个排名位置，并返回其中得分最高的 parent。它回答的是：将排序单位与 source 评测和消费单位对齐，是否改善有限 Top-k 的有效覆盖。

### 6.4 H1：两级 Top-2 加权

H1 直接对照 PC2，只把两级 max 改为预先固定的 Top-2 加权：

```text
parent_score = best_child + 0.5 × second_child
source_score = best_parent + 0.3 × second_parent
```

限制为 Top-2，是为了让多条证据可以贡献，同时避免把全部命中相加后天然偏向更长、child 更多的 source。权重在实验前固定，不根据 120 条 case 调参。

## 7. 结果

| 指标 | PC0 | PC1 | PC2 | H1 |
| --- | ---: | ---: | ---: | ---: |
| Top-1 Source | 79.2% | 80.0% | 79.2% | 75.8% |
| Top-3 Source | 88.3% | 88.3% | **92.5%** | 94.2% |
| Top-3 Topic | 96.7% | 95.8% | **97.5%** | 99.2% |
| Source MRR@3 | 0.8333 | 0.8389 | **0.8500** | 0.8361 |
| Topic MRR@3 | 0.9250 | 0.9167 | **0.9306** | 0.9403 |

### 7.1 为什么采用 PC2

PC2 相对 PC0：

- Top-3 Source 从 106/120 提升到 111/120；
- Source MRR@3 从 0.8333 提升到 0.8500；
- Top-1 Source 持平；
- 6 条预期 source 名次发生变化，全部改善，没有 source 回退；
- 105 条完整路径列表变化，主要来自同一 source 不再重复占位。

它说明当前系统的主要矛盾不是缺少复杂学习式聚合，而是排序单位没有与评测和上下文消费单位对齐。

### 7.2 为什么不采用 H1

H1 将 Top-3 Source 从 92.5% 提升到 94.2%，说明多次命中确实携带有用信号；但同时：

- Top-1 Source 从 79.2% 降到 75.8%；
- Source MRR@3 从 0.8500 降到 0.8361；
- 24 条 case 的预期 source 名次变化，12 条改善、12 条回退；
- 两条原本命中的预期 source 完全漏出 Top-3。

因此，本轮面对的是“更广覆盖”和“更稳定首位排序”的真实取舍，而不是单一指标的全面提升。

## 8. 怎样解释 H1 的负结果

H1 平均每条 query 有 43.5% 的召回 parent 获得第二 child 支持，58.8% 的召回 source 获得第二 parent 支持，因此第二项确实广泛参与了排序。

但“第二次命中”可能来自两种不同情况：

```text
独立证据
不同 section 分别回答 query 的不同部分
→ 应适当奖励

冗余证据
overlap、近重复 child 或长文档的更多中奖机会
→ 不应获得同等幅度奖励
```

PARADE 表明多 passage 聚合的价值取决于相关证据是否真正分散。项目当前只有 expected source/topic 标签，没有 evidence span 或“独立证据”标注，因此不能断言 overlap 就是 H1 失败的原因。严谨的结论是：固定 `0.5/0.3` 的重复支持奖励在当前 benchmark 上不能稳定替代 max。

项目还检查了长 source 偏置：12 条改善 case 中只有 4 条来自 child 数量最高四分位，没有触发预设的人工复核门槛。这排除了一个简单解释，但不等于证明 H1 不受任何长度或冗余影响。

## 9. 工程决策

最终选择 PC2，并停止围绕 H1 的失败 case 调权重，原因包括：

1. PC2 对主指标和 Top-3 覆盖均有改善，且没有逐 case source 回退；
2. H1 的覆盖收益伴随 MRR、Top-1 和关键 case 稳定性下降；
3. 根据当前 120 条 case 继续调整 `0.5/0.3` 容易对 benchmark 过拟合；
4. 当前标签无法区分独立证据与重复命中，继续优化公式缺乏可靠监督；
5. 学习式 representation aggregation 需要额外标签、训练和推理预算，不适合当前项目阶段。

截至本轮实验结束，PC2 仍是离线候选：实验只构建内存索引，没有保存 manifest，也没有发布或替换生产索引。

## 10. Retrieval 提升不等于最终答案提升

H1 未通过 Source MRR@3 门槛，不等同于证明它给 LLM 的最终上下文一定更差。反过来，PC2 提升 source 排名，也不等同于证明答案正确性提高。

当前生成模块按排名顺序拼接 parent，basic/general 上下文上限为 2400 字符，detail 上限为 3200 字符。后排内容可能被截断，因此 Top-1 与 Top-3 并不完全等价。要判断最终 RAG 效果，还需要：

- `expected_section`：目标章节；
- `evidence_span`：直接支持答案的文本；
- `reference_claims`：答案应覆盖的原子事实；
- 在真实 prompt budget 下的 evidence coverage；
- answer correctness 与 citation faithfulness。

在这些标签尚不可靠时，项目没有用词面匹配或未经校准的 LLM Judge 制造一个看似完整的端到端分数。

## 11. 对 RAG/LLM 应用算法工作的意义

这组工作展示的不是某个聚合公式本身，而是一套可复用的方法：

```text
发现检索、排序、评测单位不一致
→ 修复并冻结评测基准
→ 冻结召回器，只改变一个聚合变量
→ 同时观察覆盖、名次和逐 case 回退
→ 保留正向与负向结果
→ 在可靠证据边界上停止
```

论文阅读在这里的作用不是为项目贴标签，而是帮助提出更准确的问题、解释相反结果，并防止把局部实验夸大成普适结论。

## 12. 局限与未来方向

只有项目进入真实产品验证，或目标岗位明确要求更深入的 RAG evaluation 时，才值得继续：

1. 为 20-30 条高质量 case 标注 section、evidence span 和 reference claims；
2. 判断第二 child/parent 是否提供跨 section 的独立证据；
3. 在固定 prompt budget 下比较 PC2/H1 的 evidence coverage；
4. 验证去重后的 Top-2 或 diversity-aware aggregation；
5. 数据和计算预算允许时，再考虑学习式 passage representation aggregation。

这些是 future work，不是当前已经完成的实验。

## 参考资料

- Li et al. [PARADE: Passage Representation Aggregation for Document Reranking](https://arxiv.org/abs/2008.09093)，重点参见 Section 3、Table 2-4、Section 5.3-5.4 与 Table 9。
- [PARADE 精读笔记与术语表](parade_reading_notes.md)
- [项目检索实验记录](retrieval_experiment_log.md)
- [Parent-Child 消融设计](superpowers/specs/2026-07-18-parent-child-retrieval-ablation-design.md)
- [H1 Top-2 聚合实验设计](superpowers/specs/2026-07-19-parent-child-h1-top2-aggregation-design.md)
