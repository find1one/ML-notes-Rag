# PARADE × Parent-Child RAG 面试指南

> 目标岗位：RAG/LLM 应用算法实习
>
> 事实口径：[公开案例主稿](parade_parent_child_case_study.md)与[检索实验记录](retrieval_experiment_log.md)
>
> 使用方式：先熟悉 30 秒和 2 分钟版本，再按岗位选择追问深入程度。不要背诵自己尚不能解释的句子。

## 1. 简历 bullet

可从下面三条中选择 2-3 条，不需要全部堆在一起。

- 针对 Parent-Child RAG 中 child 排名、parent 返回与 source 评测单位不一致的问题，设计 source-aware 两级聚合；在冻结 FAISS、BM25、child-level RRF 与候选池的 120-case 消融中，将 Top-3 Source Accuracy 从 88.3% 提升至 92.5%，Source MRR@3 从 0.8333 提升至 0.8500，且无逐 case source 名次回退。
- 将检索 benchmark 从 35 条扩充并冻结为覆盖 34 个可索引 source、12 个 topic 的 120 条 case，引入 Source/Topic MRR@3、Top-k 护栏和逐 case 回退分析，降低小样本波动对检索方案决策的影响。
- 对照 PARADE 的 passage-to-document aggregation 框架，验证 contextual child、source-aware max 与两级 Top-2 加权；识别 Top-2 聚合“覆盖提升但首位稳定性下降”的负结果，拒绝围绕失败 case 调权重，并明确 retrieval 指标与最终 RAG 答案质量的边界。

如果简历空间有限，优先保留第一条和第二条。第三条更适合项目详情页或面试展开。

## 2. 30 秒版本

> 我在 Parent-Child RAG 中发现，系统用 child 做混合召回，却按 parent 返回、按 source 评测，同一 source 的多个 parent 会重复占据 Top-k。我先把评测集从 35 条扩到覆盖全部可索引 source 的 120 条，然后冻结 FAISS、BM25、RRF 和候选池，只比较聚合方式。Source-aware max 将 Top-3 Source Accuracy 从 88.3% 提升到 92.5%，Source MRR@3 从 0.8333 提升到 0.8500，而且没有 source 名次回退。Top-2 多证据加权虽然提高覆盖，却降低 MRR，所以最终没有采用。

## 3. 2 分钟版本

> 这个实验解决的是候选召回之后、LLM 上下文构建之前的排序问题。Parent-Child 结构用短 child 做细粒度检索，再返回更完整的 parent，但我的评测目标是正确 Markdown source。如果直接按照 child-level RRF 的 first-hit parent 返回，同一 source 的多个 parent 可能重复占据 Top-3，检索、排序和消费单位并不一致。
>
> 我先检查了原有 benchmark，发现只有 35 条有效 case，一条 query 的名次变化就可能明显改变 MRR，所以先扩充并冻结为 120 条，覆盖全部 34 个可索引 source 和 12 个 topic，同时加入 Source MRR@3、Topic MRR@3 和逐 case 回退分析。
>
> 实验中我冻结 raw child、FAISS、BM25、child-level RRF 和 `candidate_k=80`。PC2 只增加两级 source-aware max：先取每个 parent 的最高 child RRF score，再取每个 source 的最高 parent score，并让每个 source 只占一个结果位。它把 Top-3 Source Accuracy 从 88.3% 提升到 92.5%，Source MRR@3 从 0.8333 提升到 0.8500，Top-1 持平，而且所有预期 source 名次变化都是改善。
>
> 我还测试了 H1，用预先固定的 `0.5/0.3` 权重奖励第二 child 和第二 parent。它让 Top-3 Source 进一步升到 94.2%，但 Source MRR@3 降到 0.8361，Top-1 也下降，并有两个预期 source 掉出 Top-3，所以没有采用，也没有针对失败 case 继续调参。
>
> 后来我用 PARADE 的 passage aggregation 框架理解这个结果：如果一个强 passage 足以判断相关性，max 往往更稳定；如果独立证据分散在多个 passage，多证据聚合可能更好。但我的第二次命中也可能来自 overlap 或近重复内容，所以不能把重复命中直接等价为独立证据。PARADE 是有监督 document reranker，我的方案是 RRF 后的无训练两级聚合，因此论文只提供问题框架，最终结论仍来自自己的受控实验。

## 4. 一句话技术主线

> 我把 Parent-Child RAG 中的重复占位问题重新表述为 passage-to-parent-to-source evidence aggregation，并用扩充后的冻结 benchmark 做受控消融，在覆盖率与首位排序稳定性之间选择了 source-aware max。

## 5. 高频追问

### Q1：为什么 Parent-Child 检索会有重复占位？

简洁回答：

> 多个高排名 child 可能来自同一 source 的不同 parent。原始实现按 first-hit parent 返回，没有 source 去重，所以一个 source 能占多个 Top-k 位置。

深入回答：

> child 是匹配单位，parent 是上下文单位，source 是评测和知识来源单位。只在 child 或 parent 层去重，不能保证最终有限 Top-k 覆盖不同 source。PC2 的收益很大一部分来自把排序单位与评测、消费单位对齐，而不只是 max 公式本身。

### Q2：为什么使用 RRF，而不直接融合 FAISS 和 BM25 分数？

简洁回答：

> 两种检索器的原始分数含义和尺度不同，RRF 只依赖名次，更容易稳定融合。

深入回答：

> 本轮消融没有重新设计融合层。为了让聚合实验可归因，我冻结了 child-level RRF，只改变 RRF 之后的 parent/source 聚合。这样 PC0 到 PC2 的变化才能主要归因于排序单位和 source 去重。

### Q3：PC2 为什么叫 source-aware max？

> 它先按 `parent_chunk_id` 分组，对 child RRF score 取 max，得到 parent score；再按 source 分组，对 parent score 取 max，得到 source score。每个 source 只进入排名一次，并返回最高分 parent 作为上下文。

不要只回答“用了 max”，否则会漏掉 PC2 最重要的 source-level 去重。

### Q4：PC2 和 PARADE 是什么关系？

简洁回答：

> 它们研究的问题同属局部 passage 信号如何形成更高层 document 排名，但我的 PC2 不是 PARADE 实现。

深入回答：

> PARADE 的主要贡献是聚合 query-passage `[CLS]` representation，再通过 CNN 或 Transformer 学习 document score；它需要监督训练。PC2 聚合的是 child-level RRF score，而且还有 parent、source 两级与 source 去重。PC2 的 max 假设更接近传统 MaxP，而不是 PARADE-Max。

### Q5：为什么 H1 提高 Top-3，却降低 MRR？

> 奖励第二 child/parent 会把有多次支持的 source 抬高，所以可能让更多正确 source 进入 Top-3；但它也会重新排列前几名。如果多次命中来自 overlap、近重复或文档更长带来的更多中奖机会，就可能把原本稳定的正确 source 压低。实验中 H1 改善 12 条、回退 12 条，并有两条漏出 Top-3，所以覆盖收益不足以抵消排序稳定性损失。

注意：这是与实验一致的解释性假设。当前没有独立证据标签，不能断言 overlap 已被证明是失败原因。

### Q6：为什么 Source MRR@3 是主指标？

> Top-3 命中只区分“进没进前三”，无法区分第 1 名和第 3 名。当前生成阶段按排名顺序拼接上下文且有字符预算，靠前的 source 更可能完整进入 prompt，因此需要 rank-aware 指标。MRR@3 同时保留 Top-3 边界，并奖励正确 source 更靠前。

补充边界：MRR 仍不是答案质量指标，只是比 Hit@3 更符合当前上下文消费方式。

### Q7：120 条 case 是怎样避免数据泄漏的？

> case 覆盖全部 34 个可索引 source，每个至少 3 条；query 根据语料正文、标题和元数据生成，并且在生成时不查看 PC0、PC1 或其他候选实验的检索结果。评测集冻结后再用于后续聚合消融。15 条不可索引或缺失 source 的历史 case 单独作为数据质量诊断，不放入主分母。

如果被追问代表性：

> 它保证了 source-balanced coverage，但没有真实生产 query log，所以不能声称代表线上 query 分布。这是当前 benchmark 的明确限制。

### Q8：为什么不继续搜索 H1 的最佳权重？

> `0.5/0.3` 在实验前固定。结果出来后围绕 120 条 case 搜索权重，容易把 benchmark 变成调参集，而且当前标签只能说明正确 source，不能判断第二条命中是否是独立 evidence。继续调权重的证据基础不足，所以保留负结果并停止。

### Q9：为什么没有直接用 LLM Judge 做端到端评测？

> 当前只有 expected source/topic，没有 expected section、evidence span 和 reference claims。未经校准的 LLM Judge 会把参考答案质量、Judge 偏差和检索差异混在一起，可能产生看似完整但不可靠的分数。项目先在有可靠标签的 retrieval 层形成闭环，未来再对少量高质量 case 做 evidence-level 标注。

### Q10：这个方案已经进入线上了吗？

> 没有。PC2 通过的是离线消融门槛。实验只构建内存索引，没有调用 `save_index`，没有写 manifest，也没有替换生产索引。进入生产仍需要单独设计发布与回归验证。

这个回答很重要。不要把“建议采用 PC2”说成“已经上线 PC2”。

## 6. 可能的深入追问

### 如果重新做 H1，你会怎样改？

> 我不会先调权重，而会先建立 20-30 条 evidence-level case，标注第二条命中是否提供跨 section 的独立证据。然后测试去重 Top-2 或 diversity-aware aggregation，例如只在第二 child 与第一 child 低相似、或来自不同 section 时给予奖励。

### 为什么不直接训练 PARADE-Transformer？

> 当前语料和标签规模不支持可靠训练，学习式 cross-encoder reranker 还会显著增加离线训练与在线推理成本。现阶段最清晰的问题是 source 重复占位，PC2 用低成本、可解释的方式已经改善主要指标。学习式表示聚合应在积累 query-document relevance labels 和明确延迟预算后再评估。

### 为什么 PC1 contextual child 没有通过？

> PC1 将 title 和 section path 加入 child 的 embedding 与 BM25 文本。它使 Source MRR@3 小幅提高，但 Topic MRR@3 和 Top-3 topic 下降，逐 case 改善与回退相互抵消。合理假设是结构信息只对缺少上下文的 child 有帮助，也可能稀释正文信号，但当前实验只能证明该确定性前缀没有稳定通过门槛。

### Source-aware max 会不会被单个异常 child 主导？

> 会，这是 max 的已知风险，也是测试 H1 的原因。但 H1 的固定重复支持奖励没有稳定改善主指标。后续更合理的方向是校准 child score、增加独立证据标注，或在有足够监督数据时使用学习式 reranker，而不是根据少量失败 case直接累加更多分数。

## 7. 术语速记

| 术语 | 面试中的一句话解释 |
| --- | --- |
| First-stage retrieval | 从全语料快速召回候选 |
| Reranking | 用更精细的信号重排候选 |
| RRF | 只按名次融合多路检索结果 |
| MaxP | 用文档内最高 passage score 代表文档相关性 |
| Score aggregation | 聚合多个局部标量分数 |
| Representation aggregation | 先聚合局部向量表示，再预测整体分数 |
| MRR@3 | 第一个正确结果越靠前得分越高，只观察前三 |
| Ablation | 冻结其他条件，一次只改变一个变量 |
| Evidence coverage | 真正支持答案的文本是否进入最终上下文 |
| Citation faithfulness | 引用内容是否真实支持回答中的对应结论 |

## 8. 表述红线

不要说：

- “我实现了 PARADE”；
- “PARADE 证明了 PC2 最好”；
- “PC2 已经上线”；
- “PC2 让最终回答准确率提升了 4.2%”；
- “H1 失败就是因为 overlap”；
- “120 条 case 代表真实用户 query 分布”；
- “56 个检索 case 全部通过”。

可以说：

- “我参考 PARADE 的 passage aggregation 问题框架解释项目实验”；
- “PC2 在冻结的 120-case 离线 retrieval benchmark 上通过门槛”；
- “H1 暴露了覆盖率与首位排序稳定性的权衡”；
- “overlap 和重复证据是待进一步标注验证的解释假设”；
- “56 passed 指自动化代码测试，不是检索质量 case”。

## 9. 面试前最低准备

如果时间很少，只需要完成以下内容：

1. 能自然讲出 30 秒版本；
2. 记住 `88.3% → 92.5%` 和 `0.8333 → 0.8500`；
3. 能解释 child、parent、source 为什么是不同单位；
4. 能说明 PC2 与 PARADE 的差异；
5. 能回答“为什么不采用 H1”和“为什么没有端到端答案指标”。

完成这五项，已经足以在大多数 RAG/LLM 应用算法实习面试中可靠介绍这一阶段工作。
