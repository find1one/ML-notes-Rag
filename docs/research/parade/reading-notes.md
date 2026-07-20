# PARADE 精读笔记：从 Passage 聚合到 Document Reranking

> 论文：Canjia Li, Andrew Yates, Sean MacAvaney, Ben He, Yingfei Sun. *PARADE: Passage Representation Aggregation for Document Reranking*
>
> 阅读版本：[arXiv:2008.09093v2](https://arxiv.org/abs/2008.09093)，[PDF](https://arxiv.org/pdf/2008.09093)
>
> 本文用途：为 `case-study.md` 提供可追溯的论文依据。它是引导式精读底稿，不代表项目复现了 PARADE。

## 1. 先用一句话理解论文

长文档放不进 BERT 时，通常要先切成多个 passage。PARADE 研究的不是“如何召回 passage”，而是候选文档已经召回之后，如何综合文档内多个 passage 的信号，得到更可靠的 document relevance score。

作者的核心判断是：

> 如果相关性分散在文档多个位置，只取最强 passage 容易丢失文档级证据；但如果一个 passage 就能充分回答 query，复杂聚合未必优于 MaxP。

这不是“复杂模型总是更好”。论文最值得迁移到当前项目的是：聚合方式应与 query 的证据分布相匹配。

## 2. 论文解决的任务是什么

### 2.1 Ad-hoc document reranking

给定一个临时输入的 query，系统需要给语料中的 document 排序。在论文实验中，BM25 或 BM25+RM3 先召回候选 document，PARADE 再重排候选。

训练时，作者使用 first-stage retrieval 返回的 Top-1000 document；相关 document 是正样本，其余候选作为负样本。因此 PARADE 是有监督 reranker，不是 first-stage retriever，也不是无需训练的排序规则。

### 2.2 为什么要切 passage

当时常见的 BERT 类模型有固定输入长度，完整长文档无法一次输入。论文将 document 用滑动窗口切成 passage：主实验窗口为 225 tokens、stride 为 200，最多保留 16 个 passage，单个 query-passage 输入最长 256 tokens。

每个 passage 与 query 共同输入预训练语言模型，得到一个 query-aware 的 `[CLS]` 表示：

```text
p_i = BERT(query, passage_i)
```

这里的 `p_i` 不是普通 passage embedding，而是 query 与该 passage 交互后的相关性表示。

## 3. 最关键的区别：聚合分数还是聚合表示

### 3.1 Score aggregation

传统做法先把每个 passage 表示变成一个标量相关分数，再聚合这些分数：

```text
passage representation → passage score
                              ↓
                   max / sum / avg / k-max
                              ↓
                       document score
```

MaxP 就是取文档内最高的 passage score。它的隐含假设是：只要找到一个足够相关的 passage，就足以判断整个 document 相关。

### 3.2 Representation aggregation

PARADE 的主要方向是先聚合多个 passage 的向量表示，再计算 document score：

```text
passage representations
          ↓
max / attention / CNN / Transformer
          ↓
document representation
          ↓
document score
```

表示聚合保留的信息比标量分数更多，因此模型有机会学习：

- 多个 passage 是否分别提供不同证据；
- passage 之间是否存在依赖；
- passage 顺序是否有用；
- 文档内相关与无关内容的相对分布。

### 3.3 一个容易混淆的名称

论文中的 `PARADE-Max` 与传统 `MaxP` 不是同一件事：

- `MaxP`：对多个 **passage relevance score** 取最大值；
- `PARADE-Max`：对多个 passage 的 **`[CLS]` 表示逐维取最大值**，之后再从聚合表示预测 document score。

当前项目 PC2 的 `max(child RRF score)` 在思想上更接近传统 MaxP，而不是 PARADE-Max。

## 4. PARADE 的几种聚合器

### 4.1 PARADE-Max

对所有 passage 表示逐维 max pooling。它不关心 passage 顺序，也不会显式建模 passage 之间的关系，但比标量 MaxP 保留更多特征维度。

### 4.2 PARADE-Sum 与 PARADE-Avg

- Sum 将所有 passage 表示直接相加；
- Avg 对相加结果按 passage 数量归一化。

Sum 允许更多 passage 累积贡献，但可能受文档长度影响；Avg 缓解长度影响，却可能稀释少量强证据。论文把它们作为无需新增聚合参数的对照方法。

### 4.3 PARADE-Attn

模型为每个 passage 学习一个权重，然后加权求和。它能区分 passage 重要性，但聚合关系仍然比较简单。

### 4.4 PARADE-CNN

用多层 CNN 逐层合并相邻 passage 表示。它以层次方式整合局部相邻证据，但其配对窗口和层次结构由网络设计预先限定。

### 4.5 PARADE-Transformer

将所有 passage 的 query-aware `[CLS]` 表示和一个额外的 `[CLS]` 输入浅层 Transformer。Self-attention 允许 passage 表示互相作用，并保留顺序和依赖关系。最终额外 `[CLS]` 的输出作为 document representation。

它的优势来自“学习 passage 之间怎样组合”，代价是需要相关性标签、训练和更多推理计算。

## 5. 实验设计应该怎样读

论文使用多个 collection，目的不是找一个全局最优模型，而是比较不同 query 和相关性分布下的聚合行为。

| Collection | 论文中的主要特征 | 阅读时关注点 |
| --- | --- | --- |
| Robust04 | 新闻语料，信息需求可能较宽 | 多 passage 表示聚合是否有效 |
| GOV2 | 政府网页，文档较长 | 相关证据分散、更多 passage 是否有帮助 |
| Genomics | 专业领域、query 较聚焦 | 单个强 passage 是否足够 |
| TREC DL 2019/2020 | 与 MS MARCO query 有关联 | 是否存在明显 MaxP bias |
| NTCIR WWW-3 | 网页排序 | 复杂聚合在另一任务上的表现 |

主实验是有监督、端到端训练的 document reranking。不能用这些结果直接证明某个无训练 RAG 聚合公式一定有效。

## 6. 关键结果

### 6.1 Robust04 与 GOV2：复杂表示聚合更有优势

论文 Table 2 中，PARADE-CNN 和 PARADE-Transformer 总体上优于 Avg、Sum、Max、Attn 等较简单的表示聚合器，也常优于 ELECTRA-MaxP 等 score aggregation baseline。

例如在 GOV2 title query 上，PARADE-Transformer 的 nDCG@20 为 0.6093，ELECTRA-MaxP 为 0.5265；在 Robust04 description query 上，两者分别为 0.6127 和 0.5540。

合适的解读是：当多个 passage 的相关信号需要组合时，能建模 passage 交互的表示聚合具有优势。

不合适的解读是：Transformer 聚合在任何长文档任务上都优于 max。

### 6.2 Genomics 与 TREC DL：简单 max 反而更好

在 Genomics 中，PARADE-Max 是论文所比较神经方法中的最好结果，优于 PARADE-Transformer。在 TREC DL 2019 和 2020 中，PARADE-Max 同样优于 PARADE-Transformer。

作者将其与 focused query 和较少的高相关 passage 联系起来：如果一个强 passage 已足以满足 query，复杂的跨 passage 聚合价值会下降，还可能引入无关信息。

### 6.3 RQ3：更多 passage 并非对所有聚合器都同样有利

在 GOV2 上，PARADE-Transformer 随保留 passage 数增加而改善，说明它能利用更多 document-level context。PARADE-Max 和 PARADE-Attn 在使用 64 个 passage 时反而略有下降，作者认为简单聚合器面对更长文档时容量有限。

这提示 RAG 系统不能只问“候选池是否更大”，还要问聚合器能否处理新增证据和噪声。

### 6.4 RQ4：相关 passage 的数量解释了数据集差异

论文 Table 9 统计每个相关 document 中高相关 passage 的数量：

| Collection | 恰好 1 个相关 passage | 1-2 个相关 passage | 3 个及以上 |
| --- | ---: | ---: | ---: |
| GOV2 | 38% | 60% | 40% |
| DL19（FIRA） | 66% | 87% | 13% |
| DL19（作者映射） | 66% | 86% | 14% |
| DL20 | 67% | 81% | 19% |
| MS MARCO train/dev | 99% / 98% | 100% / 100% | 0% / 0% |
| Genomics 2006 | 62% | 80% | 20% |

MS MARCO 的构造方式使绝大多数 query 可由单个 passage 回答。作者把这类数据特征称为 `maximum passage bias`：benchmark 天然奖励“找到一个最强 passage”的模型。

论文据此支持一个条件性结论：相关 passage 较少时，PARADE-Max 更可能表现好；相关证据分散时，复杂表示聚合的价值更大。

这仍是跨 collection 的经验分析，不是严格因果证明。passage 标注的完整度、长度和不同 collection 的定义也不完全一致，作者在原文中明确提示了这些限制。

## 7. 与当前项目怎样对应

| PARADE | 当前项目 | 对应程度 |
| --- | --- | --- |
| passage | child chunk | 局部相关证据的角色相似 |
| document | parent/source | 项目多了一层 parent 和最终 source 单位 |
| first-stage BM25 | FAISS + BM25 + RRF | 都先产生候选，但项目是混合 child 召回 |
| MaxP | PC2 的 max score | 聚合假设相似，输入分数和层级不同 |
| k-max / 多 passage score | H1 Top-2 加权 | 都奖励多条证据，但 H1 有两级固定权重 |
| representation aggregation | 项目未实现 | 只能作为未来学习式聚合方向 |
| document relevance label | expected source/topic | 标签含义和监督粒度不同 |

### 7.1 PARADE 能帮助解释什么

- 为什么单条最强证据可能足以判断相关性；
- 为什么相关证据分散时，多 passage 聚合可能有价值；
- 为什么更多 passage 同时意味着更多噪声；
- 为什么聚合方法不存在脱离数据分布的全局最优选择；
- 为什么要分析“第二次命中是否是独立证据”。

### 7.2 PARADE 不能替项目证明什么

- 不能证明 PC2 的 source 去重一定提高最终答案质量；
- 不能证明 H1 的固定 `0.5/0.3` 权重合理；
- 不能把 PARADE-Transformer 的结果迁移为当前语料上的预期收益；
- 不能替代项目自己的 120-case retrieval evaluation；
- 不能证明 source 进入 Top-3 后，正确 evidence 一定进入 prompt。

## 8. 对 PC2/H1 结果的论文式解释

PC2 采用两级 max，并强制同一 source 只占一个排名位置。它的收益主要来自排序单位与最终消费单位对齐，以及避免同源 parent 重复占位；这部分超出了 PARADE 的直接研究范围。

H1 奖励第二 child 和第二 parent。它使更多预期 source 进入 Top-3，说明多次支持确实包含有用信号；但它也降低 Source MRR@3，并使两个预期 source 漏出 Top-3。结合 PARADE，可以形成如下假设：

```text
独立多证据 → 适合奖励
overlap / 近重复 / 长文档带来的重复命中 → 不应等价奖励
```

现有实验没有标注 passage 是否提供独立证据，因此只能把它写成解释性假设，不能写成已经验证的失败原因。

## 9. 最小术语表

| 术语 | 在本文中的直观含义 |
| --- | --- |
| Information Retrieval（IR） | 根据 query 从大量内容中找到并排序相关结果 |
| Ad-hoc retrieval | 面对当前临时输入的 query 进行检索，而非固定分类任务 |
| First-stage retrieval | 快速从全语料召回候选，通常更重视覆盖和效率 |
| Reranking | 使用更精细的方法重新排列已召回候选 |
| Relevance | 一个结果满足 query 信息需求的程度 |
| Cross-encoder | 将 query 与候选文本一起输入模型，让二者充分交互后打分 |
| Passage representation | 模型为 query-passage 对生成的向量表示 |
| Score aggregation | 先给每个 passage 一个标量分数，再聚合分数 |
| Representation aggregation | 先聚合 passage 向量，再生成 document score |
| MaxP | 用 document 内最高 passage score 作为 document score |
| Pooling | 将多个值或向量压缩成一个值或向量 |
| MAP | 对每条 query 的平均准确率再取均值，考虑相关结果整体排序 |
| P@k | Top-k 中相关结果所占比例 |
| nDCG@k | 考虑分级相关性和位置折损的 Top-k 排序指标 |
| MRR@k | 关注第一个正确结果的位置；PARADE 主实验未使用，当前项目使用 Source MRR@3 |
| Pairwise loss | 训练模型让相关 document 的分数高于不相关 document |
| Statistical significance | 判断观察到的差异是否可能只是抽样波动；显著不等于实际收益一定大 |

## 10. 精读后的核心结论

1. PARADE 的贡献是系统比较 passage score aggregation 与 passage representation aggregation，而不是提出一个通用的分数相加公式。
2. 复杂聚合是否有利，取决于相关证据是否分布在多个 passage 中。
3. MaxP 在单 passage 足以回答的 benchmark 上有结构性优势，这可能来自数据构造而非模型普适性。
4. 当前项目 PC2 更接近 MaxP 思想，但其 source 去重和两级层次是独立的工程贡献。
5. H1 的负结果与论文揭示的条件性权衡一致，但 PARADE 只能提供解释框架，不能替代项目实验。
6. 对求职叙事最有价值的不是“读过 PARADE”，而是能够说明论文问题、项目问题、实验结果和迁移边界四者之间的关系。

## 参考链接

- [PARADE arXiv 页面](https://arxiv.org/abs/2008.09093)
- [PARADE PDF](https://arxiv.org/pdf/2008.09093)
- [作者公开实现](https://github.com/canjiali/PARADE)
- [项目检索实验记录](../../experiments/retrieval/results.md)
