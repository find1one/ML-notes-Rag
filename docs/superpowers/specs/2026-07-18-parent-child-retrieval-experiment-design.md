# Parent–Child Retrieval Experiment Design

> 状态（2026-07-18）：已执行历史 35-case PC0 实验，候选未通过生产替换条件。后续 120-case PC1、PC2、H1 结果见 `docs/retrieval_experiment_log.md`；本文的 35-case 数字作为历史实验记录保留。

## Goal

Evaluate a true parent–child retrieval strategy without changing the production chunking, API path, or published index. The candidate retrieves compact child chunks for precision and returns their larger parent chunks for evaluation and eventual generation context.

The experiment may inform a later production replacement, but it does not itself replace the current implementation.

## Baseline

Use the current Markdown-aware chunks as the baseline retrieval units. The published baseline has 255 chunks and the fixed offline evaluation currently reports:

- Top-1 source accuracy: 31/35 (88.6%).
- Top-3 source accuracy: 32/35 (91.4%).
- Top-3 topic accuracy: 33/35 (94.3%).

The experiment uses the same corpus revision, embedding model, FAISS/BM25/RRF implementation, evaluation queries, source labels, and Top-3 evaluation depth.

## Candidate Architecture

Add an independent `code/evaluate_parent_child.py` experiment entry point. It reuses the existing data preparation output as parent chunks and does not alter `DataPreparationModule` production behavior.

For every parent chunk:

1. Preserve the parent content and metadata in an in-memory parent store.
2. Use the parent's stable `chunk_id` as the experiment's `parent_chunk_id`.
3. Recursively split the parent content into child chunks of 200 characters with 20 characters of overlap.
4. Copy retrieval metadata such as title, topic, chapter, relative path, and section path to every child.
5. Give every child a stable child ID and a `parent_chunk_id` reference.

Only child chunks enter the experimental FAISS and BM25 indexes. Parent chunks are not embedded or added to BM25.

## Retrieval Flow

The experimental retriever wraps the existing hybrid retriever:

```text
query
  -> child FAISS + child BM25
  -> child RRF ranking
  -> map each ranked child to parent_chunk_id
  -> keep the first occurrence of each parent
  -> return unique parent chunks as Top-1 / Top-3
```

To avoid returning fewer than the requested number of parents when several children belong to the same parent, the wrapper requests an expanded child result set before parent deduplication. It starts with at least `max(top_k * 5, 20)` ranked children. If fewer than `top_k` unique parents are found, it may increase the requested child count up to the available child corpus size. Parent order is determined by the highest-ranked child belonging to each parent; the experiment does not introduce an additional parent scoring algorithm.

## Isolation and Safety

- The candidate index is built in memory only.
- The experiment does not call `save_index`, write a manifest, or use `--publish`.
- The production `POST /v1/chat/stream` and `POST /feedback` paths are unchanged.
- The current FAISS files and `code/vector_index_ml_notes/manifest.json` remain untouched.
- The experiment does not change runtime configuration defaults.
- Failures during candidate construction or evaluation leave the current service and index usable.

## Measurements

Record both baseline and candidate results from the same source revision:

- Parent count and child count.
- Average and maximum parent length.
- Average and maximum child length.
- Top-1 source accuracy.
- Top-3 source accuracy.
- Top-3 topic accuracy.
- Per-case improvements, regressions, and ranking-only changes.

Source and topic accuracy are computed on returned parent chunks, not on child chunks. This preserves comparability with the existing evaluation.

## Adoption Rule

The candidate qualifies for a separate production replacement design only when:

1. Top-1 or Top-3 source accuracy improves over the baseline.
2. The other source accuracy metric does not decline.
3. Top-3 topic accuracy does not decline.
4. Per-case changes contain no unexplained or unacceptable regression.

A complete tie does not qualify. Passing this rule does not automatically publish an index or modify the API; it only justifies designing the production replacement.

## Testing

Add focused tests that verify:

- 200-character child size and 20-character overlap are applied within each parent.
- Every child maps to exactly one existing parent.
- Parent content is not indexed as a child unless it is produced by the child splitter.
- Multiple ranked children from one parent collapse to one returned parent.
- Expanded child retrieval can still produce the requested number of unique parents.
- The experiment never writes the published manifest or FAISS index.

Run the complete unit test suite before the offline baseline/candidate comparison. Record the final comparison and adoption decision in this document.

## Experiment Result — 2026-07-18

The baseline and candidate were built in memory from the same source revision. The experiment did not write or publish an index or manifest.

| Measurement | Current baseline | Parent–Child 200/20 |
| --- | ---: | ---: |
| Parent chunks | 255 | 255 |
| Indexed child chunks | — | 1180 |
| Average parent length | 555.1 | 555.1 |
| Maximum parent length | 1375 | 1375 |
| Average child length | — | 120.0 |
| Maximum child length | — | 200 |
| Top-1 source accuracy | 31/35 (88.6%) | 30/35 (85.7%) |
| Top-3 source accuracy | 32/35 (91.4%) | 32/35 (91.4%) |
| Top-3 topic accuracy | 33/35 (94.3%) | 34/35 (97.1%) |

Twelve cases changed their returned source ranking. Ten were ranking-only changes that preserved all evaluated outcomes. Two changed an evaluated outcome:

- `clustering_overview` gained a Top-3 topic hit, but still missed its expected source in Top-1 and Top-3.
- `appendix_python` regressed from a Top-1 source hit to a Top-3-only source hit because `01-Numpy.md` ranked above the expected Python `README.md`.

The candidate fails the adoption rule because Top-1 source accuracy declined, despite the topic accuracy improvement. Do not replace the production retrieval path or publish a Parent–Child index from this experiment.

## Design Discussion for the Next Iteration

This section records conclusions from the post-experiment review. Confirmed decisions are separated from proposals that still require discussion.

### Why Smaller Children Did Not Automatically Improve Top-1 Source Accuracy

The 200-character children improved local semantic matching, but local passage relevance and expected-source ranking are different objectives. In `appendix_python`, a Numpy child densely matched "Python programming" and moved `01-Numpy.md` above the expected overview `README.md`. The README remained in Top-3, so this was a source-ordering regression rather than a complete retrieval miss.

The current experimental parent order is effectively determined by the best-ranked child:

```text
parent_score ~= max(child_score)
```

This creates several biases:

- One unusually strong or noisy child can promote its entire parent.
- Evidence spread across several moderately relevant children receives no additional credit.
- Documents with more children receive more opportunities to produce a high-scoring outlier.
- Repeated title, path, and topic metadata boosts can be counted once per child if child scores are summed naively.
- Several parent chunks from the same source file compete independently even though source-level evaluation treats them as one document.

### Confirmed Evaluation Policy

The long-term primary objective is grounded answer quality, but the project does not yet have a reliable answer-quality evaluator. The next retrieval experiment will therefore use existing labels with rank-aware metrics and no new per-query document lists:

- **Primary:** Source MRR@3, computed from the existing `expected_path_contains` label.
- **Auxiliary:** Topic MRR@3, computed from the existing `expected_topic` label as a low-cost proxy for broader relevance.
- **Guardrail:** Top-3 source accuracy must not decline.
- **Diagnostic:** Top-1 source accuracy remains visible as a strict regression signal but does not decide adoption by itself.
- **Review:** Every outcome-changing case is inspected manually.

This policy distinguishes an expected source moving from rank 1 to rank 2 from a complete Top-3 miss. Topic MRR is only a proxy: it cannot prove that a child contains enough evidence to answer the query.

### Confirmed Low-Cost Contextual Child Representation

The next candidate will use deterministic contextual retrieval text instead of embedding raw child content alone:

```text
Document: {title}
Section: {section_path}
Content: {child_page_content}
```

The contextual text is used for FAISS embeddings and BM25 indexing. The parent store keeps the original text, and only the original parent content is returned for generation. Full paths are excluded from the vector text to avoid format noise; BM25 may continue to index normalized path and topic metadata.

This is a low-cost structural variant of contextual retrieval. Anthropic's full approach generates a chunk-specific explanatory context, commonly 50–100 tokens, from the surrounding document and prepends it before both embedding and BM25 indexing. The project will first test deterministic title and section context because it is reproducible, has no LLM indexing cost, and isolates whether structural context alone fixes the observed loss. Generated context is deferred unless the deterministic version plateaus. See [Anthropic's Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval).

The prefix must remain concise so it does not dominate a 200-character child or make all children in one section nearly identical.

### Confirmed Two-Level Child, Parent, and Source Aggregation

The next experiment will rank source files first, then select parent chunks from the ranked sources for generation. This aligns the ranking unit with source-level evaluation and prevents one source from occupying several Top-k source slots.

The agreed aggregation shape is:

```text
parent_score = best_child_score + 0.5 * second_child_score
source_score = best_parent_score + 0.3 * second_parent_score
```

The fixed Top-2 terms reward corroborating evidence without allowing a long document to win merely because it has more children. The two-level structure is confirmed; the initial `0.5` and `0.3` weights remain experimental parameters rather than final constants.

After selecting the Top-3 sources, context allocation follows these rules:

1. Each Top-3 source receives its strongest parent, guaranteeing source diversity.
2. The remaining fourth context slot is offered first to the Top-1 source's second parent.
3. The second parent is selected only when its score is at least 50% of that source's best parent score and it adds a distinct section where possible.
4. If it fails the threshold, the slot goes to the highest-scoring unselected parent across the ranked sources.

The 50% threshold is an initial experiment setting and must be reported alongside retrieval metrics.

#### Why Parent-Only Aggregation Can Misalign with Source Evaluation

The retrieval hierarchy has three distinct levels:

```text
source file: Python/README.md
  -> parent P1: Libraries section
       -> children C1, C2
  -> parent P2: Data Types section
       -> children C3, C4
  -> parent P3: Functions section
       -> children C5, C6
```

Suppose the query is `Python programming appendix` and the ranked children are:

| Rank | Child | Parent | Source |
| ---: | --- | --- | --- |
| 1 | C1 | P1 | `Python/README.md` |
| 2 | C3 | P2 | `Python/README.md` |
| 3 | C2 | P1 | `Python/README.md` |
| 4 | Numpy-C1 | Numpy-P1 | `01-Numpy.md` |

Deduplicating only by `parent_chunk_id` produces P1, P2, and Numpy-P1. The resulting Top-3 source list is therefore:

```text
Python/README.md
Python/README.md
01-Numpy.md
```

P1 and P2 may contain complementary evidence, so returning both is not inherently wrong for generation. However, they consume two source-ranking positions, reduce source diversity, and compete independently even though the evaluation label treats both as the same expected source.

Parent-only ranking also prevents evidence distributed across a source from contributing to a shared score. For example:

| Parent | Best child score | Source |
| --- | ---: | --- |
| P1 | 0.55 | `Python/README.md` |
| P2 | 0.53 | `Python/README.md` |
| Numpy-P1 | 0.70 | `01-Numpy.md` |

Independent parent ranking places Numpy-P1 first because 0.70 exceeds either README parent. A bounded source-level aggregation can recognize that two different README parents support the overview query:

```text
README source score = 0.55 + 0.3 * 0.53 = 0.709
Numpy source score  = 0.70
```

This example illustrates the agreed aggregation shape; its numeric weights remain experiment settings. Summing every parent would unfairly favor long files, so source aggregation should use only the strongest one or two parents, with a discounted second contribution. The source can then be ranked once while the strongest parent chunks are retained as generation evidence.

The underlying mismatch is:

```text
retrieval ranking unit: parent chunk
evaluation label unit: source file path
```

Source-aware aggregation aligns these units, but it must preserve the ability to return multiple complementary parent chunks after source ranking when generation needs them.

### Confirmed Source-Level Metadata Boost

Contextual child representation and metadata boost serve different stages. `title + section_path + child` changes the child evidence retrieved by FAISS and BM25. Metadata boost is computed only after child evidence has been aggregated into parent and source scores.

Child RRF scores must not contain a repeatedly accumulated metadata bonus. The final source score uses a bounded multiplicative correction:

```text
title_coverage = matched query terms in source title / effective query terms
topic_match = 1 when query-derived topic matches source topic, otherwise 0
path_match = 1 when query terms match normalized source path components, otherwise 0

metadata_boost = min(
    0.15,
    0.08 * title_coverage
    + 0.05 * topic_match
    + 0.02 * path_match
)

final_source_score = source_evidence_score * (1 + metadata_boost)
```

The boost is computed once per source, is capped at 15%, and cannot promote a source with no retrieval evidence. `section_path` is excluded from this source-level correction because it is already present in the contextual child retrieval text. Evaluation labels such as `expected_path_contains` and `expected_topic` must never be used to compute the online boost. An overview- or README-specific bonus is deferred until the general metadata correction has been evaluated.

### Confirmed Bounded Adaptive Candidate Coverage

The baseline indexes 255 chunks, while Parent–Child 200/20 indexes 1180 children. Keeping `candidate_k=80` changes the approximate candidate coverage from 31% to 6.8%. A fixed candidate count therefore does not represent an equivalent search budget after increasing index granularity.

The next experiment will use a corpus-relative starting point with bounded adaptive expansion. `candidate_k` applies independently to FAISS and BM25 before RRF:

```text
initial_k = min(child_count, round_up_to_10(max(80, 0.10 * child_count)))
maximum_k = min(
    child_count,
    max(initial_k, round_up_to_10(min(300, 0.25 * child_count)))
)
```

For the current 1180-child corpus, the stages are 120, 240, and at most 300. After each stage, the system recomputes child RRF, parent aggregation, and source aggregation.

Expansion is triggered when any of the following measurable conditions holds:

- Fewer than three unique sources remain after aggregation.
- The relative score margin between source ranks 3 and 4 is below 5%.
- The Top-1 source's second parent is near its 50% allocation threshold; an initial uncertainty band of 45%–55% is used.

If no trigger applies, retrieval stops at the current stage. The first expansion doubles `candidate_k` from 120 to 240; a final expansion is capped at 300. Reaching the cap returns the best available result rather than expanding without bound.

Coverage is measured after parent/source aggregation, not only as a raw child percentage. Experiment logs must record the starting and final `candidate_k`, expansion reason, number of unique parents and sources, source rank-3/rank-4 margin, whether Top-3 changed after expansion, and incremental retrieval latency. These measurements will show whether 120 is sufficient for most queries or whether the aggregation design systematically requires a larger pool.
