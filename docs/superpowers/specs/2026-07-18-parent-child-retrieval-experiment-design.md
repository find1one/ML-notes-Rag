# Parent–Child Retrieval Experiment Design

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
