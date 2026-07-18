# Chunk Size 1200/1000 A/B Evaluation Design

## Goal

Evaluate whether reducing the recursive character chunk size from 1200 to 1000 improves retrieval quality without changing the current Markdown-aware splitting strategy or other retrieval parameters.

The values in this experiment are character counts, not tokenizer token counts.

## Baseline

The currently published index uses:

- Markdown heading splitting before length-based splitting.
- Recursive character chunk size: 1200 characters.
- Chunk overlap: 150 characters.
- Length-based splitting only for heading sections longer than 1400 characters.
- 255 indexed chunks.
- 50 evaluation cases, of which 35 have an evaluable source.
- Top-1 source accuracy: 31/35 (88.6%).
- Top-3 source accuracy: 32/35 (91.4%).
- Top-3 topic accuracy: 33/35 (94.3%).

## Experiment

Run two isolated variants against the same corpus and fixed evaluation set:

| Variant | Chunk size | Chunk overlap | Other settings |
| --- | ---: | ---: | --- |
| Baseline | 1200 characters | 150 characters | unchanged |
| Candidate | 1000 characters | 150 characters | unchanged |

The Markdown heading splitter, 1400-character secondary-split threshold, separator order, embedding model, FAISS/BM25/RRF retrieval, candidate count, evaluation queries, and Top-3 evaluation depth remain unchanged. This makes chunk size the only experimental variable.

## Configuration and Execution

Expose the recursive character chunk size as an explicit configuration value with a default of 1200, preserving current runtime behavior. The offline index build/evaluation command accepts an override for the candidate run.

Each variant must be evaluated without publishing first. Evaluation must not overwrite the currently published, verified index. Only the winning candidate may be rebuilt and published through the existing `python code/build_index.py --publish` workflow after it passes the comparison and existing publication thresholds.

## Measurements

Record the following for both variants:

- Total chunk count.
- Average and maximum chunk character length.
- Top-1 source accuracy.
- Top-3 source accuracy.
- Top-3 topic accuracy.
- The set of per-case Top-1 and Top-3 regressions and improvements.

Aggregate accuracy alone is insufficient: the comparison must identify which fixed evaluation cases changed so a gain in one area cannot silently hide a regression in another.

## Decision Rule

Adopt the 1000-character candidate only when all of the following hold:

1. Top-1 source accuracy is at least the current 88.6% baseline.
2. Top-3 source accuracy is at least the current 91.4% baseline.
3. Top-3 topic accuracy is at least the current 94.3% baseline.
4. Any per-case regression is reviewed and judged acceptable in light of an equal or greater improvement elsewhere.
5. The candidate continues to pass the existing publication thresholds: Top-1 at least 60% and Top-3 at least 85%.

If these conditions are not met, retain the published 1200-character index.

## Safety and Compatibility

- The API continues to load only an index whose manifest evaluation status is `passed`.
- The experiment does not change `POST /v1/chat/stream`, `POST /feedback`, Streamlit boundaries, Redis behavior, query logging, or feedback validation.
- The default chunk size remains 1200 until the 1000-character candidate wins the A/B comparison.
- Index publication remains an explicit offline action; API startup never builds an index.

## Testing

Add focused tests for configuration propagation and chunk-size override behavior. Run the existing unit test suite, then run both offline retrieval evaluations from the same source revision. Capture the A/B metrics and changed cases in the implementation report before deciding whether to publish.
