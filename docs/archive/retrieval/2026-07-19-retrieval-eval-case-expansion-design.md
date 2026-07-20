# Retrieval Evaluation Case Expansion Design

> Status (2026-07-19): completed. The frozen primary benchmark now contains 120 source-evaluable cases; 15 invalid historical rows remain separate diagnostics. Results are recorded in `docs/experiments/retrieval/results.md`.

## Goal

Expand the retrieval benchmark from 35 source-evaluable cases to 120 source-evaluable cases before continuing Parent–Child ablations. The expanded benchmark must measure corpus-grounded retrieval quality without being tailored to PC0, PC1, or any later candidate's observed results.

This work changes offline evaluation data and its validation only. It does not change retrieval behavior, the API, runtime defaults, or any published index.

## Current Limitations

The repository currently defines 50 cases, but only 35 enter source metrics. Fifteen cases point to missing, empty, or too-short Markdown files. Topic coverage is uneven: several topics have no source-evaluable case, while Regression and Appendix carry much more weight. With only 35 evaluable cases, one rank-2 to rank-1 change moves Source MRR@3 by approximately 0.0143, so a single case can determine an ablation result.

The corpus currently contains 34 indexable source documents across 14 topics. The expanded benchmark therefore uses source-balanced coverage as an explicit proxy because no representative production query log is available.

## Benchmark Composition

The primary benchmark contains exactly 120 source-evaluable cases.

- Every one of the 34 indexable source documents receives at least three cases, producing a minimum of 102 source-covered cases.
- The remaining 18 cases emphasize corpus-wide overview questions, comparisons between neighboring concepts, and queries with plausible competing sources.
- Existing evaluable cases are retained only after the same validation applied to new cases.
- Existing cases whose expected source is empty, missing, or too short remain visible as data-quality diagnostics but do not count toward the 120-case primary benchmark.
- Every primary case uses an exact relative Markdown file path as its expected source. Broad directory-only expectations are normalized to an exact source or kept outside the primary benchmark when no single source is justified.

The three minimum cases per source cover distinct query intents where the document supports them:

1. A core concept or definition.
2. A method, step, assumption, API, or concrete detail.
3. A natural paraphrase or a question that distinguishes the source from a neighboring concept.

The target is diversity of information need, not three superficial rewrites of one query.

## Case Generation Rules

Cases are generated from the checked-in corpus content, titles, headings, and metadata. Generation must not inspect retrieval output from PC0, PC1, or another candidate.

Each case contains the existing fields:

- `name`: stable, unique identifier.
- `user_query`: natural user-facing Chinese query.
- `retrieval_query`: deterministic bilingual or English retrieval formulation consistent with the existing evaluator.
- `expected_topic`: topic derived from corpus metadata.
- `expected_path_contains`: exact relative path of an indexable Markdown source.

Queries must be answerable from the expected source, materially distinct from other cases for that source, and specific enough that the expected source is defensible. A case is rejected when its label relies only on a filename, when the source contains no supporting evidence, or when two sources are equally valid and the case cannot be made unambiguous.

## Validation

Automated tests verify:

- Exactly 120 primary cases are source-evaluable.
- All 34 indexable sources have at least three primary cases.
- Every primary expected path exists, is a file, and meets the corpus indexing threshold.
- Expected topics match the metadata produced for their source documents.
- Case names and normalized user/retrieval queries are unique.
- No primary case uses a directory-only expected path.
- The 15 existing data-quality cases remain excluded from source metrics unless their source content is later repaired.

The final review records case counts by source, topic, and intent type so imbalances are visible. Tests validate structure and labels; the generated cases also receive a manual content pass before the benchmark is frozen.

## Evaluation Policy

After validation, the 120-case primary benchmark is frozen before rerunning PC0 or PC1. Source MRR@3 remains the primary metric, Topic MRR@3 remains auxiliary, and Top-3 source accuracy remains the guardrail.

Data-quality diagnostics and any future challenge cases are reported separately and do not enter the primary aggregate. New cases discovered from candidate failures may be added to a challenge set, but not retroactively mixed into the frozen primary benchmark for the same experiment cycle.

## Implementation Boundary

- Expand the offline `EVAL_CASES` data used by retrieval evaluation.
- Add focused validation tests for benchmark size, source coverage, labels, and duplicates.
- Do not modify `RetrievalOptimizationModule`, API code, runtime configuration, index construction behavior, or production index files.
- Build no persistent index and publish nothing while validating the benchmark.
- Once the expanded benchmark is frozen, rerun PC0 and PC1 only. Do not proceed to PC2 without a separate instruction.

## Acceptance Criteria

The expansion is complete when:

1. The primary source-metric denominator is exactly 120.
2. All 34 current indexable sources have at least three valid cases.
3. All benchmark validation tests and the complete test suite pass.
4. PC0 and PC1 complete against the same frozen cases using in-memory indexes.
5. The report includes aggregate metrics and per-case PC0-to-PC1 changes.
6. Production retrieval code and published index files are unchanged.
