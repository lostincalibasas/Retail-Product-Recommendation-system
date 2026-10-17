# Phase 2: data and feature integration checkpoint

Generated from the completed local artifact manifest and verified Parquet files. No model was trained.

- Build code commit: `7de684455276b4285fe8eba77498fcabeb8d4892`; dirty at build: `False`.
- Python: 3.12.15; data build time (excluding initial source load): 278.0 seconds.
- Artifacts: `artifacts/data/retail-ranking-v1`; 111 Parquet files verified by checksum and row count.
- Manifest SHA-256: `3d09e8064531ddea7b173dc278468e781d4b52f9a5a5ce18e5581c74f69ba925`.
- Parquet storage: 297,534,073 bytes. All ranking-group offsets independently verified from disk.
- 25 ordered float32 features; int64 identifiers; int8 binary labels; maximum 200 candidates per household.

## Measured development data

| Metric | Train | Validation |
| --- | ---: | ---: |
| Weekly snapshots | 25 | 12 |
| Candidate rows | 12,488,000 | 5,995,400 |
| Household-week groups | 62,440 | 29,977 |
| Positive candidate labels | 221,909 | 102,952 |
| Trainable groups (both label values) | 30,299 | 14,565 |
| Zero-positive candidate groups | 32,141 | 15,412 |
| No-purchase groups | 28,831 | 13,780 |
| Purchase groups with no candidate hit | 3,310 | 1,632 |
| Groups with target purchases | 33,609 | 16,197 |
| All actual household/product targets | 679,462 | 324,833 |
| Targets missing from candidates | 457,553 | 221,881 |
| Excluded new-household window occurrences | 1 | 1 |
| Micro candidate recall | 32.6595% | 31.6938% |
| Macro candidate recall | 37.1512% | 36.0608% |
| Candidate-miss rate among purchase groups | 9.8486% | 10.0759% |
| Group size min / median / p95 / max | 200 / 200 / 200 / 200 | 200 / 200 / 200 / 200 |

## Temporal and source checks

- Train cutoffs: 365–533, weekly; target days 366–540. Validation cutoffs: 540–617; target days 541–624.
- Transaction loader filter: `DAY <= 624`; 2,234,076 rows loaded.
- Excluded nonpositive-quantity source rows: 12,405.
- Positive-quantity rows with negative sales retained/audited: 0.
- First/last snapshot of each split passed exact feature/candidate equality with all future transactions removed, and complete Parquet roundtrip equality.
- All snapshot labels, candidate uniqueness, finite feature dtypes, historically eligible household coverage and group sizes validated during build.
- Test target days 625–708 remain sealed: no test snapshots, labels, metrics or model-selection use.

## Interpretation

Candidate recall measures retrieval before ranking; a ranker cannot recover missing candidates. Zero-positive groups include both no-purchase weeks and complete candidate misses. The miss rate denominator includes only groups with actual purchases. All eligible households, including inactive ones, remain in evaluation.

The fixed 200-product budget and recency/category/popularity rules are an initial retrieval design, not a tuned optimum. Historical sales do not establish current stock. Static catalog metadata is an explicit assumption; demographics and promotions are unused.

No NDCG@5 or Recall@5 ranking result exists yet. At the next approved checkpoint, baseline and LambdaRank must use identical candidates; end-to-end ideal DCG must count actual targets missed by retrieval. Consider candidate coverage before spending time tuning ranking.

Environment smoke results are in `environment_smoke.json`. Contracts and agent ownership are in `docs/phase2_contract.md`. Reproduce this report with `python scripts/report_data_checkpoint.py` from the Phase 2 environment.
