# Phase 4: final held-out evaluation

Strategy A was fixed before test access. No training, tuning or policy selection used these results.

| Metric | Validation popularity | Test popularity | Change | Validation LambdaRank | Test LambdaRank | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| End-to-end NDCG@5 | 0.08201625 | 0.08192072 | -0.00009552 | 0.18354011 | 0.18082041 | -0.00271971 |
| Precision@5 | 0.06546352 | 0.06694229 | +0.00147876 | 0.15712713 | 0.15694329 | -0.00018384 |
| Recall@5 micro | 0.03020629 | 0.02963175 | -0.00057454 | 0.07250187 | 0.06947035 | -0.00303152 |
| Candidate recall micro | 0.31693824 | 0.31764918 | +0.00071093 | 0.31693824 | 0.31764918 | +0.00071093 |
| Repeat Recall@5 micro | 0.05555295 | 0.05296278 | -0.00259017 | 0.13794478 | 0.12816545 | -0.00977932 |
| New Recall@5 micro | 0.00215326 | 0.00204834 | -0.00010492 | 0.00007134 | 0.00007730 | +0.00000595 |

All values are fractions. NDCG and precision average over every household-week group, including groups without purchases. Recall and candidate recall use micro denominators of all actual target products, including retrieval misses. Undefined per-group ratios are zero. Segment metrics preserve original Top 5 positions.

Test population: 5,998,600 candidates, 29,993 groups, 338,792 actual targets, 107,617 retrieved positives.
Group sizes: {"max": 200, "mean": 200.0, "min": 200, "p50": 200.0, "p95": 200.0}.
No-purchase groups: 13,927; zero-positive groups: 15,525; complete candidate-miss groups: 1,598.

## Temporal and artifact checks

- Twelve weekly prediction cutoffs 624, 631, ..., 701; target days 625-708. History includes the cutoff; the target window excludes it and includes cutoff+7. Days 709-711 were not loaded.
- Historical-only and full-input snapshots matched exactly at every cutoff. Original candidate/feature code and schema hashes matched the frozen model run. Every label, group offset/size and historical repeat flag passed verification.
- Earlier test purchases may become history for later test cutoffs, as in weekly deployment. The model is never updated. This is rolling point-in-time evaluation, not a single forecast made on day 624 for the entire test horizon.
- Both scorers rank the exact same candidate rows; ties resolve by ascending PRODUCT_ID. Every group is evaluated. No future positives are injected.
- Frozen model: `lambdarank-v1`, 152 trees, SHA-256 `af9492f58af0b891b6f766213a5eb823bbb7fbf46aeb1719aba6c406515219ec`.
- Feature schema SHA-256: `2d319c1da73dd254e0e609c3b76f1982bef4b0a7b153dec17ac2630305adde8f`; all 25 ordered float32 features match model.feature_name().
- Test data manifest SHA-256: `6e531261aa8ab95c5fbcaac18fe2c770c15809e3fd132a4c203fc52a68f1a68b`.
- Run Git provenance: `dbbe3c072b4214d6fc96049ef6b11a81249d93a0`; dirty: `False`. Approved predecessor: `b5517b4`.
- Build/scoring/evaluation elapsed time: 254.47 s.

## Limits and 80/20 explanation

The model only orders the historical candidates it receives. Candidate coverage limits recall, especially discovery. Ranking gains measure prediction of observed purchases, not causal sales uplift. Inactive households stay in the denominator; historically unseen households are excluded and counted. Catalog brand/category metadata is assumed static, and stock availability is unavailable.

This final test measures temporal generalization after validation-based selection. Repeated household-week queries are dependent; no independence-based confidence interval is claimed. The test period is now consumed for evaluation and must not be reused as fresh model-selection evidence. Results do not authorize tuning or deployment.

Saved predictions, group metrics, schemas and checksummed manifests support audit and independent verification. Stop for approval before production integration.
