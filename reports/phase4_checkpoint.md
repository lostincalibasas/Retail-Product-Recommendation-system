# Phase 4 final held-out checkpoint

Strategy A was approved before test access: native LightGBM model
`lambdarank-v1`, 152 trees, original 25 ordered features, original candidate
generation, unmodified Top 5 with ascending product-ID tie breaks. No fitting,
tuning, new policy, SHAP or serving work occurred.

## Actual final results

| Test metric | Popularity baseline | Frozen LambdaRank |
| --- | ---: | ---: |
| End-to-end NDCG@5, all groups | 0.081921 | 0.180820 |
| Precision@5, all groups | 6.6942% | 15.6943% |
| Recall@5, micro | 2.9632% | 6.9470% |
| Candidate recall, micro | 31.7649% | 31.7649% |
| Repeat-product Recall@5, micro | 5.2963% | 12.8165% |
| New-product Recall@5, micro | 0.2048% | 0.0077% |

LambdaRank has 120.73% higher test NDCG than popularity. The result evaluates the
already-approved policy; it does not select a new policy from test performance.

There are **5,998,600 candidates**, **29,993 household-week groups**, exactly
**200 candidates per group**, **338,792 actual targets** and **107,617 positive
candidate labels** (5,890,983 negative labels). Both scorers see identical rows.
All **13,927 no-purchase groups** and **15,525 zero-positive groups** remain in
evaluation. The 1,598 complete candidate-miss groups represent **9.9465%** of
the 16,066 groups with actual purchases.

The target set contains 183,544 repeat products and 155,248 new-to-household
products. Candidate recall is 54.5989% for repeats and 4.7691% for new products.
Popularity finds 9,721 repeat and 318 new purchases; LambdaRank finds 23,524 repeat
and 12 new purchases. The discovery weakness established in validation persists.

## Changes from validation

The [generated comparison](phase4_test.md) reports every metric for both models
on validation and test. The main LambdaRank changes are:

- NDCG@5: 0.183540 -> 0.180820, **-0.002720 / -1.48% relative**.
- Precision@5: 15.7127% -> 15.6943%, **-0.0184 percentage points**.
- Recall@5: 7.2502% -> 6.9470%, **-0.3032 percentage points / -4.18% relative**.
- Repeat recall: 13.7945% -> 12.8165%, **-0.9779 percentage points / -7.09% relative**.
- New recall: 0.0071% -> 0.0077%, with **11 -> 12 hits**; that small count limits
  interpretation of its 8.34% relative increase.
- Overall candidate recall: 31.6938% -> 31.7649%, **+0.0711 percentage points**.

Observed counts explain much of the recall change mathematically: total targets
increase from 324,833 to 338,792 (**+4.30%**) while model hits decrease only from
23,551 to 23,536. Repeat targets increase **7.56%**, while repeat hits change from
23,540 to 23,524. Repeat candidate coverage drops **1.4984 percentage points**;
new candidate coverage increases **0.0845 percentage points**. Overall coverage
alone hides those segment differences.

All-group NDCG also depends on purchase incidence. Purchase-group-only LambdaRank
NDCG moves from 0.339691 to 0.337567 (**-0.63% relative**), versus -1.48% for the
all-group metric. These are measured distribution/performance changes, not causal
claims about seasonality or proof of statistical significance. Popularity NDCG
is almost unchanged (-0.12% relative), while its recall declines 1.90% relative.

## Point-in-time protocol

The new final-evaluation runner calls the unchanged `build_snapshot`,
`build_targets`, `label_snapshot` and `snapshot_groups` functions. The development
runner and training loader still exclude test access. New final artifacts are
separate from the immutable development data and model outputs.

- Cutoffs: 624, 631, 638, 645, 652, 659, 666, 673, 680, 687, 694, 701.
- History: positive-quantity transactions with `DAY <= cutoff`.
- Labels: distinct positive-quantity purchases with `cutoff < DAY <= cutoff+7`;
  the disjoint target windows cover days 625-708 exactly.
- Source reading stops at day 708; days 709-711 are not loaded.
- Test snapshots built from historical-only input match snapshots built with the
  full permitted source input **exactly at all 12 cutoffs**.
- Candidate products must have a prior global positive purchase; every eligible
  historical household receives candidates. Future positives are not injected.
- Novelty uses historical household-product purchase membership, not source flags.
  Target repeat/new flags include all actual targets, even candidate misses.
- Earlier test-period purchases become history at later cutoffs. This matches
  weekly deployment with a frozen model and is not a single 84-day forecast made
  on day 624. One new-household/window occurrence is excluded before that household
  has history; it becomes eligible at a subsequent cutoff.

## Executed checks and provenance

The initial `main` HEAD was **b5517b4**, with a clean working tree. Before test
records were opened, 21 hashes verified the model run, its original implementation,
development manifest and source files. The tested evaluation procedure was then
committed as **dbbe3c072b4214d6fc96049ef6b11a81249d93a0**. The actual test run
records that commit and a **clean working tree at run start**. A separate final
checkpoint records the results and independent verifier.

- Automated suite: **58 passed in 3.79 seconds**, including new exact-boundary,
  historical-novelty, future-perturbation and rolling-history tests. Existing
  synthetic training tests do not retrain the saved model.
- All 36 test Parquet shards passed exact disk round-trip checks. Feature order,
  float32 dtypes, identifier order, every target label, group offsets/sizes and
  positive counts were checked before evaluation.
- `scripts/verify_heldout.py` imports neither the production evaluator nor its
  data/metric helpers. It independently reconstructs all target windows and
  repeat/new classification from positive raw transactions, verifies the 36 shard
  hashes, reloads the original model and reproduces **all 5,998,600 scores exactly**.
- Independent pandas sorting and direct formulas reproduce every requested
  aggregate metric within **1e-12**; per-query NDCG and hit counts also match.
- All 21 frozen input hashes remain unchanged. Candidate/feature/training code,
  model configuration, schema, model file and development artifacts are preserved.
- Measured build/scoring/evaluation time: **254.47 seconds**. This includes two
  snapshot builds per cutoff for leakage checks, serialization and evaluation;
  it is not a production inference latency benchmark.

Model SHA-256:
`af9492f58af0b891b6f766213a5eb823bbb7fbf46aeb1719aba6c406515219ec`.

Feature-schema SHA-256:
`2d319c1da73dd254e0e609c3b76f1982bef4b0a7b153dec17ac2630305adde8f`.

Test-data manifest SHA-256:
`6e531261aa8ab95c5fbcaac18fe2c770c15809e3fd132a4c203fc52a68f1a68b`.

New files: `src/retail_recs/heldout.py`, `scripts/evaluate_heldout.py`,
`scripts/verify_heldout.py`, `tests/test_heldout.py`, these reports and README links.
No execution failures or metric mismatches occurred.

## Artifacts, reproduction and limitations

- `artifacts/data/retail-ranking-test-v1/`: schema, frozen configuration, 12
  row/target/group snapshot sets and a complete source/provenance/checksum manifest.
- `artifacts/evaluations/heldout-v1/`: per-cutoff and consolidated predictions,
  per-query metrics and a manifest containing frozen model parameters, package
  versions, schema and checksums. Both scorers' predictions remain available.
- [phase4_test.json](phase4_test.json): full-precision results and validation comparison.
- [phase4_verification.json](phase4_verification.json): independent verification record.

The evaluator refuses to overwrite either final output directory. Verification
can be rerun with `.\.venv-phase2\python.exe scripts/verify_heldout.py`; it never
fits or selects a model. Large data/model outputs remain ignored by Git.

**80/20:** the candidate stage determines which purchases are recoverable, while
the frozen ranker prioritizes them. Temporal cutoffs enforce what could have been
known at prediction time; independent target sets expose retrieval misses. The
held-out results show that the overall advantage over popularity persists, and
that the ranker remains much stronger for replenishment than discovery.

These are offline purchase-relevance results, not causal uplift, revenue or online
acceptance. The population requires historical households; inventory and changing
catalog metadata are not modeled. Household-weeks are correlated, and no confidence
interval or formal significance claim is made. Test data has now been consumed
for final evaluation: do not reuse these results to tune or select a model and
then claim the same period as an untouched final test.

Strategy A remains the pre-approved policy. Stop here and obtain approval before
production integration. No FastAPI, Docker, SHAP or other infrastructure was started.
