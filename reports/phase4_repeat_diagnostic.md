# Phase 4 repeat-purchase diagnostic

Descriptive, post-hoc analysis of the already-opened historical test period at
`57e6647` on `main`. Strategy A remains frozen. The household-frequency baseline
was introduced after final testing; it was not part of untouched model selection.
No training, tuning, candidate/feature changes or deployment occurred.

**Conclusion:** LambdaRank beats household frequency on these observations, but
the improvement is much smaller than its advantage over global popularity:
23,524 versus 21,944 repeat hits, **+1,580 / +7.200146%**. This is useful offline
evidence of incremental ranking value, not statistical significance or proof of
customer convenience, revenue uplift, or superiority on a new untouched sample.

## Inputs and recommendation rules

Reused `artifacts/evaluations/heldout-v1/test_predictions.parquet`,
`test_group_metrics.parquet`, and the 12 snapshots under
`artifacts/data/retail-ranking-test-v1/test/cutoff_*/`. All three strategies draw
from the same original 200 candidates per household-week; no products are added.

- **Popularity:** original saved `popularity_score`, equal to historical distinct
  purchasing households in the preceding 28 days (`product_households_28d`).
- **Household frequency:** descending existing `pair_purchase_rows_84d` among
  historically purchased candidates only. This counts positive-quantity transaction
  rows in **cutoff-84 < DAY <= cutoff**, not units, lifetime purchases, or distinct
  trips. Ascending `PRODUCT_ID` breaks ties. Previously purchased products with
  zero recent frequency remain eligible. Never-purchased products are excluded;
  there is no non-repeat fallback.
- **LambdaRank:** original saved `lambdarank_score` and unmodified Strategy A
  Top 5. Ascending `PRODUCT_ID` breaks score ties for both original strategies.

Repeat membership means a positive purchase on or before the cutoff, determined
from `pair_recency_days <= cutoff_day`, consistent with saved `repeat_product`.
It is not the quota/source flag `is_repeat_candidate`. See `add_features()` in
`src/retail_recs/features.py`. The historical nonpurchase sentinel is cutoff+1.
Labels require a positive purchase in **cutoff < DAY <= cutoff+7**, counted once
per product/household/week. Cutoffs 624,631,...,701 cover target days 625-708.
Earlier test purchases can be history for later cutoffs. Target labels and the
active-week filter enter evaluation only, never recommendation selection.

There are **38 groups with fewer than five eligible repeat candidates**, leaving
**86 unfilled frequency-baseline slots** (9 in active weeks). Eligible repeat
counts range from 1 to 170. Missing slots count as misses; Precision@5 always uses
five slots. This is a disclosed eligibility difference inside the common candidate
pool: the original popularity and LambdaRank lists still allow non-repeat items.
LambdaRank contains 146,259 historical-repeat slots out of 149,965; it favors
replenishment but does not enforce a repeat-only policy. No policy was changed.

## Populations and definitions

| Denominator/count | All household-weeks | Weeks with any target purchase |
|---|---:|---:|
| Groups | 29,993 | 16,066 |
| Five recommendation slots per group | 149,965 | 80,330 |
| Actual distinct product targets | 338,792 | 338,792 |
| Purchased items present in candidates | 107,617 | 107,617 |
| Actual repeat-product targets | 183,544 | 183,544 |
| Purchased repeat items present in candidates | 100,213 | 100,213 |
| Zero candidate-positive groups | 15,525 | 1,598 |
| Zero repeat-candidate-positive groups | 15,834 | 1,907 |
| Groups with at least one actual repeat target | 14,817 | 14,817 |

The 13,927 no-purchase weeks remain in all-population metrics. Active means any
target purchase, including new products, not necessarily a repeat purchase. It is
a retrospective evaluation view, not a prediction-time eligibility rule.

For binary relevance, DCG@5 = sum(relevance at rank i / log2(i+1)), i=1..5.
IDCG(q) = sum(1/log2(i+1)), i=1..min(5,q).

- **Candidate-only NDCG:** DCG / IDCG(number of purchased candidate items).
  Repeat candidate NDCG uses repeat relevance and repeat candidate positives.
- **End-to-end NDCG:** same numerator, ideal denominator uses all actual target
  items, including candidate misses; repeat end-to-end NDCG uses all repeat targets.
- **All NDCGs are macro means over the stated groups.** Zero ideal denominator
  gives zero and stays in the mean, matching Phase 3 conventions. Repeat metrics
  retain positions in the original Top 5; non-repeat items are zero relevance.
- **Candidate Recall@5 is micro:** summed hits / summed purchased candidate items.
  Repeat candidate recall uses 100,213 as denominator. End-to-end micro recall
  uses 338,792 targets overall or 183,544 repeat targets. It is not mean group recall.
- **Precision@5:** summed hits / (5 times group count), also equal to macro mean
  group precision because the denominator is fixed at five.
- **Hit Rate@5:** groups with at least one hit / group count. Repeat hit rate
  requires at least one repeat hit. Empty groups contribute zero.

Micro recall is identical in both population views: excluded no-purchase groups
have neither hits nor targets. Decimal tables are rounded; exact integer counts
and full floating-point results are in [phase4_repeat_diagnostic.json](phase4_repeat_diagnostic.json).

## Overall performance, including new-product relevance

| View | Strategy | Candidate NDCG@5 | Candidate micro Recall@5 | End-to-end NDCG@5 | End-to-end micro Recall@5 | Precision@5 | Hit Rate@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| All | Popularity | 0.088187360 | 9.328452% | 0.081920724 | 2.963175% | 6.694229% | 24.365685% |
| All | Household frequency | 0.179978117 | 20.390830% | 0.166793892 | 6.477131% | 14.632748% | 34.264662% |
| All | LambdaRank | 0.196199240 | 21.870151% | 0.180820406 | 6.947035% | 15.694329% | 36.631881% |
| Active | Popularity | 0.164633605 | 9.328452% | 0.152934661 | 2.963175% | 12.497199% | 45.487365% |
| Active | Household frequency | 0.335994252 | 20.390830% | 0.311381128 | 6.477131% | 27.317316% | 63.967385% |
| Active | LambdaRank | 0.366276846 | 21.870151% | 0.337566690 | 6.947035% | 29.299141% | 68.386655% |

Exact all-product hit numerators: popularity **10,039**, frequency **21,944**,
LambdaRank **23,536**. Candidate-only recall divides each by **107,617**;
end-to-end recall divides by **338,792**. These reproduce the original saved
Phase 4 results for the two existing strategies.

## Repeat-purchase performance

| View | Strategy | Precision@5 | Candidate NDCG@5 | End-to-end NDCG@5 | End-to-end micro Recall@5 | Hit Rate@5 |
|---|---|---:|---:|---:|---:|---:|
| All | Popularity | 6.482179% | 0.086945846 | 0.083038792 | 5.296278% | 23.715534% |
| All | Household frequency | 14.632748% | 0.182495869 | 0.174354322 | 11.955716% | 34.264662% |
| All | LambdaRank | 15.686327% | 0.199341183 | 0.190107267 | 12.816545% | 36.601874% |
| Active | Popularity | 12.101332% | 0.162315870 | 0.155021939 | 5.296278% | 44.273621% |
| Active | Household frequency | 27.317316% | 0.340694548 | 0.325495405 | 11.955716% | 63.967385% |
| Active | LambdaRank | 29.284203% | 0.372142419 | 0.354903976 | 12.816545% | 68.330636% |

Exact repeat hit numerators are **9,721 / 21,944 / 23,524**, respectively.
Repeat candidate-only micro recall, using **100,213**, is **9.700338% /
21.897359% / 23.474000%** in either view. Repeat end-to-end recall uses **183,544**.
LambdaRank adds **1.966887 percentage points** of active-week repeat precision and
**4.363252 points** of active-week repeat hit rate over household frequency.

Supplementary **nonstandard normalized repeat recall@5** is the macro mean of
repeat hits / min(5, actual repeat target count), restricted to the **14,817**
groups with repeat targets; zero-relevance groups are excluded. Popularity:
**0.160679175**; frequency: **0.339683022**; LambdaRank: **0.374021957**.
This is not standard micro recall, and does not replace the metrics above.

## Distribution of correct recommendations

Counts below are household-weeks, not distinct households. In each row the six
columns sum to the population size. Active counts are displayed in parentheses
after all-population counts; only the zero-hit count changes between views.

| Relevance | Strategy | 0 correct: all (active) | 1 | 2 | 3 | 4 | 5 |
|---|---|---:|---:|---:|---:|---:|---:|
| Any purchase | Popularity | 22,685 (8,758) | 5,081 | 1,777 | 403 | 40 | 7 |
| Any purchase | Household frequency | 19,716 (5,789) | 4,027 | 2,805 | 1,913 | 1,092 | 440 |
| Any purchase | LambdaRank | 19,006 (5,079) | 4,223 | 3,056 | 2,098 | 1,143 | 467 |
| Repeat purchase | Popularity | 22,880 (8,953) | 4,985 | 1,701 | 381 | 39 | 7 |
| Repeat purchase | Household frequency | 19,716 (5,789) | 4,027 | 2,805 | 1,913 | 1,092 | 440 |
| Repeat purchase | LambdaRank | 19,015 (5,088) | 4,217 | 3,053 | 2,098 | 1,143 | 467 |

## Coverage, five-slot limits, and sparse histories

All-product candidate coverage is **107,617/338,792 = 31.764918%**; repeat coverage
is **100,213/183,544 = 54.598897%**. A perfect five-item oracle with unrestricted
retrieval could recover only **60,845/183,544 = 33.150089%** of repeat targets.
Restricting that oracle to these candidates lowers the ceiling to
**53,976/183,544 = 29.407662%**. LambdaRank achieves **23,524/53,976 = 43.582333%**
of candidate-feasible repeat hits. This is a diagnostic ceiling, not a model.
For all-product relevance the corresponding ceilings are **72,393/338,792 =
21.367978%** unrestricted and **56,248/338,792 = 16.602517%** with candidates.

Thus, candidate misses matter, but coverage and five-slot capacity are overlapping
constraints, not additive percentages. The five-slot limit alone makes high
micro recall impossible for large weekly purchase sets, and substantial ranking
headroom remains. No-purchase weeks further depress all-group precision, NDCG and
hit rate; they do not dilute micro recall.

Recovered segment diagnostics use fixed descriptive bins, without tuning. Recall
below is repeat end-to-end micro recall; parentheses give the exact hit numerator.
The repeat-target column is its denominator. Weekly distinct target products are
not necessarily one shopping basket.

| Eligible repeats within 200 | Groups all / active | Repeat targets | Mean repeat targets per active week | Popularity recall | Frequency recall | LambdaRank recall |
|---|---:|---:|---:|---:|---:|---:|
| 0-4 | 38 / 4 | 0 | 0.00 | undefined (0) | undefined (0) | undefined (0) |
| 5-19 | 267 / 31 | 25 | 0.81 | 8.00% (2) | 56.00% (14) | 72.00% (18) |
| 20-49 | 999 / 158 | 162 | 1.03 | 8.64% (14) | 36.42% (59) | 42.59% (69) |
| 50-99 | 12,599 / 4,798 | 26,488 | 5.52 | 6.54% (1,733) | 17.90% (4,741) | 20.16% (5,339) |
| 100-200 | 16,090 / 11,075 | 156,869 | 14.16 | 5.08% (7,972) | 10.92% (17,130) | 11.54% (18,098) |

| Household purchase rows in prior 84 days | Groups all / active | Repeat targets | Mean repeat targets per active week | Popularity recall | Frequency recall | LambdaRank recall |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2,227 / 180 | 436 | 2.42 | 6.65% (29) | 5.73% (25) | 17.66% (77) |
| 1-20 | 4,151 / 927 | 2,340 | 2.52 | 8.29% (194) | 13.42% (314) | 19.62% (459) |
| 21-100 | 9,584 / 4,233 | 20,462 | 4.83 | 6.90% (1,411) | 15.52% (3,176) | 18.26% (3,736) |
| 101-250 | 8,727 / 5,995 | 60,890 | 10.16 | 5.55% (3,382) | 13.53% (8,239) | 14.39% (8,762) |
| 251+ | 5,304 / 4,731 | 99,416 | 21.01 | 4.73% (4,705) | 10.25% (10,190) | 10.55% (10,490) |

The busiest-history segment averages 21.01 repeat targets per purchasing week;
LambdaRank repeat precision is 44.35% and hit rate 85.73%, yet micro recall is only
10.55%. Sparse recent histories have lower precision and hit rate (for zero recent
rows: 8.56% and 28.33%). These associations support both purchase-set size and
history sparsity as relevant diagnostics; they do not establish causal effects.
Small eligible-repeat bins have few active groups and should not be overinterpreted.

## Verification, reproduction, and interpretation

The interrupted calculation was recovered successfully rather than discarding
completed work. It checked all 36 test-shard hashes, prediction/group checksums,
model/data-manifest hashes, target-pair label membership and historical-repeat
flags; independent pandas and NumPy rankings/metrics agreed. This continuation
checked ID alignment, 200-row groups, offsets, repeat flags and model/manifest
hashes again. An additional NumPy calculation reproduced the recovered results
and matched `group_metrics()` in `src/retail_recs/ranking_metrics.py` for every
group within 1e-12. Frequency selections also matched per-group Python sorting.
No model inference or fitting was necessary.

Executed:

```powershell
.\.venv-phase2\python.exe -B -m pytest -q -p no:cacheprovider tests/test_ranking.py tests/test_heldout.py -k 'not fit_save_reload'
```

**15 passed, 1 deselected in 2.88s.** The synthetic training test was deliberately
excluded. Tests cover metric zeros/misses, deterministic ordering, schema/group
alignment and rejection of corrupt inputs, target boundaries, historical novelty,
future-perturbation invariance and rolling history. These tests do not refit the
frozen model. The earlier Phase 4 full-suite result remains separately documented.

To reproduce the diagnostic, align prediction rows and snapshot features by
`(household_key, cutoff_day, PRODUCT_ID)`; check group offsets and sizes; select the
first five by the three historical score rules above; join saved target relevance
only after selection; apply the formulas and population masks above. Existing
`top_indices()` and `group_metrics()` provide reference implementations. The
frequency baseline must filter historical repeats before sorting; pad missing
positions with -1 when using `group_metrics()`. Never filter recommendations by
target purchases. The JSON companion contains exact counts and unrounded metrics.

Frozen model `lambdarank-v1`, 152 trees, original 25-feature schema:

- Model SHA-256: `af9492f58af0b891b6f766213a5eb823bbb7fbf46aeb1719aba6c406515219ec`
- Test-manifest SHA-256: `6e531261aa8ab95c5fbcaac18fe2c770c15809e3fd132a4c203fc52a68f1a68b`
- Feature-schema SHA-256: `2d319c1da73dd254e0e609c3b76f1982bef4b0a7b153dec17ac2630305adde8f`

LambdaRank's all-product candidate NDCG is 0.196199 across all groups, 0.366277 in
purchasing weeks, and 0.406732 among the 14,468 groups with candidate positives.
The first two retain zero-positive groups; candidate-only normalization does not
remove the effect of completely missed or no-purchase groups. Its advantage over
household frequency is 9.012831% in candidate NDCG and 7.200146% in repeat hits.
Household frequency already captures much of the replenishment signal.

Legitimate interview claim: the frozen ranker shows incremental offline purchase
relevance over a simple personalized frequency heuristic on this observed period.
The experiment does not show causal time savings, online adoption or statistical
significance; weeks from the same household are correlated. Inventory and product
availability are not modeled. Do not tune on these results and relabel this test
as untouched. Strategy A remains unchanged.

Only this report and its JSON companion are added. Original Phase 4 reports,
model artifacts and source/configuration files are preserved. No commit or push
is created for this diagnostic; the Git checkpoint remains `57e6647` on `main`.
