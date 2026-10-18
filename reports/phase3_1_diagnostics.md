# Phase 3.1: why discovery recall fell

The dominant measured mechanism is **repeat candidates crowding purchased new
products out of the Top 5**. The saved model's trees rely strongly on
household-product history. Ordering within the new-product pool is broadly
similar to popularity; ordering new products against repeats is very different.

Recommendation: retain the current model as the replenishment benchmark and test
a discovery-aware alternative after approval. Do not present this checkpoint as
a satisfactory discovery model. No model, candidate, hyperparameter or serving
change was made during this investigation.

## Scope and definitions

Analyzed the existing 152-tree model and all **5,995,400 candidates / 29,977 weekly
household groups** in validation, with the original score ties resolved by product
ID. A new product means **new to that household**: no positive purchase at or
before its cutoff. It does not mean a newly introduced catalog SKU. All ranking
and target checks use the original seven-day windows, ending no later than day
624. The held-out test period was not accessed.

The model is unchanged at SHA-256
`af9492f58af0b891b6f766213a5eb823bbb7fbf46aeb1719aba6c406515219ec`.
The complete measurements, quantiles, five examples with both Top 5 lists and
feature vectors are in [phase3_1_diagnostics.json](phase3_1_diagnostics.json).
The [Phase 3 comparison](phase3_validation.md) remains the source for overall
performance: NDCG@5 0.082016 -> 0.183540; micro Recall@5 3.0206% -> 7.2502%.

## 1. Feature importance and learned tree structure

Gain is the sum of training split gains using a feature, normalized across the
saved model. Split count is the number of internal nodes using it. These are
model-structure statistics, not causal effects, validation permutation scores or
SHAP values. Correlated features can share or substitute importance.

| Feature | Share of total gain | Splits |
| --- | ---: | ---: |
| pair_purchase_rows_84d | 51.5563% | 400 |
| pair_quantity_84d | 16.3710% | 168 |
| pair_recency_days | 8.1517% | 468 |
| hh_purchase_rows_84d | 7.5652% | 549 |
| product_households_7d | 5.7938% | 508 |
| product_purchase_rows_28d | 4.7176% | 217 |
| pair_purchase_rows_28d | 1.2364% | 155 |
| category_affinity_84d | 0.7986% | 252 |
| is_repeat_candidate | 0.7166% | 37 |
| pair_spend_84d | 0.4596% | 146 |
| hh_purchase_rows_28d | 0.4456% | 164 |
| product_mean_price_28d | 0.3520% | 222 |
| product_households_84d | 0.3391% | 255 |
| hh_spend_28d | 0.3331% | 148 |
| product_recency_days | 0.2935% | 134 |
| brand_affinity_84d | 0.1494% | 148 |
| price_to_household_mean | 0.1379% | 73 |
| hh_purchase_rows_7d | 0.1313% | 125 |
| hh_products_84d | 0.1114% | 70 |
| hh_recency_days | 0.1104% | 89 |
| product_households_28d | 0.1047% | 99 |
| pair_mean_price_84d | 0.0774% | 69 |
| pair_purchase_rows_7d | 0.0443% | 61 |
| is_category_candidate | 0.0030% | 3 |
| is_popular_candidate | 0.0000% | 0 |

All `pair_*` features together account for **77.90% of gain**. Every tree starts
with a pair-history split: 84-day frequency (63 trees), 28-day frequency (39),
quantity (21), spending (15), 7-day frequency (7), or recency (7).

There are **1,145 household-feature splits** and **252 category-affinity splits**
below pair-history splits. Household features are constant within a query, so
their ranking influence comes through conditional tree interactions rather than
an independent household-only ordering. This establishes that such interactions
exist; the counts do not quantify a causal interaction effect.

The baseline's exact feature, `product_households_28d`, has only 0.1047% of gain.
However, popularity is not entirely ignored: 7-day household popularity has
5.7938% and 28-day purchase-row popularity 4.7176%. Category affinity has 0.7986%.
Changing the explicit `is_repeat_candidate` flag alone would not remove the
model's strong pair-history information.

## 2. Score distributions and candidate base rates

| Candidate segment | Rows | Purchased positives | Positive rate |
| --- | ---: | ---: | ---: |
| Repeat | 2,941,127 | 95,729 | 3.2548% |
| New to household | 3,054,273 | 7,223 | 0.2365% |

Repeat candidates are **13.76 times** as likely to have a positive validation
label in this candidate population. This is observational and depends on
retrieval; it is not a causal effect of recommending repeats. All new candidates
have zero pair frequency/quantity/spend features, and their recency is the
existing `cutoff + 1` never-purchased sentinel. This gives the trained trees a
consistent way to distinguish them from recent/frequent purchases.

| LambdaRank score population | Mean | 5th percentile | Median | 95th percentile |
| --- | ---: | ---: | ---: | ---: |
| All repeat candidates | -1.2885 | -2.6801 | -1.4111 | 0.3014 |
| All new candidates | -2.6646 | -3.6598 | -2.6955 | -1.5251 |
| Purchased repeat candidates | -0.1708 | -1.8969 | -0.4490 | 2.4457 |
| Purchased new candidates | -2.6030 | -3.5830 | -2.6268 | -1.5035 |

For comparison, the baseline's 28-day household-popularity scores have medians
**13 for repeats** and **70 for new products** (means 51.41 and 78.65). The two
models' numerical score scales cannot be compared directly. Ranker scores are
not calibrated probabilities, and pooled scores may also reflect household
context, so the following within-query measurements are more diagnostic:

- Across 281,246,601 repeat/new pairs within the same queries, a repeat scores
  above a new product **91.08%** of the time for LambdaRank versus **27.20%** for
  popularity. Exact ties receive half credit in these pair-weighted percentages.
- The median within-query mean-repeat minus mean-new LambdaRank score is **1.5912**.
- The median best-new minus fifth-ranked-candidate score is **-1.7168**. Even the
  highest-scoring new candidate is usually well below the recommendation threshold.

## 3. New-product Top 5 exposure

The denominator is the same **149,885 recommendation slots** in both models.
No-purchase groups are preserved; subgroup counts use future labels only for
post-hoc diagnostics, never to decide which groups receive recommendations.

| Measurement | Popularity | LambdaRank |
| --- | ---: | ---: |
| New-product slots | 65,777 | 4,651 |
| Share of all Top 5 slots | 43.8850% | 3.1030% |
| Groups with at least one new product | 26,257 | 1,769 |
| Share of all groups with new exposure | 87.5905% | 5.9012% |
| New slots in the 16,197 purchase groups | 29,816 | 369 |
| New-slot share within purchase groups | 36.8167% | 0.4556% |
| New-product hits | 332 | 11 |
| Purchases per recommended new-product slot | 0.5047% | 0.2365% |
| End-to-end new-product micro Recall@5 | 0.2153% | 0.0071% |

Most LambdaRank new-product slots occur in groups that subsequently have no
purchases: **4,282 of 4,651**. Exposure is especially restricted in the purchase
groups where new-product recall can improve. This is a measured association, not
a justification to filter groups using future activity.

## 4. Which new-product hits were lost?

All 11 LambdaRank new-product hits are also baseline hits. It loses **321** of
the baseline's 332 and gains **zero** others. Among those 321 losses:

- **302 (94.08%)** still rank in LambdaRank's first five **among new candidates
  only**. The remaining 19 are below fifth even within that restricted pool.
- Their median overall LambdaRank rank is **62**, with **60 repeat candidates**
  above them; overall ranks range from 6 to 161.
- **318** have a repeat product in LambdaRank's fifth position.
- **Zero** tie the fifth score. The median score deficit is **1.7346**, so the
  losses are due to substantive score ordering rather than product-ID tie-breaks.

Across all 7,223 retrieved new-product positives, the median rank among new
candidates is **28 for LambdaRank versus 29 for popularity**. Taking five new
candidates per query would include **910** positives under LambdaRank versus
**898** under popularity. This conditional calculation isolates within-segment
ordering; it is not a recommendation to replace the list with five new products,
an implemented alternative policy, or an independent performance estimate.

### Representative verified purchases

Examples were selected mechanically near the 0/25/50/75/100 percentiles of lost
items' LambdaRank ranks, with distinct household/cutoff queries. They illustrate
the distribution and are not a random sample. Every SKU below is in the original
independent seven-day target set and has the never-purchased recency sentinel.

| Household | Cutoff | Purchased new SKU | Target days | Popularity rank | LambdaRank rank | LambdaRank new-only rank | Repeats above | Total hits: baseline / ranker |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2150 | 589 | 1029743 | 590-596 | 3 | 6 | 4 | 2 | 1 / 1 |
| 1051 | 575 | 1029743 | 576-582 | 4 | 35 | 3 | 32 | 1 / 0 |
| 1499 | 554 | 995242 | 555-561 | 3 | 62 | 2 | 60 | 2 / 3 |
| 184 | 561 | 1082185 | 562-568 | 1 | 89 | 1 | 88 | 1 / 1 |
| 2296 | 554 | 923746 | 555-561 | 5 | 161 | 23 | 138 | 2 / 5 |

For household **1499**, SKU **995242** was bought by 400 households in the prior
28 days but had no prior purchase by this household. Its ranker score is
**-1.6111**. The fifth-ranked repeat SKU **1017799** has only five-household
popularity, but three household purchase rows in 84 days, quantity five, recency
59 days, and category affinity 0.1268. Its score is **0.8082**, a gap of 2.4193.
The new SKU is second among new candidates yet 62nd overall. The ranker still
improves this query's total hits from two to three, showing the segment tradeoff.

For household **184**, new SKU **1082185** has popularity 735 and is first under
both popularity and LambdaRank's new-only ordering. It falls to 89th overall.
The fifth repeat, SKU **907957**, has popularity two but four prior 84-day purchase
rows and recency nine days. It scores 0.5055 versus -1.7488 for the purchased new
SKU. That fifth repeat is also purchased, so this is not simply an incorrect
recommendation replacing a correct one.

For household **2296**, new SKU **923746** is also weak within the new pool
(23rd), illustrating the minority with both within-segment and cross-segment
ordering losses. Its 7-day popularity is 24 versus 28-day popularity 255. The
ranker recommends five purchased repeats in this query, despite losing this
new-product hit. The popularity discrepancy is an observed feature difference;
it does not by itself establish why this particular new-only rank changed.

### Exact tree-path score-gap accounting

For each lost purchase, compare its observed feature vector with the actual
fifth-ranked candidate in the same query. In each tree, find their first
divergent split and assign that tree's **entire leaf-score difference** to that
split. Summing across all trees reproduces each saved score gap to **1e-10**.
The total of these 321 gaps is **572.116654** score units.

| First divergent split feature | Share of summed signed gap |
| --- | ---: |
| pair_purchase_rows_84d | 41.50% |
| pair_recency_days | 28.04% |
| pair_quantity_84d | 20.86% |
| All pair-history features together | 94.49% |

This is an exact, descriptive partition of existing predictions. It includes
all interactions below the first divergent split; it does **not** isolate a
feature's causal contribution, imply that removing it eliminates that share,
or substitute for SHAP/permutation attribution. Together with the root splits,
score distributions and new-only ranks, it links the observed repeat preference
to learned pair-history decision paths rather than a sorting implementation bug.

## 5. Evidence versus hypotheses

**Verified:** retrieval, labels, candidates and model scores match the Phase 3
artifacts. The candidate pool contains slightly more new than repeat rows, so
the exposure drop is not explained by too few new candidates by count. New
candidate recall remains just **4.6846%**, but is identical for both scorers and
cannot explain the difference between 332 and 11 hits. Pair-history features
dominate training gain and the lost-hit score-gap paths. Within-new ordering is
similar, while cross-segment ordering heavily favors repeats. Ties explain none
of the lost hits. No alignment, serialization or recomputation error was found.

**Interpretation supported by these measurements:** the observed regression is
primarily the result of learned repeat-favoring score ordering, expressed through
pair-history features and their conditional interactions. The much higher repeat
label rate makes that preference consistent with optimizing aggregate NDCG.
The gain in overall ranking and the drop in discovery are therefore compatible.

**Not established:** the isolated causal effect of recency versus correlated
frequency/quantity, which hyperparameter or training-weight change would help,
or whether discovery increases customer/business value. No feature intervention,
ablation, retraining, online experiment or independent test evaluation was run.

## 6. Recommendation and next checkpoint, requiring approval

Retain the current artifact as a replenishment champion and comparison point.
For an objective that includes discovery, test a controlled alternative; keeping
the current model unchanged as the sole general-purpose recommender does not
address the measured regression.

The first proposed experiment is a **fixed-candidate discovery-aware reranking
policy**, such as reserving one of five positions for the best new-to-household
candidate. Keep the existing model and candidate pool fixed to isolate the
allocation tradeoff. This is a proposal only: no such policy was implemented or
scored. Agree on acceptable aggregate NDCG/repeat-recall loss and desired discovery
exposure before selecting a policy. Report overall and segment recall, NDCG,
exposure and coverage together, retaining empty groups.

This comes before retraining because within-new ordering is already similar to
the baseline while cross-segment exposure collapses. If a controlled reranking
policy cannot meet the agreed tradeoff, a later candidate would be a segment-aware
objective or separate discovery scorer. Do not remove history features solely
because their importance is high; they also support the measured overall gain.

Validation has already informed early stopping and these diagnostics. It can
support development comparisons, but the current findings do not justify fresh
independent-performance claims. Keep test data sealed until the policy and
evaluation protocol are finalized and test access is explicitly approved.

## Verification, artifacts and handover

- Diagnostic command: `.\.venv-phase2\python.exe scripts/diagnose_ranker.py`.
  It never calls model fitting and refuses to overwrite its output directory.
- Verified all **36 validation shard hashes**, exact identifiers, labels,
  feature order, group offsets/sizes and repeat flags. All **5,995,400** scores
  were recomputed from the existing model and matched saved scores exactly.
- All nine model-run files, including the manifest, were checksum-verified before
  and after diagnostics and remain unchanged. No raw transaction data or test
  targets were read. Only original validation row/group/target shards were loaded.
- An independent pandas sort/group check reproduced both models' new-product
  exposure, new-only Top 5 hit counts, 321 lost / zero gained hits, and the 302
  losses still ranked among the first five new candidates. Both diagnostic JSON
  copies and the recorded diagnostic source hash also matched.
- Targeted tests: **16 passed in 1.86 seconds**. Full suite: **50 passed in
  3.03 seconds**, including existing synthetic training fixtures, not retraining
  the saved Phase 3 artifact. New tests cover exposure/tie conventions,
  conditional ranks, signed tree-gap interactions and pairwise comparisons.
- Measured diagnostic runtime: **28.94 seconds**. No dependency installation.
- New implementation: `src/retail_recs/model_diagnostics.py`,
  `scripts/diagnose_ranker.py`, `tests/test_model_diagnostics.py`.
- Ignored outputs: `artifacts/diagnostics/lambdarank-v1/diagnostics.json`,
  `feature_importance.csv` (all 25 features), `lost_new_hits.csv` (all 321 losses).
  Aggregate measurements and five examples are versioned in this report and JSON.
- The diagnostic source SHA is recorded in the JSON. Execution started from
  checkpoint `1bdaf62`, with the new diagnostic implementation not yet committed;
  the final local checkpoint contains that implementation and these results.

No experiment or further implementation should start until this report is reviewed.
