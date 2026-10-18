# Phase 3: frozen-candidate ranking comparison

This phase resumes data checkpoint `829e0b9` and changes no candidate generation,
feature engineering or relevance labels. Only train/validation shards are loaded.
Test target days 625-708 stay sealed; there is no test-loader option.

## Inputs and checks

`training_data.check_manifest()` requires the complete `retail-ranking-v1` schema,
configuration and fixed development cutoff lists. It checks all 111 Parquet file
hashes and original code/source hashes before loading. `verify_shard()` checks
the exact column sequence, finite float32 features, int64 identifiers, int8 binary
labels, target-pair alignment, per-group sizes/positives, offsets, true target
counts and trainability flags. After concatenation, global offsets are rebuilt.

Training retains complete groups that have both positive and zero labels. In the
frozen data that is 30,299 groups / 6,059,800 candidate rows; the 32,141 excluded
groups have zero positives and cannot supply a ranking preference. No positives
or sampled negatives are added. All 29,977 validation groups / 5,995,400 candidates
remain for both scorers. Each group has 200 rows. Identifiers never enter features.

## Baseline and model

The recent-popularity baseline is `product_households_28d`, already computed from
history in each saved row. Equal scores are ordered by ascending PRODUCT_ID.
The ranker receives the 25 features in the explicit `feature_order` list, binary
labels, and the exact corresponding group sizes. Queries are household/cutoff
pairs, not whole households spanning many weeks.

One fixed CPU LightGBM configuration is selected before running. There is no
hyperparameter search. `configs/lambdarank_v1.json` specifies 200 maximum rounds,
20-round early stopping with minimum improvement 1e-5, learning rate .05,
31 leaves, minimum 100 rows/leaf, L2=1, 63 bins, four threads, deterministic mode,
seed 42 and binary gains. LambdaRank truncation is 10 to focus learning near the
top-five evaluation depth. All retained training rows are used. Smaller histogram
bins and four threads bound local resource use; no claim is made that these
parameters are optimal.

The selected boosting iteration maximizes custom end-to-end validation NDCG@5
with the defined all-group denominator. Built-in candidate-only NDCG is disabled
to prevent its ideal ranking/empty-group conventions determining selection.
Validation is therefore a development estimate, not an unbiased final test.

## Metric definitions and empty groups

For group g, let T be its number of actual target products (including retrieval
misses), P the candidate-positive count, and h_i the binary relevance of the item
at recommended rank i. DCG@5 = sum(h_i / log2(i+1)), i=1..5.

- End-to-end NDCG@5 divides DCG by the ideal sum for min(5,T) positives.
- Candidate-only NDCG@5 divides DCG by the ideal sum for min(5,P) positives.
- An empty ideal denominator produces zero. Primary macro means retain **all**
  validation groups, including no-purchase and zero-positive groups.
- Companion end-to-end NDCG and macro Recall@5 average over purchase groups only;
  companion candidate-only NDCG averages over groups with candidate positives.
  Their different denominators are explicit in output names.
- Micro Recall@5 = total top-five purchased-product hits / total actual targets.
  Macro Recall@5 averages per-group hits/T, assigning zero when T=0, with a separate
  purchase-only companion.
- Micro candidate recall = sum(P)/sum(T). It must be identical for both scorers.
- Purchase groups with P=0 contribute zero to all end-to-end ranking metrics.

Repeat/new segmentation is by **all historical household-product purchases**, not
the limited repeat candidate source flag. First purchase days are read using
`DAY <= 624`; a target is repeat iff its first positive purchase day <= cutoff.
The saved pair-recency feature determines the same classification for candidates.
Segment metrics retain the original top-five list and positions; they do not
rerank a subset. Segment true-target counts include targets retrieval missed.

## Artifacts, runtime and limits

`artifacts/models/lambdarank-v1/` contains the native LightGBM text model, exact
configuration/schema, full scored validation candidates, per-group metrics,
evaluation history, progress records and a manifest with file/code/data hashes.
The model is saved at the selected iteration and reloaded; every validation score
must match exactly. Artifact directories cannot be overwritten.

The report records data-load, baseline, fit and prediction timings. Fit timing
includes histogram construction and per-round validation. Memory records process
resident working set and OS lifetime peak, plus a 250 ms sampled fitting peak.
These measurements are not system-wide RAM usage. Float32 arrays are preallocated
and populated shard by shard to avoid concatenating full feature DataFrames.

Model scores are ordering scores, not calibrated purchase probabilities. No SHAP,
FastAPI, Docker, MLflow or held-out test evaluation belongs to this checkpoint.
