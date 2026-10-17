# Phase 2 data and agent contract

This independent portfolio project ranks household product purchases over the next
seven relative dataset days. It has no sales-representative acceptance, inventory,
impression, or causal uplift labels. Current checkpoint stops before training.

## Time and population

At cutoff `c`, history is `DAY <= c`; the target window is `c < DAY <= c + 7`.
All purchase histories and labels require `QUANTITY > 0`. Zero/nonpositive
quantities are counted in the source audit and excluded; transactions are not
deduplicated. Labels deduplicate household/product pairs across the target week.

| Split | Weekly cutoffs, inclusive | Target days, inclusive | State |
| --- | --- | --- | --- |
| Train | 365, 372, ..., 533 (25) | 366–540 | Materialized |
| Validation | 540, 547, ..., 617 (12) | 541–624 | Materialized |
| Test | 624, 631, ..., 701 (12) | 625–708 | Sealed until model selection |

Day 1–365 provides warm-up history. The loader predicate excludes transactions
after day 624, and this checkpoint has no test-materialization CLI option.
During later rolling evaluation, history may include purchases already observed
in earlier evaluation weeks, while model weights remain fixed. Training labels
end before validation targets begin; validation labels end before test targets.

Eligible households have at least one positive-quantity purchase on/before the
cutoff, including currently inactive households. Future first-time households are
excluded from known-household evaluation and counted per target window. An eventual
serving fallback for unknown households will be evaluated separately.

## Candidates and features

Per household, union up to 80 historical repeat products and 60 products popular
within its top three categories. Repeat ranking uses last purchase day, purchase-row
frequency, then product ID. Category preference and product popularity use 84-day
purchase-row counts, with category/product tie breakers. These quotas cap source
pools before deduplication, not disjoint final counts. Fill to 200 with 28-day
distinct-household product popularity; backfill with all-history popularity.
Global ties use product ID. Small catalogs can yield fewer than 200 candidates.

Only historically purchased products enter candidates. Future positive targets
are never inserted. Older repeats remain eligible because sales history is not
an inventory feed. Source flags describe pool memberships and may overlap.

The shared `FEATURE_COLUMNS` list in `contracts.py` contains 25 ordered float32
features: household activity/spend, product popularity/price, household-product
activity/recency, category/brand affinity, relative price, and candidate-source
flags. Windows are `(c - window, c]` for 7/28/84 days; recencies use all history.
Unit prices are sum of sales divided by sum of positive quantity, not mean row
sales. No-history counts/ratios are zero; unseen pair recency is `c + 1`. These
sentinels are part of schema v1. No raw identifiers or labels enter model features.

Product catalog category/brand are assumed static; they are joined only for
historically observed products. `BRAND` is the dataset's broad national/private
label, not manufacturer identity. Demographics and causal/promotion tables are
excluded in this version. Negative sales on positive-quantity rows are retained
and audited, rather than silently changing source meaning.

## Artifact and training interfaces

`build_snapshot(transactions, products, cutoff, config)` returns sorted IDs plus
ordered features, with no labels. `build_targets(...)` returns unique independent
actual purchases. Only the coordinator joins them to create binary `relevance`.

Each cutoff writes:

- `rows.parquet`: int64 household/cutoff/product IDs, 25 float32 features, int8 relevance.
- `targets.parquet`: all unique eligible household target products, including retrieval misses.
- `groups.parquet`: household/cutoff, group size, shard-relative row offset, positive
  labels, all-target count, and whether the group has both positive and zero labels.
- `diagnostics.json`: measured group sizes, positives, coverage and misses.

Rows are ordered by household, cutoff, product inside each shard. A future trainer
must concatenate shards in manifest order with their corresponding group sizes;
it must never globally sort one without rebuilding the other. Filter whole
non-trainable groups for LambdaRank fitting, while preserving all evaluation groups.
The baseline and ranker must receive the exact same candidate rows.

`manifest.json` with `status=complete` is the artifact completion marker. Output
directories cannot be overwritten. A failed run retains an `INCOMPLETE` marker
for inspection and must be rerun to a new directory. Manifests include schema,
configuration, seed, code/source/output hashes, environment versions, Git base
commit and dirty state. File hashes cover complete source files; no test rows are
decoded into the development DataFrame or used for diagnostics. No model artifact
exists yet; future models must reference this manifest and feature-schema version.

Micro candidate recall is retrieved household/product targets divided by all
targets. Macro recall averages per-household-week recall over purchase weeks.
Candidate-miss rate is the fraction of purchase weeks with zero retrieved targets.
No-purchase and zero-positive candidate groups are reported separately. Rates with
zero denominators are zero alongside their denominator counts. NDCG/Recall@5 are
deferred until ranking; end-to-end ideal DCG must use all actual targets, including
candidate misses, and candidate-only ranking scores must be labeled separately.

## Exclusive agent ownership

| Owner | Files | Prerequisite |
| --- | --- | --- |
| Coordinator | contracts, pipeline, scripts, configuration, environment locks, docs, integration tests, Git | Approved design |
| Data Agent | `data.py`, `features.py`, `tests/test_data.py` | Shared contract |
| ML Agent | `diagnostics.py`, `tests/test_diagnostics.py`; later modeling/evaluation modules | Shared contract; approved data checkpoint before modeling |
| Serving Agent | Future API, Docker, endpoint tests | Inference contract; trained artifact before integration |

No Serving Agent is needed for this checkpoint. Shared-file changes require a
message to the coordinator; agents do not modify Git or one another's files.
