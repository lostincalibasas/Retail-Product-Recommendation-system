# Phase 3 integration checkpoint

Completed the authorized baseline-versus-LambdaRank validation comparison from
`829e0b9`. Implementation was committed as `327c3da` before the full run, with a
clean working tree. Results and this handover are committed separately. No remote
push, candidate changes, SHAP, API, Docker, MLflow or test evaluation was performed.

## Actual outcome

See [the generated comparison](phase3_validation.md) and its full-precision
[machine-readable record](phase3_validation.json). On the identical 5,995,400
validation candidates, end-to-end NDCG@5 across all groups improved from 0.082016
to 0.183540 (+123.79% relative). Micro Recall@5 increased from 3.0206% to 7.2502%.
Candidate-only NDCG@5 across all groups increased from 0.089504 to 0.200770.

This is a strong repeat-purchase result with a substantial discovery regression:

| Segment | Actual targets | Candidate recall | Popularity top-five hits | LambdaRank top-five hits | Popularity micro Recall@5 | LambdaRank micro Recall@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeat for household | 170,648 | 56.0973% | 9,480 | 23,540 | 5.5553% | 13.7945% |
| New to household | 154,185 | 4.6846% | 332 | 11 | 0.2153% | 0.0071% |

These are observed counts, not an inferred causal explanation. The ranker selected
very few successful new-to-household items. Sparse candidate coverage constrains
discovery, but does not fully explain the regression: both scorers had the same
7,223 new-product positives available. Aggregate NDCG selection did not protect
this segment. A future experiment should first agree whether the business goal
is replenishment, discovery, or both; no corrective experiment has been started.

## Populations and boundaries

| Population | Groups | Rows | Positive labels | Negative labels |
| --- | ---: | ---: | ---: | ---: |
| Frozen training | 62,440 | 12,488,000 | 221,909 | 12,266,091 |
| Informative training used for fit | 30,299 | 6,059,800 | 221,909 | 5,837,891 |
| Validation, all retained | 29,977 | 5,995,400 | 102,952 | 5,892,448 |

Every group has 200 candidates (minimum = maximum = 200). Groups identify a
household and weekly cutoff, and their rows are contiguous. Fitting excludes
32,141 all-negative training groups as whole groups; it samples no rows or
negatives from the remaining groups. These excluded queries have no positive
ranking preference to learn.

Validation retains 13,780 no-purchase groups and 1,632 purchase groups with no
candidate positives: 15,412 zero-positive groups in total. The complete candidate
miss rate is 1,632 / 16,197 purchase groups = **10.0759%**. Candidate recall is
102,952 / 324,833 actual household-week-product targets = **31.6938%**; 68.3062%
of targets are absent from the candidate pool.

All-group NDCG and macro recall assign zero when their denominator is zero.
Purchase-only and positive-candidate-only companion metrics are separately named
in the generated report. Micro recall includes all actual target products,
including retrieval misses. No-purchase groups remain in outputs and macro
evaluation, while contributing zero targets to the micro denominator.

History includes `DAY <= cutoff`; targets are `cutoff < DAY <= cutoff + 7`.
Training cutoffs 365-533 yield target days 366-540. Validation cutoffs 540-617 yield
target days 541-624. Test target days 625-708 remain sealed and unmaterialized.
Past validation purchases can become history at later validation cutoffs, matching
weekly deployment. No feature or candidate construction was changed in this phase.

## Executed verification

- `.\.venv-phase2\python.exe -m pytest -q`: **47 passed**, latest suite runtime
  **4.44 seconds**. This includes the original leakage/data tests and 13 added
  cases covering metric arithmetic, empty groups, deterministic ties, schema and
  group corruption, disabled test loading, and synthetic fit/save/reload.
- The full run checked hashes of all **111** frozen Parquet files, original
  data-generation source hashes, the fixed configuration and all 37 cutoff shards.
- Every candidate label was matched to its independent target set. Exact feature
  order/dtypes, group identities/sizes/offsets/positive counts, target counts and
  trainability flags passed. Global offsets were rebuilt after whole-group filtering.
- The trained and reloaded model has exactly the 25 ordered schema features.
  All **5,995,400** validation scores matched exactly after native model reload.
- A separate post-run Python check verified **8 artifact hashes** and **9 source
  hashes**, matched every saved prediction ID/label/popularity score/repeat flag
  against the original validation shards, and confirmed all **29,977** groups.
- That check independently sorted predictions with pandas and recomputed
  end-to-end NDCG, candidate-only NDCG, micro Recall@5, candidate recall and hit
  counts for both models and all three populations (overall/repeat/new).
  Results matched the saved report to absolute tolerance **1e-12**.
- The test artifact directory is absent; the loader explicitly rejects `test`.
  Development-only transaction reads stop at day 624. Source-file hashing checks
  bytes without using held-out records as training or evaluation targets.

## Execution and reproducibility

Python 3.12.15; LightGBM 4.6.0; NumPy 2.2.6; pandas 2.3.3; PyArrow 24.0.0.
The existing isolated environment and dependency locks were used unchanged.

One fixed configuration ran 172 rounds and selected iteration **152**, using
validation end-to-end NDCG@5 with patience 20 and minimum improvement 1e-5.
Main parameters: learning rate 0.05, 31 leaves, minimum 100 rows/leaf, L2=1,
63 histogram bins, LambdaRank truncation 10, binary gains, seed 42, deterministic
CPU execution with four threads. No hyperparameter search was performed.

Measured wall-clock fit time was **534.05 seconds**, including bin construction
and per-round validation. Loading/verification took **43.33 seconds**; baseline
evaluation **1.38 seconds**; final prediction **13.55 seconds**. The recorded run
took **610.25 seconds** through scoring/reload/output preparation. Progress logs
show a long interval between rounds 130 and 135; its cause was not established.
These are actual elapsed times, not isolated CPU benchmarks.

OS peak process resident memory was **2.703 GiB**; the 250 ms sampled fitting peak
was **2.702 GiB**. Train plus validation float32 feature arrays used **1.123 GiB**.
Memory figures cover the training Python process, not the whole workstation.

The ignored directory `artifacts/models/lambdarank-v1/` holds the saved native
model, exact schema/configuration, all validation predictions, per-group metrics,
evaluation/progress history and a complete checksum manifest. It is protected
against overwrite. Source and small reports are versioned in Git; datasets and
model outputs remain local. The manifest identifies implementation commit
`327c3dabb886798dc9792bec423b845979df8c78` and records a clean run-start code state.

## Files and 80/20 handover

- `src/retail_recs/training_data.py`: checks and loads frozen development data.
- `src/retail_recs/ranking_metrics.py`: shared popularity scoring, deterministic
  top-five ranking, full-target and candidate-only metrics, repeat/new segments.
- `src/retail_recs/train_ranker.py` and `scripts/train_ranker.py`: fit, measure,
  reload-check and save the comparison with provenance.
- `configs/lambdarank_v1.json`, `docs/phase3_contract.md`, `tests/test_ranking.py`:
  reproducible parameters, exact evaluation policy and regression checks.
- `README.md` and `reports/phase3_*`: entry points, measured results and handover.

Holding candidates fixed makes this a comparison of ordering quality. The model
learns which existing candidates to move into the first five positions; it cannot
retrieve missing purchases. Removing uninformative groups only from fitting
avoids useless ranking work while retaining every validation query exposes misses
and inactivity. Explicit feature order and contiguous query groups prevent silent
interface errors. A custom full-target metric prevents candidate-only evaluation
from hiding retrieval failures.

No execution failure remains unresolved. The main limitations are the discovery
regression, low candidate coverage, and validation reuse for early stopping.
Repeated household-week observations are also dependent; no confidence intervals
or independent test-performance claims are made. The next step is user review of
the replenishment-versus-discovery tradeoff and approval of a subsequent checkpoint.
No further training, test evaluation or serving work should run automatically.
