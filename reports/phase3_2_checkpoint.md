# Phase 3.2 verification and decision record

Completed the approved experiment from checkpoint `555c263` using the three
preserved draft files. The frozen model and validation predictions were reused.
No retraining, candidate/feature changes, test access or infrastructure work occurred.

## Recommendation: retain original ranking A

Do not adopt the unconditional discovery-slot rule as the default. It increases
new-product hits from **11 to 212**, but removes **3,210 repeat hits** to obtain
**201 additional new hits**. Total hits fall from **23,551 to 20,542**, a net
loss of **3,009**. End-to-end NDCG@5 declines **7.21% relative**, overall precision
and recall decline **12.78%**, and repeat recall declines **13.64%**.

| Metric | A: original | B: discovery slot | Absolute change |
| --- | ---: | ---: | ---: |
| End-to-end NDCG@5, all groups | 0.183540 | 0.170303 | -0.013237 |
| Precision@5, all groups | 15.7127% | 13.7052% | -2.0075 pp |
| Recall@5, micro | 7.2502% | 6.3239% | -0.9263 pp |
| Repeat-product Recall@5, micro | 13.7945% | 11.9134% | -1.8811 pp |
| New-product Recall@5, micro | 0.0071% | 0.1375% | +0.1304 pp |
| New-product Top 5 slot share | 3.1030% | 22.1890% | +19.0860 pp |

The rule changes **28,607 / 29,977 groups (95.4298%)**. New recommendation slots
increase from **4,651 to 33,258**; groups with any new recommendation increase
from **1,769 to all 29,977**. An eligible unselected new candidate exists in every
group, so fallback is never needed. This applies even when the first four already
contain new products: the fifth slot is always reserved for another eligible new
candidate, exactly as requested.

Group NDCG improves in **160** groups, declines in **3,169**, and stays unchanged
in **26,648**. The fifth-slot transition counts are 25,237 nonhit->nonhit,
160 nonhit->hit, 3,169 hit->nonhit and 41 hit->hit. All changed slots replace a
repeat product with a new one; ranks 1-4 are identical in every group.

The large relative discovery gain starts from only 11 hits. It should be weighed
against the absolute purchase-hit losses, not interpreted as sufficient business
value by itself. No economic value, causal uplift or online conversion was measured.
Retain A pending user review; B has not been promoted or connected to inference.

## Implementation review and tests

The policy function accepts only frozen scores, query sizes, product IDs and
historical new/repeat flags. It cannot inspect future relevance labels or target
counts. Ties use ascending product ID; an exhausted new-product pool preserves
the original fifth item. Membership checks prevent duplicate products, crossing
query boundaries and modifications to ranks 1-4.

The initial targeted run had **4 passes and 1 failure**: the label-invariance test
accessed a `product` column as `f.product`, which resolves to pandas' product
method. Explicit bracket column access fixed the fixture. No change to the
ranking rule was necessary.

- Five proposed tests: **5 passed in 0.72 seconds** after that correction.
- Full regression suite: **55 passed in 3.10 seconds**. The existing synthetic
  training fixtures run as unit tests; the saved LambdaRank model was not retrained.
- The actual experiment completed evaluation/output preparation in **8.99 seconds**.

The experiment checked all **36 validation shard hashes**, candidate identifiers,
target-label alignment, group counts/offsets, feature schema and historical novelty.
Every new flag matches `pair_recency_days == cutoff + 1`. Original historical
features use `DAY <= cutoff`; validation labels cover the next seven days through
day 624. The test period remains sealed. No raw transactions were read.

## Independent verification

A separate Python calculation, importing neither the experiment helpers nor
the ranking-metric helpers, reconstructed both lists using pandas sorting and
grouping. It selected solely from scores, identifiers and historical novelty.

- Exactly **149,885 recommendations per strategy** matched the saved lists,
  including source row index, product/household/cutoff, rank, score, novelty and label.
- Every recommendation belongs to its original candidate group; no duplicates.
  Both lists have exactly five recommendations per group.
- All six requested aggregate metrics and all saved per-group metrics matched
  independent direct formulas within absolute tolerance **1e-12**.
- The affected-group count and eligible-group count matched exactly.
- All **four experiment artifact hashes** passed. Both result JSON copies match,
  and the recorded experiment source hash matches the source file.
- All **nine frozen model-run artifact hashes**, including the manifest, remain
  unchanged. Strategy A reproduces the earlier Phase 3 overall and segment metrics.

The independent numerical record is [phase3_2_verification.json](phase3_2_verification.json).
Full experiment results are in [phase3_2_discovery.md](phase3_2_discovery.md) and
[phase3_2_discovery.json](phase3_2_discovery.json). Both recommendation lists and
per-group metrics remain local under `artifacts/experiments/discovery-v1/`.

## Reproduction and stopping boundary

```powershell
.\.venv-phase2\python.exe -m pytest -q
.\.venv-phase2\python.exe scripts/experiment_discovery.py
```

The second command intentionally refuses to overwrite an existing experiment.
Inspect the saved reports to review this run; an explicitly authorized rerun can
use a new `--output` directory. No model training command is required.

NDCG/precision retain all groups, including no-purchase groups; micro recall uses
all actual target products, including candidate misses. Candidate recall remains
31.6938%. Validation previously informed early stopping and diagnostics, so this
comparison is development evidence rather than an independent test estimate.

Stop here for review. Do not promote B or begin another policy experiment,
training run, held-out evaluation or serving implementation without approval.
