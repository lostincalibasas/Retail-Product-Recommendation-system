# Phase 3.2: controlled discovery-slot experiment

Measured validation experiment only. Strategy B is not promoted.

A is the unchanged LambdaRank Top 5. B keeps ranks 1-4 exactly and puts the highest-scoring new-to-household candidate not already selected in position 5. This slot is reserved even when ranks 1-4 already contain new products. Ties use ascending PRODUCT_ID. If no eligible new candidate exists, the original fifth item remains.

## Measured results

| Metric | A: LambdaRank | B: discovery slot | B minus A | Relative change |
| --- | ---: | ---: | ---: | ---: |
| End-to-end NDCG@5, all groups | 0.18354011 | 0.17030300 | -0.01323711 | -7.21% |
| Precision@5, all groups | 0.15712713 | 0.13705174 | -0.02007539 | -12.78% |
| Recall@5, micro | 0.07250187 | 0.06323865 | -0.00926322 | -12.78% |
| Repeat-product Recall@5, micro | 0.13794478 | 0.11913412 | -0.01881065 | -13.64% |
| New-product Recall@5, micro | 0.00007134 | 0.00137497 | +0.00130363 | +1827.27% |
| New-product Top 5 slot share | 0.03103046 | 0.22189012 | +0.19085966 | +615.07% |

Metrics are fractions, not percentages. NDCG is the mean across all groups with full-target ideal DCG. Precision@5 is mean hits/5 across all groups (equivalently total hits / all 149,885 slots). Recall and segment recall are micro: hits / all actual targets, including candidate misses. The new-slot share is new recommendations / all recommendation slots.

No-purchase and zero-positive-candidate groups remain in evaluation. Undefined per-group NDCG/recall is zero. The JSON also includes purchase-only and macro companion metrics with explicit denominators.

## Exposure and affected groups

- Same 5,995,400 frozen candidates in 29,977 household-week groups; 200 candidates/group and five recommendations/group.
- Changed fifth item in **28,607 groups (95.4298%)**; ranks 1-4 identical everywhere.
- An eligible unselected new candidate existed in 29,977 groups; fallback used in 0.
- New slots: 4,651 -> 33,258; groups with any new recommendation: 1,769 -> 29,977.
- All-target candidate recall stays 0.31693824; no candidates or future positives were injected.

## Hit-level accounting and regressions

- Total hits: 23,551 -> 20,542.
- Repeat hits: 23,540 -> 20,330 (denominator 170,648 true repeat targets).
- New hits: 11 -> 212 (denominator 154,185 true new targets).
- Changed-group fifth-slot transitions: {"0->0": 25237, "0->1": 160, "1->0": 3169, "1->1": 41}. Keys describe old_hit -> new_hit.
- Group NDCG improved in 160, regressed in 3,169, and was unchanged in 26,648 groups.

Only position 5 changes, so a replaced repeat purchase and added new purchase trade off at the same discount. All measured regressions are reported above; increased exposure alone does not establish business value.

## Verification and provenance

- The selection function accepts only scores, product IDs, group sizes and historical novelty. Relevance labels and target counts are used afterwards solely for evaluation.
- Historical novelty is verified against the original pair-recency feature: never purchased means pair_recency_days == cutoff + 1. Candidate history was built with DAY <= cutoff; validation targets span cutoff < DAY <= cutoff + 7.
- All 36 validation row/group/target file hashes passed. Every saved prediction ID, relevance label, group count and repeat flag was matched to its frozen shard. Candidate labels were independently checked against seven-day target sets.
- Both lists contain five distinct original candidates from the same query. No output can add an SKU, cross a household/week boundary, or change ranks 1-4. All changed fifth items are new products replacing repeats.
- Strategy A reproduces every existing overall/repeat/new metric within 1e-12. All model-run artifacts were checksum-verified before and after the experiment and remain unchanged.
- Model SHA-256: `af9492f58af0b891b6f766213a5eb823bbb7fbf46aeb1719aba6c406515219ec`; prediction SHA-256: `78f3fe9baee4e509a7f8435d69ae740bd9ddd5550bc7c6dee3c94f018c0680ec`.
- Base Git checkpoint: `555c26353641bba181f6703926c7c18af5fca82f`. New experiment source SHA-256: `1f22af23e405fd6e094a69c7209941c4fe839cd3ef5075a1cef7e166de3620f6`.
- Experiment elapsed time through evaluation and output preparation: 8.99 seconds.
- Validation cutoffs 540-617 / target days 541-624 only. No raw transaction reads, test access, fitting, feature changes, candidate changes or hyperparameter changes.

Artifacts: `artifacts/experiments/discovery-v1/` contains both complete recommendation lists, per-group metrics, and the result JSON. Existing model files and earlier phase reports are unchanged. The experiment refuses to overwrite its output directory.

## Decision boundary

Keep A as the current model policy pending review. B is a development comparison, not an automatically selected or deployed strategy. Validation was already used for model selection and diagnostics; this is not independent test evidence. Review the discovery gain against repeat and overall losses before approving any next step.
