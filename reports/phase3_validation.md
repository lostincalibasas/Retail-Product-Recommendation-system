# Phase 3: popularity versus LambdaRank validation

Measured on the frozen validation candidate sets. Test targets remain sealed.

## Overall validation metrics

| Metric | Recent popularity | LambdaRank |
| --- | ---: | ---: |
| End-to-end NDCG@5, all groups | 0.082016 | 0.183540 |
| End-to-end NDCG@5, purchase groups | 0.151794 | 0.339691 |
| Candidate-only NDCG@5, all groups | 0.089504 | 0.200770 |
| Candidate-only NDCG@5, groups with positive candidates | 0.184213 | 0.413216 |
| Recall@5 micro | 0.030206 | 0.072502 |
| Recall@5 macro, all groups | 0.029037 | 0.062078 |
| Recall@5 macro, purchase groups | 0.053742 | 0.114893 |
| Candidate recall micro | 0.316938 | 0.316938 |

## Repeat and new-product segments

Repeat means this household bought the product before the cutoff. New means no such prior household purchase, including globally unseen products. Segment metrics retain the original top-five positions; recommendations are not reranked within a segment.

| Segment | Targets | Retrieved | Candidate recall | Popularity Recall@5 micro | LambdaRank Recall@5 micro |
| --- | ---: | ---: | ---: | ---: | ---: |
| repeat | 170,648 | 95,729 | 0.560973 | 0.055553 | 0.137945 |
| new | 154,185 | 7,223 | 0.046846 | 0.002153 | 0.000071 |

## Population, selection and checks

- Training: 6,059,800 rows / 30,299 informative groups. Excluded from fitting only: 32,141 groups without both label values.
- Validation: 5,995,400 rows / 29,977 groups, including 13,780 no-purchase and 1,632 complete candidate-miss groups.
- All-group macro metrics assign zero to no-purchase/zero-positive groups. Purchase-only and positive-candidate-only companion metrics explicitly state their denominators. Micro recall divides total top-five hits by all actual targets.
- End-to-end ideal DCG includes true targets missing from the candidate set. Candidate-only ideal DCG uses positives present in the candidate set. Both use binary gains and log2 rank discounts.
- Ties resolve by ascending PRODUCT_ID. Baseline and ranker score exactly the same stored candidates.
- Artifact checksums, schema order, group sizes/offsets, target counts and every candidate label were checked before fitting. All validation groups retained.
- One fixed configuration; best iteration 152 selected by validation end-to-end NDCG@5 across all groups. 172 rounds executed; patience 20.
- Validation is used for early stopping; these results are development estimates, not an independent test result.
- Serialized model reload reproduced all validation scores exactly: True.

## Runtime and memory

- Verification/data loading: 43.33 s.
- Baseline evaluation: 1.38 s.
- Fit including bin construction and per-round validation: 534.05 s.
- Final LambdaRank prediction: 13.55 s.
- Peak process resident working set: 2.703 GiB. Sampled fit peak: 2.702 GiB (250 ms samples). These are this Python process, not system-wide memory.
- Float32 train + validation feature arrays: 1.123 GiB.

## Reproducibility

- Data manifest SHA-256: `3d09e8064531ddea7b173dc278468e781d4b52f9a5a5ce18e5581c74f69ba925`.
- Validation candidate identity SHA-256: `a77e0c891adf6e45cb6d1f23b9f44ba4f8f1aa99bbff0772c47a1e4869d7be14`.
- Model: `artifacts/models/lambdarank-v1/model.txt`; SHA-256 `af9492f58af0b891b6f766213a5eb823bbb7fbf46aeb1719aba6c406515219ec`.
- Artifacts also include parameters, feature schema, full scored validation candidates, group metrics and evaluation history. Large artifacts stay outside Git.

```json
{
  "num_boost_round": 200,
  "early_stopping_rounds": 20,
  "early_stopping_min_delta": 1e-05,
  "params": {
    "objective": "lambdarank",
    "device_type": "cpu",
    "metric": "None",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 100,
    "lambda_l2": 1.0,
    "max_bin": 63,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "label_gain": [
      0,
      1
    ],
    "lambdarank_truncation_level": 10,
    "lambdarank_norm": true,
    "seed": 42,
    "deterministic": true,
    "force_col_wise": true,
    "num_threads": 4,
    "verbosity": -1
  }
}
```

## 80/20 explanation

Retrieval is held fixed so the comparison measures ordering. Popularity gives every household the same product-popularity score within its candidate pool. LambdaRank can use household-product recency, frequency and preferences to personalize the order. No-purchase groups teach no preference ordering and are excluded from fitting, while every validation group remains represented. Candidate recall limits what either scorer can recover.
