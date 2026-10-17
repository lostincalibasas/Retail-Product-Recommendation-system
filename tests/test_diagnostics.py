import json

import numpy as np
import pandas as pd
import pytest

from retail_recs.contracts import FEATURE_COLUMNS
from retail_recs.diagnostics import aggregate_summaries, summarize_snapshot


def snapshot(pairs, targets, cutoff=100):
    rows = pd.DataFrame(pairs, columns=["household_key", "PRODUCT_ID"])
    rows["cutoff_day"] = cutoff
    for name in FEATURE_COLUMNS:
        rows[name] = np.zeros(len(rows), dtype=np.float32)
    target_set = set(targets)
    rows["relevance"] = np.array([int(pair in target_set) for pair in pairs], dtype=np.int8)
    return rows, pd.DataFrame(targets, columns=["household_key", "PRODUCT_ID"])


def test_hand_computed_coverage_and_group_sizes():
    rows, targets = snapshot([(1, 10), (1, 11), (2, 10), (3, 12), (3, 13), (3, 14)],
                             [(1, 10), (1, 99), (2, 88)])
    report = summarize_snapshot(rows, targets)
    assert report["candidate_rows"] == 6
    assert report["groups"] == 3
    assert report["group_size"] == {"min": 1, "max": 3, "mean": 2.0, "p50": 2.0, "p95": 2.9}
    assert report["group_size_histogram"] == {"1": 1, "2": 1, "3": 1}
    assert report["positive_labels"] == report["retrieved_target_products"] == 1
    assert report["zero_positive_groups"] == 2
    assert report["groups_with_target_purchases"] == 2
    assert report["candidate_miss_groups"] == report["no_purchase_groups"] == 1
    assert report["candidate_miss_rate"] == 0.5
    assert report["target_products"] == 3
    assert report["completely_missed_target_products"] == 2
    assert report["candidate_recall_micro"] == pytest.approx(1 / 3)
    assert report["candidate_recall_macro"] == 0.25
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("pairs", [[], [(1, 10), (1, 11)]])
def test_no_targets_zero_denominators(pairs):
    report = summarize_snapshot(*snapshot(pairs, []))
    assert report["candidate_recall_micro"] == report["candidate_recall_macro"] == report["candidate_miss_rate"] == 0.0
    assert report["candidate_miss_groups"] == report["groups_with_target_purchases"] == 0
    assert report["zero_positive_groups"] == report["no_purchase_groups"] == int(bool(pairs))
    json.dumps(report, allow_nan=False)


def test_aggregate_exact_group_weighting_and_quantiles():
    a = summarize_snapshot(*snapshot([(1, 1), (2, 1)], [(1, 1), (2, 99)]))
    b = summarize_snapshot(*snapshot([(1, 1), (1, 2), (1, 3), (1, 4)], [(1, 1), (1, 99)], cutoff=107))
    total = aggregate_summaries([a, b])
    assert total["candidate_recall_macro"] == 0.5
    assert total["candidate_recall_micro"] == 0.5
    assert total["candidate_miss_rate"] == pytest.approx(1 / 3)
    assert total["groups"] == 3 and total["snapshots"] == 2
    assert total["group_size"]["p95"] == pytest.approx(np.percentile([1, 1, 4], 95))
    assert total["group_size"]["mean"] == 2
    assert aggregate_summaries([total])["candidate_recall_macro"] == 0.5
    empty = aggregate_summaries([])
    assert empty["groups"] == empty["snapshots"] == 0
    assert empty["group_size"]["p95"] == 0.0


def test_macro_aggregation_weights_target_groups():
    a = summarize_snapshot(*snapshot([(1, 1)], [(1, 1)]))
    b = summarize_snapshot(*snapshot([(1, 1), (2, 1), (3, 1)], [(1, 99), (2, 99)]))
    assert aggregate_summaries([a, b])["candidate_recall_macro"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("problem,match", [
    ("duplicate", "Duplicate candidate"), ("mismatch", "do not match"),
    ("binary", "binary"), ("label_dtype", "dtype int8"),
    ("nonfinite", "finite"), ("feature_dtype", "dtype float32"),
    ("unsorted", "sorted"), ("multi_cutoff", "single cutoff"),
    ("missing_column", "missing required"), ("missing_id", "missing values"),
])
def test_invalid_snapshot_fails(problem, match):
    rows, targets = snapshot([(1, 1), (2, 2)], [(1, 1)])
    if problem == "duplicate":
        rows = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    elif problem == "mismatch":
        rows.loc[0, "relevance"] = 0
    elif problem == "binary":
        rows.loc[0, "relevance"] = 2
    elif problem == "label_dtype":
        rows["relevance"] = rows["relevance"].astype("int64")
    elif problem == "nonfinite":
        rows.loc[0, FEATURE_COLUMNS[0]] = np.inf
    elif problem == "feature_dtype":
        rows[FEATURE_COLUMNS[0]] = rows[FEATURE_COLUMNS[0]].astype("float64")
    elif problem == "unsorted":
        rows = rows.iloc[::-1]
    elif problem == "multi_cutoff":
        rows.loc[0, "cutoff_day"] = 99
    elif problem == "missing_column":
        rows = rows.drop(columns=FEATURE_COLUMNS[0])
    elif problem == "missing_id":
        rows.loc[0, "household_key"] = np.nan
    with pytest.raises(ValueError, match=match):
        summarize_snapshot(rows, targets)


def test_invalid_targets_fail():
    rows, targets = snapshot([(1, 1)], [(1, 1)])
    with pytest.raises(ValueError, match="unique"):
        summarize_snapshot(rows, pd.concat([targets, targets]))
    with pytest.raises(ValueError, match="without candidates"):
        summarize_snapshot(rows, pd.DataFrame([(2, 1)], columns=targets.columns))
    targets["cutoff_day"] = 101
    with pytest.raises(ValueError, match="must match"):
        summarize_snapshot(rows, targets)


def test_candidate_cap():
    rows, targets = snapshot([(1, 1), (1, 2)], [])
    assert summarize_snapshot(rows, targets, max_candidates=2)["group_size"]["max"] == 2
    with pytest.raises(ValueError, match="exceeds"):
        summarize_snapshot(rows, targets, max_candidates=1)
    with pytest.raises(ValueError, match="positive integer"):
        summarize_snapshot(rows, targets, max_candidates=0)
