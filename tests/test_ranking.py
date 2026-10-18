import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from retail_recs.contracts import FEATURE_COLUMNS, ID_COLUMNS
from retail_recs.pipeline import label_snapshot, snapshot_groups
from retail_recs.ranking_metrics import evaluate, top_indices, make_validation_metric, recent_popularity_scores
from retail_recs.training_data import RankingData, verify_shard, load_split
from retail_recs.train_ranker import fit_ranker, process_memory


def test_metrics_include_no_purchase_and_complete_retrieval_misses():
    sizes = np.array([3, 3, 3])
    labels = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    ids = np.tile([30, 10, 20], 3)
    scores = np.ones(9)
    groups = pd.DataFrame({"target_count": [3, 1, 0], "repeat_target_count": [1, 0, 0], "new_target_count": [2, 1, 0]})
    repeat = np.tile([True, False, False], 3)
    report, per_group, top = evaluate(scores, labels, sizes, ids, groups, repeat)
    discount = 1/np.log2(3) + 1/np.log2(4)
    assert top[0].tolist() == [1, 2, 0, -1, -1]  # ID ascending breaks equal scores
    overall = report["overall"]
    assert overall["groups"] == 3
    assert overall["no_purchase_groups"] == overall["candidate_miss_groups"] == 1
    assert overall["ndcg5_e2e_all_groups"] == pytest.approx(discount / (1 + discount) / 3)
    assert overall["ndcg5_candidate_all_groups"] == pytest.approx(discount / (1 + 1/np.log2(3)) / 3)
    assert overall["ndcg5_e2e_purchase_groups"] == pytest.approx(discount / (1 + discount) / 2)
    assert overall["recall5_micro"] == 0.5
    assert overall["candidate_recall_micro"] == 0.5
    assert report["repeat"]["recall5_micro"] == 1.0
    assert report["new"]["recall5_micro"] == pytest.approx(1/3)
    # Segment uses its position in the original top-five list, not a reranked list.
    assert report["repeat"]["ndcg5_e2e_purchase_groups"] == 0.5
    assert per_group.loc[2, "ndcg5_e2e"] == 0
    callback = make_validation_metric(labels, sizes, ids, groups.target_count)
    name, value, higher = callback(scores, None)
    assert value == pytest.approx(overall["ndcg5_e2e_all_groups"])
    assert higher and name == "ndcg5_e2e_all_groups"
    json.dumps(report, allow_nan=False)


def test_variable_groups_and_score_order():
    result = top_indices([0, 3, 2, 5], np.array([1, 3]), [4, 3, 2, 1])
    assert result.tolist() == [[0, -1, -1, -1, -1], [3, 1, 2, -1, -1]]
    with pytest.raises(ValueError):
        top_indices([np.nan], [1], [1])
    with pytest.raises(ValueError):
        top_indices([1], [2], [1])


def test_baseline_is_recent_popularity_and_requires_exact_features():
    X = np.zeros((2, 25), dtype=np.float32)
    X[:, FEATURE_COLUMNS.index("product_households_28d")] = [2, 8]
    assert recent_popularity_scores(X, FEATURE_COLUMNS).tolist() == [2, 8]
    with pytest.raises(ValueError, match="feature order"):
        recent_popularity_scores(X, list(reversed(FEATURE_COLUMNS)))


def shard():
    rows = pd.DataFrame({"household_key": [1, 1, 2, 2], "cutoff_day": [365]*4, "PRODUCT_ID": [10, 20, 10, 20]}, dtype="int64")
    for name in FEATURE_COLUMNS:
        rows[name] = np.zeros(4, dtype=np.float32)
    targets = pd.DataFrame({"household_key": [1, 2], "cutoff_day": [365, 365], "PRODUCT_ID": [10, 99]}, dtype="int64")
    rows = label_snapshot(rows, targets)
    return rows, targets, snapshot_groups(rows, targets)


@pytest.mark.parametrize("corruption", ["offset", "size", "positive", "target", "flag", "feature_order", "label", "cutoff"])
def test_reject_corrupt_training_interfaces(corruption):
    rows, targets, groups = shard()
    verify_shard(rows, targets, groups, 365)
    if corruption == "offset":
        groups.loc[1, "row_offset"] = 1
    elif corruption == "size":
        groups.loc[1, "group_size"] = 1
    elif corruption == "positive":
        groups.loc[0, "positive_labels"] = 0
    elif corruption == "target":
        groups.loc[1, "target_count"] = 0
    elif corruption == "flag":
        groups.loc[1, "trainable"] = True
    elif corruption == "feature_order":
        rows = rows[ID_COLUMNS + list(reversed(FEATURE_COLUMNS)) + ["relevance"]]
    elif corruption == "label":
        rows.loc[0, "relevance"] = 0
    else:
        targets.loc[0, "cutoff_day"] = 366
    with pytest.raises(ValueError):
        verify_shard(rows, targets, groups, 365)


def test_no_test_loader():
    with pytest.raises(ValueError, match="test loading is disabled"):
        load_split(None, {}, "test")


def test_fit_save_reload_with_exact_schema(tmp_path):
    rng = np.random.default_rng(42)
    def sample(n_groups):
        X = rng.normal(size=(n_groups*10, 25)).astype(np.float32)
        y = np.tile([0]*5 + [1]*5, n_groups).astype(np.int8)
        X[:, 0] = y
        groups = pd.DataFrame({"group_size": [10]*n_groups, "target_count": [7]*n_groups})
        ids = np.column_stack([np.repeat(np.arange(n_groups), 10), np.zeros(n_groups*10), np.tile(np.arange(10), n_groups)]).astype(np.int64)
        return RankingData(X, y, groups, ids, np.zeros(len(y), dtype=bool), "synthetic")
    train, valid = sample(20), sample(8)
    config = {"num_boost_round": 10, "early_stopping_rounds": 3, "early_stopping_min_delta": 1e-5,
              "params": {"objective": "lambdarank", "metric": "None", "num_threads": 2,
                         "num_leaves": 7, "min_data_in_leaf": 2, "verbosity": -1, "seed": 42,
                         "deterministic": True, "force_col_wise": True}}
    model, history, timing = fit_ranker(train, valid, config, tmp_path / "progress.jsonl")
    assert model.feature_name() == FEATURE_COLUMNS and model.best_iteration > 0
    assert max(history["validation"]["ndcg5_e2e_all_groups"]) == pytest.approx(1.0)
    model.save_model(str(tmp_path / "model.txt"))
    restored = lgb.Booster(model_file=str(tmp_path / "model.txt"))
    np.testing.assert_array_equal(model.predict(valid.X), restored.predict(valid.X))
    assert timing["sampled_fit_peak_rss_bytes"] > 0
    assert process_memory()["peak_rss_bytes"] > 0
