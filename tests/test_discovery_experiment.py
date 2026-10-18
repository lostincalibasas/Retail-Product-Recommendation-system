import numpy as np
import pandas as pd
import pytest

from retail_recs.discovery_experiment import discovery_top5, evaluate_list, verify_recommendations


def test_reserved_slot_keeps_four_even_when_new_item_already_present():
    scores = np.arange(12, 0, -1, dtype=float)
    ids = np.arange(12)
    new = np.array([True, False, False, False, False, True, False, False, False, False, True, True])
    a, b, eligible = discovery_top5(scores, [6, 6], ids, new)
    assert a.tolist() == [[0, 1, 2, 3, 4], [6, 7, 8, 9, 10]]
    assert b.tolist() == [[0, 1, 2, 3, 5], [6, 7, 8, 9, 10]]
    assert eligible.all()
    assert verify_recommendations(a, b, [6, 6], new).tolist() == [True, False]


def test_unavailable_new_falls_back_and_ties_use_product_id():
    new = np.array([True, False, False, False, False]+[False]*5+[True, True])
    a, b, available = discovery_top5(np.r_[np.arange(5, 0, -1), [9, 8, 7, 6, 5, 1, 1]],
                                    [5, 7], [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 20, 10], new)
    assert available.tolist() == [False, True]
    np.testing.assert_array_equal(a[0], b[0])
    assert b[1, 4] == 11  # equal score: smaller product ID
    verify_recommendations(a, b, [5, 7], new)


def test_membership_duplicates_and_first_four_are_enforced():
    a = np.array([[0, 1, 2, 3, 4]])
    for b in (np.array([[0, 1, 2, 3, 6]]), np.array([[0, 1, 2, 3, 3]]), np.array([[1, 0, 2, 3, 5]])):
        with pytest.raises(ValueError):
            verify_recommendations(a, b, [6], np.array([False]*5+[True]))


def test_metrics_account_for_tradeoff_and_keep_empty_groups():
    new = np.tile([False]*5+[True], 3)
    a, b, _ = discovery_top5(np.tile([6, 5, 4, 3, 2, 1], 3), [6]*3, np.tile(np.arange(6), 3), new)
    labels = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]+[0]*6, dtype=np.int8)
    groups = pd.DataFrame({"household_key": [1, 2, 3], "cutoff_day": [540]*3,
                           "target_count": [2, 1, 0], "repeat_target_count": [2, 0, 0], "new_target_count": [0, 1, 0]})
    ma, _ = evaluate_list(a, labels, [6]*3, groups, new)
    mb, _ = evaluate_list(b, labels, [6]*3, groups, new)
    assert ma["precision5_all_groups"] == mb["precision5_all_groups"] == pytest.approx(1/15)
    assert ma["overall"]["recall5_micro"] == mb["overall"]["recall5_micro"] == pytest.approx(1/3)
    assert ma["repeat"]["recall5_micro"] == .5 and mb["repeat"]["recall5_micro"] == 0
    assert ma["new"]["recall5_micro"] == 0 and mb["new"]["recall5_micro"] == 1
    assert ma["overall"]["ndcg5_e2e_all_groups"] == pytest.approx(1/np.log2(6)/(1+1/np.log2(3))/3)
    assert mb["overall"]["ndcg5_e2e_all_groups"] == pytest.approx(1/np.log2(6)/3)
    assert mb["overall"]["no_purchase_groups"] == 1 and mb["new_slot_share"] == .2


def test_policy_selection_does_not_use_future_label_columns():
    frame = pd.DataFrame({"score": [6, 5, 4, 3, 2, 1], "product": np.arange(6),
                          "new": [False]*5+[True], "relevance": [0]*6, "future_quantity": [0]*6})
    def select(f):
        return discovery_top5(f["score"].to_numpy(), [6], f["product"].to_numpy(), f["new"].to_numpy())[1]
    before = select(frame)
    frame["relevance"] = [1]*6
    frame["future_quantity"] = [1000]*6
    np.testing.assert_array_equal(select(frame), before)
