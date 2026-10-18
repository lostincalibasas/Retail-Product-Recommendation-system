import numpy as np
import pandas as pd
import pytest

from retail_recs.model_diagnostics import (
    describe, exposure, paired_score_gaps, rank_positions, tree_gap, within_group_ordering,
)


def test_new_only_ranks_and_exposure_keep_original_top5_positions():
    scores = np.array([10, 9, 8, 7, 6, 5, 1, 1, 1, 1, 1, 1])
    ids = np.r_[np.arange(6), [6, 5, 4, 3, 2, 1]]
    new = np.array([False]*5+[True]+[True]*6)
    rank, new_rank, top = rank_positions(scores, np.array([6, 6]), ids, new)
    assert rank.tolist() == [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1]
    assert new_rank.tolist() == [0, 0, 0, 0, 0, 1, 6, 5, 4, 3, 2, 1]
    labels = np.array([0]*5+[1]+[0]*5+[1])
    groups = pd.DataFrame({"target_count": [1, 1], "new_target_count": [1, 1]})
    result = exposure(top, new, labels, groups)
    assert result["new_slots"] == 5 and result["new_hits"] == 1
    assert result["groups_with_new"] == 1 and result["new_slot_share"] == .5
    assert result["new_count_histogram"]["0"] == result["new_count_histogram"]["5"] == 1


def test_first_divergence_includes_downstream_interactions_and_signed_gaps():
    leaf = lambda x: {"leaf_value": x}
    child = {"split_feature": 1, "threshold": 2, "decision_type": "<=",
             "left_child": leaf(-2), "right_child": leaf(3)}
    tree = {"split_feature": 0, "threshold": 1, "decision_type": "<=",
            "left_child": child, "right_child": leaf(1)}
    names = ["pair_frequency", "affinity"]
    assert tree_gap(tree, [0, 0], [0, 3], names) == ("affinity", 5)
    assert tree_gap(tree, [0, 3], [2, 0], names) == ("pair_frequency", -2)
    assert tree_gap(tree, [0, 0], [0, 1], names) == (None, 0)
    gaps, counts = paired_score_gaps([{"tree_structure": tree}]*2, [0, 3], [2, 0], names)
    assert gaps == {"pair_frequency": -4} and counts == {"pair_frequency": 2}
    with pytest.raises(ValueError):
        tree_gap({**tree, "decision_type": "=="}, [0, 0], [2, 0], names)


def test_pairwise_ordering_counts_ties_and_handles_single_segment_groups():
    scores = np.array([3, 1, 2, 1, 0, 8, 7, 6, 5, 4])
    new = np.array([False, False, True, True, True]+[True]*5)
    result = within_group_ordering(scores, new, [5, 5])
    assert result["repeat_new_pairs"] == 6
    assert result["repeat_strictly_above_new"] == 4
    assert result["tied_pairs"] == 1
    assert result["prob_repeat_above_new_half_ties"] == .75
    assert result["mean_repeat_minus_mean_new"]["count"] == 1
    assert describe([]) == {"count": 0}
    with pytest.raises(ValueError):
        describe([np.nan])
