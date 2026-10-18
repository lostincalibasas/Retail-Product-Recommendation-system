"""Deterministic top-five evaluation including retrieval misses and empty groups."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import FEATURE_COLUMNS


def recent_popularity_scores(features, feature_order):
    if list(feature_order) != FEATURE_COLUMNS or features.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("Baseline input feature order differs from the frozen schema")
    return features[:, FEATURE_COLUMNS.index("product_households_28d")].copy()


def top_indices(scores, group_sizes, product_ids, k=5):
    scores, sizes, ids = np.asarray(scores), np.asarray(group_sizes), np.asarray(product_ids)
    if k < 1 or sizes.ndim != 1 or not np.issubdtype(sizes.dtype, np.integer) or (sizes <= 0).any():
        raise ValueError("Positive integer group sizes and k required")
    if len(scores) != int(sizes.sum()) or len(ids) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Scores/IDs must align with groups and scores must be finite")
    result = np.full((len(sizes), k), -1, dtype=np.int64)
    if len(sizes) == 0:
        return result
    if np.all(sizes == sizes[0]):
        width = int(sizes[0]); take = min(k, width)
        order = np.lexsort((ids.reshape(-1, width), -scores.reshape(-1, width)), axis=1)[:, :take]
        result[:, :take] = order + np.arange(len(sizes))[:, None] * width
    else:
        start = 0
        for i, size in enumerate(sizes):
            take = min(k, int(size))
            order = np.lexsort((ids[start:start+size], -scores[start:start+size]))[:take]
            result[i, :take] = start + order
            start += size
    return result


def ideal_dcg(counts, k=5):
    cumulative = np.r_[0.0, np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))]
    return cumulative[np.minimum(np.asarray(counts, dtype=np.int64), k)]


def safe_ratio(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros(np.broadcast_shapes(a.shape, b.shape)), where=b > 0)


def group_metrics(top, labels, group_sizes, target_counts, candidate_mask=None, k=5):
    """Segment hits retain their positions in the full recommendation list."""
    labels = np.asarray(labels)
    sizes = np.asarray(group_sizes, dtype=np.int64)
    targets = np.asarray(target_counts, dtype=np.int64)
    if len(labels) != sizes.sum() or len(targets) != len(sizes) or (targets < 0).any():
        raise ValueError("Invalid label, group or target-count alignment")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Binary relevance required")
    if candidate_mask is not None:
        labels = labels * np.asarray(candidate_mask, dtype=bool)
    offsets = np.r_[0, np.cumsum(sizes)[:-1]]
    positives = np.add.reduceat(labels.astype(np.int64), offsets) if len(sizes) else np.array([], dtype=np.int64)
    if (positives > targets).any():
        raise ValueError("Candidate positives exceed independent target counts")
    valid = top >= 0
    top_labels = labels[np.maximum(top, 0)] * valid if len(labels) else np.zeros(top.shape)
    hits = top_labels.sum(axis=1).astype(np.int64)
    dcg = (top_labels / np.log2(np.arange(2, k + 2))).sum(axis=1)
    return pd.DataFrame({"target_count": targets, "candidate_positives": positives, "hits_at_5": hits,
                         "ndcg5_e2e": safe_ratio(dcg, ideal_dcg(targets, k)),
                         "ndcg5_candidate": safe_ratio(dcg, ideal_dcg(positives, k)),
                         "recall5": safe_ratio(hits, targets)})


def summarize_metrics(frame):
    has_target = frame.target_count > 0
    has_positive = frame.candidate_positives > 0
    total_targets, retrieved, hits = (int(frame[c].sum()) for c in ("target_count", "candidate_positives", "hits_at_5"))
    def mean(col, mask=None):
        values = frame.loc[mask, col] if mask is not None else frame[col]
        return float(values.mean()) if len(values) else 0.0
    return {"groups": len(frame), "purchase_groups": int(has_target.sum()),
            "no_purchase_groups": int((~has_target).sum()), "zero_positive_groups": int((~has_positive).sum()),
            "candidate_miss_groups": int((has_target & ~has_positive).sum()),
            "targets": total_targets, "retrieved_targets": retrieved, "hits_at_5": hits,
            "ndcg5_e2e_all_groups": mean("ndcg5_e2e"),
            "ndcg5_e2e_purchase_groups": mean("ndcg5_e2e", has_target),
            "ndcg5_candidate_all_groups": mean("ndcg5_candidate"),
            "ndcg5_candidate_positive_groups": mean("ndcg5_candidate", has_positive),
            "recall5_micro": hits / total_targets if total_targets else 0.0,
            "recall5_macro_all_groups": mean("recall5"),
            "recall5_macro_purchase_groups": mean("recall5", has_target),
            "candidate_recall_micro": retrieved / total_targets if total_targets else 0.0,
            "candidate_recall_macro_purchase_groups": float(safe_ratio(
                frame.loc[has_target, "candidate_positives"], frame.loc[has_target, "target_count"]).mean()) if has_target.any() else 0.0}


def evaluate(scores, labels, group_sizes, product_ids, groups, repeat_candidates):
    top = top_indices(scores, group_sizes, product_ids)
    overall = group_metrics(top, labels, group_sizes, groups.target_count)
    repeat = group_metrics(top, labels, group_sizes, groups.repeat_target_count, repeat_candidates)
    new = group_metrics(top, labels, group_sizes, groups.new_target_count, ~np.asarray(repeat_candidates))
    return {"overall": summarize_metrics(overall), "repeat": summarize_metrics(repeat),
            "new": summarize_metrics(new)}, overall, top


def make_validation_metric(labels, group_sizes, product_ids, target_counts):
    """Primary early-stopping metric: all groups, true all-target ideal DCG."""
    labels = np.asarray(labels)
    denominator = ideal_dcg(target_counts)
    weights = 1.0 / np.log2(np.arange(2, 7))
    def metric(predictions, dataset):
        top = top_indices(predictions, group_sizes, product_ids)
        dcg = ((labels[np.maximum(top, 0)] * (top >= 0)) * weights).sum(axis=1)
        return "ndcg5_e2e_all_groups", float(safe_ratio(dcg, denominator).mean()), True
    return metric
