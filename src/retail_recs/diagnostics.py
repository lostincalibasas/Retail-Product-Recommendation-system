"""Validate snapshot contracts and measure candidate coverage before ranking.

All rates with a zero denominator are reported as 0.0, accompanied by their
denominator counts. Every historically eligible household must have candidates;
targets containing households absent from candidates therefore raise an error.
Households with neither rows nor targets cannot be discovered by this interface;
the snapshot builder must independently check historical household coverage.
"""
from bisect import bisect_right
from collections import Counter
from math import fsum

import numpy as np
import pandas as pd

from .contracts import FEATURE_COLUMNS, ID_COLUMNS

_COUNT_FIELDS = (
    "candidate_rows", "groups", "positive_labels", "zero_positive_groups",
    "groups_with_target_purchases", "candidate_miss_groups", "target_products",
    "retrieved_target_products", "completely_missed_target_products",
    "no_purchase_groups",
)


def _group_size_stats(histogram: dict[str, int]) -> dict:
    sizes = sorted((int(size), count) for size, count in histogram.items())
    n = sum(count for _, count in sizes)
    if not n:
        return {"min": 0, "max": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    cumulative = np.cumsum([count for _, count in sizes]).tolist()

    def quantile(q: float) -> float:
        rank = (n - 1) * q
        lower, upper = int(np.floor(rank)), int(np.ceil(rank))
        a = sizes[bisect_right(cumulative, lower)][0]
        b = sizes[bisect_right(cumulative, upper)][0]
        return float(a + (b - a) * (rank - lower))

    return {"min": sizes[0][0], "max": sizes[-1][0],
            "mean": sum(size * count for size, count in sizes) / n,
            "p50": quantile(0.5), "p95": quantile(0.95)}


def _finish_summary(counts: dict, histogram: dict[str, int], macro_sum: float,
                    snapshots: int) -> dict:
    target_groups = counts["groups_with_target_purchases"]
    targets = counts["target_products"]
    return {
        **counts, "snapshots": snapshots,
        "group_size": _group_size_stats(histogram),
        "group_size_histogram": histogram,
        "candidate_miss_rate": counts["candidate_miss_groups"] / target_groups
        if target_groups else 0.0,
        "candidate_recall_micro": counts["retrieved_target_products"] / targets
        if targets else 0.0,
        "candidate_recall_macro": macro_sum / target_groups if target_groups else 0.0,
        "candidate_recall_macro_sum": macro_sum,
    }


def summarize_snapshot(rows: pd.DataFrame, targets: pd.DataFrame,
                       max_candidates: int = 200) -> dict:
    """Check a single cutoff and report coverage over unique household/SKU targets.

    Rows must be sorted by the shared ID_COLUMNS order, with float32 features and
    int8 binary relevance. Targets require household_key and PRODUCT_ID columns.
    Candidate misses use only groups with actual target purchases as denominator.
    Counts of target products refer to household/product pairs, not catalog SKUs.
    """
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    required = [*ID_COLUMNS, *FEATURE_COLUMNS, "relevance"]
    missing = set(required) - set(rows.columns)
    if missing:
        raise ValueError(f"Snapshot missing required columns: {sorted(missing)}")
    target_ids = ["household_key", "PRODUCT_ID"]
    missing = set(target_ids) - set(targets.columns)
    if missing:
        raise ValueError(f"Targets missing required columns: {sorted(missing)}")
    if rows.columns.duplicated().any() or targets.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed")
    if rows[ID_COLUMNS].isna().any().any() or targets[target_ids].isna().any().any():
        raise ValueError("Identifiers must not contain missing values")
    if rows["cutoff_day"].nunique() > 1:
        raise ValueError("summarize_snapshot requires a single cutoff_day")
    if rows.duplicated(ID_COLUMNS).any():
        raise ValueError("Duplicate candidate identifiers in snapshot")
    if targets.duplicated(target_ids).any():
        raise ValueError("Targets must contain unique household/product pairs")
    row_index = pd.MultiIndex.from_frame(rows[ID_COLUMNS])
    if not row_index.is_monotonic_increasing:
        raise ValueError("Candidate rows must be sorted by household_key, cutoff_day, PRODUCT_ID; groups must be contiguous")
    if rows["relevance"].dtype != np.dtype("int8"):
        raise ValueError("relevance must have dtype int8")
    if not rows["relevance"].isin([0, 1]).all():
        raise ValueError("relevance must be binary (0 or 1)")
    wrong_dtypes = [col for col in FEATURE_COLUMNS if rows[col].dtype != np.dtype("float32")]
    if wrong_dtypes:
        raise ValueError(f"Features must have dtype float32: {wrong_dtypes}")
    if not np.isfinite(rows[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Features must contain only finite values")
    if not targets["household_key"].isin(rows["household_key"]).all():
        raise ValueError("Target households without candidates: snapshot must cover every eligible household")
    if "cutoff_day" in targets.columns and len(targets):
        if targets["cutoff_day"].isna().any() or not targets["cutoff_day"].eq(rows["cutoff_day"].iloc[0]).all():
            raise ValueError("Target cutoff_day must match snapshot cutoff_day")
    candidate_pairs = pd.MultiIndex.from_frame(rows[target_ids])
    target_pairs = pd.MultiIndex.from_frame(targets[target_ids])
    expected = candidate_pairs.isin(target_pairs).astype(np.int8)
    if not np.array_equal(expected, rows["relevance"].to_numpy()):
        raise ValueError("Candidate relevance labels do not match target purchases")
    grouped = rows.groupby("household_key", sort=False)["relevance"].agg(["size", "sum"])
    if len(grouped) and grouped["size"].max() > max_candidates:
        raise ValueError(f"Ranking group exceeds max_candidates={max_candidates}")
    target_counts = targets.groupby("household_key").size().reindex(grouped.index, fill_value=0)
    has_targets = target_counts > 0
    retrieved = grouped["sum"]
    histogram = {str(int(size)): int(count) for size, count in grouped["size"].value_counts().sort_index().items()}
    counts = {
        "candidate_rows": len(rows), "groups": len(grouped),
        "positive_labels": int(retrieved.sum()),
        "zero_positive_groups": int((retrieved == 0).sum()),
        "groups_with_target_purchases": int(has_targets.sum()),
        "candidate_miss_groups": int((has_targets & (retrieved == 0)).sum()),
        "target_products": len(targets),
        "retrieved_target_products": int(retrieved.sum()),
        "completely_missed_target_products": int(len(targets) - retrieved.sum()),
        "no_purchase_groups": int((~has_targets).sum()),
    }
    macro_sum = float((retrieved[has_targets] / target_counts[has_targets]).sum())
    return _finish_summary(counts, histogram, macro_sum, snapshots=1)


def aggregate_summaries(summaries: list[dict]) -> dict:
    """Aggregate disjoint snapshots; compute exact quantiles from size histograms.

    Household/cutoff groups remain separate observations. Macro recall weights
    every group with target purchases equally, rather than every snapshot equally.
    """
    counts = {name: sum(summary[name] for summary in summaries) for name in _COUNT_FIELDS}
    histogram = Counter()
    for summary in summaries:
        histogram.update(summary["group_size_histogram"])
    histogram = dict(sorted(histogram.items(), key=lambda pair: int(pair[0])))
    macro_sum = fsum(summary["candidate_recall_macro_sum"] for summary in summaries)
    return _finish_summary(counts, histogram, macro_sum,
                           snapshots=sum(summary["snapshots"] for summary in summaries))
