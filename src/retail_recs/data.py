"""Deterministic, history-only candidates and independently generated targets.

COMMODITY_DESC and BRAND are assumed static catalog attributes. BRAND in Complete
Journey is a broad private/national label, not a manufacturer identity. Products
must have a positive-quantity historical purchase to enter any candidate source.
Recent category/global sources favor active products; older repeats remain valid
because this dataset does not contain inventory or delisting information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import DataConfig, FEATURE_COLUMNS, ID_COLUMNS, PRODUCT_COLUMNS, TRANSACTION_COLUMNS
from .features import add_features


def _validate_cutoff(cutoff: int) -> None:
    if isinstance(cutoff, (bool, np.bool_)) or not isinstance(cutoff, (int, np.integer)) or cutoff < 1:
        raise ValueError("cutoff must be a positive integer dataset day")


def _as_integer(values: pd.Series, name: str) -> pd.Series:
    """Reject fractional identifiers rather than silently truncating with astype."""
    if pd.api.types.is_bool_dtype(values.dtype) or values.isna().any():
        raise ValueError(f"{name} must contain finite integer values")
    try:
        if not np.isfinite(values.to_numpy(dtype="float64")).all():
            raise ValueError(f"{name} must contain finite integer values")
        converted = values.astype("int64")
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite integer values") from exc
    if not (values == converted).all():
        raise ValueError(f"{name} must contain integer values without fractions")
    if name == "DAY" and (converted < 1).any():
        raise ValueError("DAY must start at one")
    return converted


def _history(transactions: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    missing = set(TRANSACTION_COLUMNS) - set(transactions.columns)
    if missing:
        raise ValueError(f"Missing transaction columns: {sorted(missing)}")
    _validate_cutoff(cutoff)
    result = transactions.loc[
        (transactions.DAY <= cutoff) & (transactions.QUANTITY > 0), TRANSACTION_COLUMNS
    ].copy()
    if result.isna().any().any():
        raise ValueError("Historical transaction columns must not contain nulls")
    for key in ("household_key", "PRODUCT_ID", "DAY", "BASKET_ID"):
        result[key] = _as_integer(result[key], key)
    for key in ("QUANTITY", "SALES_VALUE"):
        result[key] = result[key].astype("float64")
        if not np.isfinite(result[key]).all():
            raise ValueError(f"Historical {key} must be finite")
    return result


def _catalog(products: pd.DataFrame, observed: np.ndarray) -> pd.DataFrame:
    missing = set(PRODUCT_COLUMNS) - set(products.columns)
    if missing:
        raise ValueError(f"Missing product columns: {sorted(missing)}")
    catalog = products.loc[products.PRODUCT_ID.isin(observed), PRODUCT_COLUMNS].copy()
    if catalog.PRODUCT_ID.duplicated().any():
        raise ValueError("Product metadata must have unique PRODUCT_ID")
    if len(catalog) != len(observed):
        raise ValueError("Product metadata missing historically observed products")
    catalog["PRODUCT_ID"] = _as_integer(catalog.PRODUCT_ID, "PRODUCT_ID")
    for key in ("COMMODITY_DESC", "BRAND"):
        catalog[key] = catalog[key].fillna("__UNKNOWN__").astype(str)
    return catalog


def _popularity(frame: pd.DataFrame) -> list[int]:
    """Descending distinct purchasing households; product ID breaks ties."""
    counts = frame.groupby("PRODUCT_ID", sort=False).household_key.nunique().rename("count")
    return counts.reset_index().sort_values(
        ["count", "PRODUCT_ID"], ascending=[False, True]
    ).PRODUCT_ID.tolist()


def build_snapshot(transactions: pd.DataFrame, products: pd.DataFrame,
                   cutoff: int, config: DataConfig) -> pd.DataFrame:
    """Return one row per eligible household/candidate, without future labels.

    Repeat pool: most recent historical products, then purchase-row frequency.
    Category pool: popular products across top household categories in the
    affinity window; purchase-row counts rank both categories and products.
    Category strings and product IDs break ties. Source pools are capped at their
    quotas before union. Global popularity fills the union, then all-history
    popularity fills shortages. Flags record source-pool membership, allowing
    overlapping source flags. The global flag covers the first max_candidates
    recent-global products and any selected historical fallback products.
    """
    history = _history(transactions, cutoff)
    if history.empty:
        return pd.DataFrame({**{c: pd.Series(dtype="int64") for c in ID_COLUMNS},
                             **{c: pd.Series(dtype="float32") for c in FEATURE_COLUMNS}})
    catalog = _catalog(products, history.PRODUCT_ID.unique())
    recent = history.loc[history.DAY > cutoff - config.affinity_days].merge(
        catalog, on="PRODUCT_ID", validate="many_to_one"
    )
    pairs = history.groupby(["household_key", "PRODUCT_ID"], sort=False).agg(
        last=("DAY", "max"), frequency=("DAY", "size")
    ).reset_index().sort_values(
        ["household_key", "last", "frequency", "PRODUCT_ID"],
        ascending=[True, False, False, True],
    )
    repeats = {int(h): g.PRODUCT_ID.tolist() for h, g in
               pairs.groupby("household_key", sort=False).head(config.repeat_quota).groupby("household_key")}
    category_counts = recent.groupby(["household_key", "COMMODITY_DESC"], sort=False).size().rename("count")
    preferred = category_counts.reset_index().sort_values(
        ["household_key", "count", "COMMODITY_DESC"], ascending=[True, False, True]
    ).groupby("household_key", sort=False).head(config.preferred_categories)
    preferred = {int(h): g.COMMODITY_DESC.tolist() for h, g in preferred.groupby("household_key")}
    product_counts = recent.groupby(["COMMODITY_DESC", "PRODUCT_ID"], sort=False).size().rename("count")
    category_ranked = product_counts.reset_index().sort_values(
        ["count", "PRODUCT_ID"], ascending=[False, True]
    )
    # Each category can contribute at most the entire quota. Pre-truncation keeps
    # per-household merging small without changing the top-K union result.
    category_pools = {cat: list(zip(g.PRODUCT_ID, g["count"])) for cat, g in
                      category_ranked.groupby("COMMODITY_DESC", sort=False).head(config.category_quota).groupby("COMMODITY_DESC")}
    global_recent = _popularity(history.loc[history.DAY > cutoff - config.popularity_days])
    global_pool = set(global_recent[:config.max_candidates])
    historical = _popularity(history)
    rows = []
    for household in sorted(repeats):
        repeat = repeats[household]
        category = sorted(
            [item for cat in preferred.get(household, []) for item in category_pools.get(cat, [])],
            key=lambda x: (-x[1], x[0]),
        )[:config.category_quota]
        category_ids = [int(p) for p, _ in category]
        repeat_set, category_set = set(repeat), set(category_ids)
        chosen = dict.fromkeys(repeat + category_ids)
        for product in global_recent:
            if len(chosen) >= config.max_candidates:
                break
            chosen[product] = None
        fallback = set()
        for product in historical:
            if len(chosen) >= config.max_candidates:
                break
            if product not in chosen:
                chosen[product] = None
                fallback.add(product)
        for product in sorted(chosen):
            rows.append((household, cutoff, product, float(product in repeat_set),
                         float(product in category_set), float(product in global_pool or product in fallback)))
    candidates = pd.DataFrame(rows, columns=ID_COLUMNS + [
        "is_repeat_candidate", "is_category_candidate", "is_popular_candidate"
    ])
    return add_features(candidates, history, catalog, cutoff, config)


def build_targets(transactions: pd.DataFrame, cutoff: int, config: DataConfig,
                  households=None) -> pd.DataFrame:
    """Distinct actual purchases; candidate coverage is evaluated independently."""
    _validate_cutoff(cutoff)
    missing = {"household_key", "PRODUCT_ID", "DAY", "QUANTITY"} - set(transactions.columns)
    if missing:
        raise ValueError(f"Missing target transaction columns: {sorted(missing)}")
    mask = ((transactions.DAY > cutoff) &
            (transactions.DAY <= cutoff + config.horizon_days) & (transactions.QUANTITY > 0))
    if households is not None:
        mask &= transactions.household_key.isin(households)
    result = transactions.loc[mask, ["household_key", "PRODUCT_ID", "DAY", "QUANTITY"]].copy()
    for key in ("household_key", "PRODUCT_ID", "DAY"):
        result[key] = _as_integer(result[key], key)
    if not np.isfinite(result.QUANTITY.to_numpy(dtype="float64")).all():
        raise ValueError("Target quantities must be finite")
    return result[["household_key", "PRODUCT_ID"]].drop_duplicates().sort_values(
        ["household_key", "PRODUCT_ID"]).reset_index(drop=True)
