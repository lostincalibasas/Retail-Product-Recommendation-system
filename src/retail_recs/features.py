"""Vectorized historical aggregates with explicit (cutoff-window, cutoff] windows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import DataConfig, FEATURE_COLUMNS, ID_COLUMNS


def add_features(candidates: pd.DataFrame, history: pd.DataFrame, catalog: pd.DataFrame,
                 cutoff: int, config: DataConfig) -> pd.DataFrame:
    """history is already restricted to positive purchases through cutoff."""
    if (history.DAY > cutoff).any() or (history.QUANTITY <= 0).any():
        raise ValueError("Feature history contains future or nonpositive purchases")
    result = candidates.copy()
    pair_keys = ["household_key", "PRODUCT_ID"]
    hh = history.groupby("household_key", sort=False).DAY.max()
    product = history.groupby("PRODUCT_ID", sort=False).DAY.max()
    pair = history.groupby(pair_keys, sort=False).DAY.max()
    result["hh_recency_days"] = cutoff - result.household_key.map(hh)
    result["product_recency_days"] = cutoff - result.PRODUCT_ID.map(product)
    pair_index = pd.MultiIndex.from_frame(result[pair_keys])
    result["pair_recency_days"] = cutoff - pair.reindex(pair_index).fillna(-1).to_numpy()
    for window in (7, 28, 84):
        recent = history.loc[history.DAY > cutoff - window]
        hh_group = recent.groupby("household_key", sort=False)
        product_group = recent.groupby("PRODUCT_ID", sort=False)
        pair_group = recent.groupby(pair_keys, sort=False)
        result[f"hh_purchase_rows_{window}d"] = result.household_key.map(hh_group.size()).fillna(0)
        result[f"product_households_{window}d"] = result.PRODUCT_ID.map(product_group.household_key.nunique()).fillna(0)
        result[f"pair_purchase_rows_{window}d"] = pair_group.size().reindex(pair_index, fill_value=0).to_numpy()
        if window == 28:
            result["hh_spend_28d"] = result.household_key.map(hh_group.SALES_VALUE.sum()).fillna(0)
            result["product_purchase_rows_28d"] = result.PRODUCT_ID.map(product_group.size()).fillna(0)
            prices = product_group.SALES_VALUE.sum() / product_group.QUANTITY.sum()
            result["product_mean_price_28d"] = result.PRODUCT_ID.map(prices).fillna(0)
            hh_prices = hh_group.SALES_VALUE.sum() / hh_group.QUANTITY.sum()
            denominator = result.household_key.map(hh_prices).fillna(0).to_numpy(dtype=float)
            result["price_to_household_mean"] = np.divide(
                result.product_mean_price_28d.to_numpy(dtype=float), denominator,
                out=np.zeros(len(result), dtype=float), where=denominator > 0,
            )
        if window == 84:
            result["hh_products_84d"] = result.household_key.map(hh_group.PRODUCT_ID.nunique()).fillna(0)
            quantities = pair_group.QUANTITY.sum().reindex(pair_index, fill_value=0).to_numpy(dtype=float)
            spending = pair_group.SALES_VALUE.sum().reindex(pair_index, fill_value=0).to_numpy(dtype=float)
            result["pair_quantity_84d"] = quantities
            result["pair_spend_84d"] = spending
            result["pair_mean_price_84d"] = np.divide(spending, quantities,
                out=np.zeros(len(result), dtype=float), where=quantities > 0)
    # Affinity window is configurable for candidate generation but v1 feature
    # names promise 84 days, so feature affinities always use exactly 84 days.
    enriched = history.loc[history.DAY > cutoff - 84].merge(catalog, on="PRODUCT_ID", validate="many_to_one")
    candidate_metadata = result[["household_key", "PRODUCT_ID"]].merge(catalog, on="PRODUCT_ID", validate="many_to_one")
    totals = enriched.groupby("household_key", sort=False).size()
    for column, name in (("COMMODITY_DESC", "category_affinity_84d"), ("BRAND", "brand_affinity_84d")):
        counts = enriched.groupby(["household_key", column], sort=False).size()
        index = pd.MultiIndex.from_frame(candidate_metadata[["household_key", column]])
        numerator = counts.reindex(index, fill_value=0).to_numpy(dtype=float)
        denominator = result.household_key.map(totals).fillna(0).to_numpy(dtype=float)
        result[name] = np.divide(numerator, denominator, out=np.zeros(len(result)), where=denominator > 0)
    result = result[ID_COLUMNS + FEATURE_COLUMNS].astype(
        {**{c: "int64" for c in ID_COLUMNS}, **{c: "float32" for c in FEATURE_COLUMNS}}
    ).sort_values(ID_COLUMNS).reset_index(drop=True)
    if not np.isfinite(result[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Nonfinite historical features")
    return result
