"""Shared interfaces for data preparation, ranking, and future inference.

DAY is a relative dataset index. A snapshot at c sees DAY <= c and predicts
positive-quantity purchases on c + 1 through c + horizon, inclusive.
Identifiers and relevance are NEVER model features.
"""
from dataclasses import asdict, dataclass
from numbers import Integral

SCHEMA_VERSION = "retail-ranking-v1"
ID_COLUMNS = ["household_key", "cutoff_day", "PRODUCT_ID"]
FEATURE_COLUMNS = [
    "hh_purchase_rows_7d", "hh_purchase_rows_28d", "hh_purchase_rows_84d",
    "hh_spend_28d", "hh_products_84d", "hh_recency_days",
    "product_households_7d", "product_households_28d", "product_households_84d",
    "product_purchase_rows_28d", "product_mean_price_28d", "product_recency_days",
    "pair_purchase_rows_7d", "pair_purchase_rows_28d", "pair_purchase_rows_84d",
    "pair_quantity_84d", "pair_spend_84d", "pair_recency_days",
    "pair_mean_price_84d", "category_affinity_84d", "brand_affinity_84d",
    "price_to_household_mean", "is_repeat_candidate", "is_category_candidate",
    "is_popular_candidate",
]
TRANSACTION_COLUMNS = [
    "household_key", "PRODUCT_ID", "DAY", "QUANTITY", "SALES_VALUE", "BASKET_ID",
]
PRODUCT_COLUMNS = ["PRODUCT_ID", "COMMODITY_DESC", "BRAND"]


@dataclass(frozen=True)
class DataConfig:
    seed: int = 42
    horizon_days: int = 7
    max_candidates: int = 200
    repeat_quota: int = 80
    category_quota: int = 60
    preferred_categories: int = 3
    popularity_days: int = 28
    affinity_days: int = 84
    train_first_cutoff: int = 365
    train_last_cutoff: int = 533
    validation_first_cutoff: int = 540
    validation_last_cutoff: int = 617
    test_first_cutoff: int = 624
    test_last_cutoff: int = 701

    def __post_init__(self):
        if any(not isinstance(value, Integral) or isinstance(value, bool)
               for value in asdict(self).values()):
            raise ValueError("All configuration values must be integers, not booleans")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.horizon_days != 7:
            raise ValueError("v1 requires seven-day targets")
        if not 0 <= self.repeat_quota + self.category_quota < self.max_candidates:
            raise ValueError("Candidate quotas must reserve global-popularity slots")
        if min(self.repeat_quota, self.category_quota, self.preferred_categories,
               self.popularity_days, self.affinity_days) <= 0:
            raise ValueError("Quotas and historical windows must be positive")
        for split in ("train", "validation", "test"):
            first = getattr(self, f"{split}_first_cutoff")
            last = getattr(self, f"{split}_last_cutoff")
            if first < self.affinity_days or last < first or (last - first) % 7:
                raise ValueError(f"Invalid weekly cutoff grid: {split}")
        if self.train_last_cutoff + 7 > self.validation_first_cutoff:
            raise ValueError("Training labels overlap validation future")
        if self.validation_last_cutoff + 7 > self.test_first_cutoff:
            raise ValueError("Validation labels overlap test future")

    def cutoffs(self, split: str) -> list[int]:
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Unknown split: {split}")
        return list(range(getattr(self, f"{split}_first_cutoff"),
                          getattr(self, f"{split}_last_cutoff") + 1, 7))

    def to_dict(self) -> dict:
        return asdict(self)


def feature_schema() -> dict:
    return {"version": SCHEMA_VERSION, "identifiers": ID_COLUMNS,
            "feature_order": FEATURE_COLUMNS,
            "features": {name: "float32" for name in FEATURE_COLUMNS},
            "label": {"relevance": "int8"},
            "group": ["household_key", "cutoff_day"],
            "history": "DAY <= cutoff_day; positive QUANTITY only",
            "target": "cutoff_day < DAY <= cutoff_day + 7; positive QUANTITY only"}
