"""Synthetic behavioral tests; no real held-out transactions are loaded."""
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from retail_recs.contracts import DataConfig, FEATURE_COLUMNS, ID_COLUMNS
from retail_recs.data import build_snapshot, build_targets


def transactions(rows):
    frame = pd.DataFrame(rows, columns=["household_key", "PRODUCT_ID", "DAY", "QUANTITY", "SALES_VALUE"])
    frame["BASKET_ID"] = np.arange(len(frame))
    return frame


def products(ids):
    return pd.DataFrame({"PRODUCT_ID": ids, "COMMODITY_DESC": ["food"] * len(ids), "BRAND": ["National"] * len(ids)})


class DataTests(unittest.TestCase):
    def setUp(self):
        self.config = DataConfig(max_candidates=5, repeat_quota=1, category_quota=1)
        self.tx = transactions([
            (1, 10, 16, 1, 2), (1, 10, 17, 2, 6), (1, 20, 72, 1, 3),
            (1, 20, 73, 1, 4), (1, 30, 93, 1, 5), (1, 30, 94, 1, 6),
            (1, 40, 100, 2, 8), (2, 50, 99, 1, 2),
            (1, 60, 101, 1, 3), (1, 60, 107, 1, 3), (1, 70, 108, 1, 3),
            (3, 80, 105, 1, 3), (1, 90, 100, 0, 3), (4, 90, 99, -1, -3),
        ])
        self.catalog = products(list(range(10, 100, 10)))

    def test_exact_windows_prices_schema_and_no_future_candidates(self):
        snapshot = build_snapshot(self.tx, self.catalog, 100, self.config)
        self.assertEqual(snapshot.columns.tolist(), ID_COLUMNS + FEATURE_COLUMNS)
        self.assertTrue(all(snapshot[c].dtype == np.dtype("float32") for c in FEATURE_COLUMNS))
        self.assertTrue(all(snapshot[c].dtype == np.dtype("int64") for c in ID_COLUMNS))
        self.assertEqual(set(snapshot.household_key), {1, 2})
        self.assertEqual(set(snapshot.PRODUCT_ID), {10, 20, 30, 40, 50})
        self.assertTrue(np.isfinite(snapshot[FEATURE_COLUMNS]).all().all())
        row = snapshot[(snapshot.household_key == 1) & (snapshot.PRODUCT_ID == 40)].iloc[0]
        self.assertEqual(row.hh_purchase_rows_7d, 2)  # excludes day93
        self.assertEqual(row.hh_purchase_rows_28d, 4)  # excludes day72
        self.assertEqual(row.hh_purchase_rows_84d, 6)  # excludes day16
        self.assertEqual(row.pair_mean_price_84d, 4)
        self.assertEqual(row.pair_recency_days, 0)
        unseen = snapshot[(snapshot.household_key == 1) & (snapshot.PRODUCT_ID == 50)].iloc[0]
        self.assertEqual(unseen.pair_recency_days, 101)
        self.assertEqual(unseen.pair_purchase_rows_84d, 0)

    def test_future_perturbations_and_input_order_do_not_change_snapshot(self):
        expected = build_snapshot(self.tx, self.catalog, 100, self.config)
        changed = self.tx.copy()
        changed.loc[changed.DAY > 100, "SALES_VALUE"] = 999999
        changed = pd.concat([changed, transactions([(99, 9999, 101, 1000, 1)])], ignore_index=True)
        catalog = pd.concat([self.catalog, products([9999])], ignore_index=True)
        actual = build_snapshot(changed.sample(frac=1, random_state=7), catalog, 100, self.config)
        assert_frame_equal(expected, actual)

    def test_targets_exact_window_unique_and_household_eligibility(self):
        result = build_targets(self.tx, 100, self.config, households=[1, 2])
        self.assertEqual(result.to_dict("records"), [{"household_key": 1, "PRODUCT_ID": 60}])
        snapshot = build_snapshot(self.tx, self.catalog, 100, self.config)
        self.assertNotIn(60, snapshot.PRODUCT_ID.tolist())
        self.assertEqual(build_targets(self.tx, 100, self.config).shape[0], 2)

    def test_nonpositive_targets_excluded(self):
        tx = pd.concat([self.tx, transactions([(1, 10, 102, -1, -2), (1, 20, 103, 0, 0)])])
        assert_frame_equal(build_targets(self.tx, 100, self.config), build_targets(tx, 100, self.config))

    def test_group_cap_fallback_and_repeat_order(self):
        tx = transactions([(1, p, 1, 1, 1) for p in range(1, 20)] + [(1, 19, 100, 1, 1)])
        result = build_snapshot(tx, products(list(range(1, 20))), 100, self.config)
        self.assertEqual(len(result), 5)
        self.assertEqual(result.PRODUCT_ID.tolist(), [1, 2, 3, 4, 19])
        self.assertEqual(result.loc[result.is_repeat_candidate == 1, "PRODUCT_ID"].tolist(), [19])
        self.assertEqual(result.is_popular_candidate.sum(), 5)

    def test_popularity_uses_unique_households_and_repeats_use_recency(self):
        tx = transactions(
            [(1, 10, 100, 1, 2), (1, 20, 99, 1, 3)] +
            [(1, 20, 98, 1, 3)] * 20 +
            [(h, 30, 100, 1, 2) for h in (2, 3, 4)] +
            [(h, 40, 100, 1, 2) for h in (5, 6)] +
            [(7, 50, 100, 1, 2), (8, 60, 100, 1, 2)]
        )
        config = DataConfig(max_candidates=4, repeat_quota=1, category_quota=1)
        result = build_snapshot(tx, products([10, 20, 30, 40, 50, 60]), 100, config)
        group = result.loc[result.household_key == 1]
        self.assertEqual(group.PRODUCT_ID.tolist(), [10, 20, 30, 40])
        self.assertEqual(group.loc[group.is_repeat_candidate == 1, "PRODUCT_ID"].tolist(), [10])
        self.assertEqual(group.loc[group.is_category_candidate == 1, "PRODUCT_ID"].tolist(), [20])

    def test_affinity_fractions_and_weighted_prices(self):
        tx = transactions([(1, 10, 99, 2, 6), (1, 10, 100, 1, 6),
                           (1, 20, 100, 1, 8), (2, 30, 100, 1, 2)])
        catalog = pd.DataFrame({"PRODUCT_ID": [10, 20, 30],
            "COMMODITY_DESC": ["fruit", "dairy", "fruit"],
            "BRAND": ["National", "Private", "National"]})
        result = build_snapshot(tx, catalog, 100, self.config)
        row = result.loc[(result.household_key == 1) & (result.PRODUCT_ID == 10)].iloc[0]
        self.assertAlmostEqual(row.category_affinity_84d, 2 / 3, places=6)
        self.assertAlmostEqual(row.brand_affinity_84d, 2 / 3, places=6)
        self.assertEqual(row.pair_mean_price_84d, 4)
        self.assertEqual(row.product_mean_price_28d, 4)
        self.assertAlmostEqual(row.price_to_household_mean, 4 / 5, places=6)

    def test_empty_history_and_inactive_household(self):
        empty = build_snapshot(self.tx, self.catalog, 1, self.config)
        self.assertEqual(empty.shape, (0, len(ID_COLUMNS + FEATURE_COLUMNS)))
        tx = transactions([(1, 10, 1, 1, 1), (2, 20, 100, 1, 1)])
        result = build_snapshot(tx, products([10, 20]), 100, self.config)
        self.assertEqual(result.loc[result.household_key == 1, "hh_purchase_rows_84d"].sum(), 0)
        self.assertTrue(np.isfinite(result[FEATURE_COLUMNS]).all().all())

    def test_duplicate_and_missing_catalog_rejected(self):
        with self.assertRaises(ValueError):
            build_snapshot(self.tx, products([10]), 100, self.config)
        with self.assertRaises(ValueError):
            build_snapshot(self.tx, pd.concat([self.catalog, self.catalog]), 100, self.config)

    def test_fractional_history_identifiers_and_days_rejected(self):
        for column in ("household_key", "PRODUCT_ID", "DAY", "BASKET_ID"):
            tx = self.tx.astype({column: "float64"})
            tx.loc[0, column] += 0.5
            with self.subTest(column=column), self.assertRaises(ValueError):
                build_snapshot(tx, self.catalog, 100, self.config)

    def test_fractional_target_identifiers_and_days_rejected(self):
        for column in ("household_key", "PRODUCT_ID", "DAY"):
            tx = self.tx.astype({column: "float64"})
            tx.loc[8, column] += 0.5
            with self.subTest(column=column), self.assertRaises(ValueError):
                build_targets(tx, 100, self.config)

    def test_invalid_cutoff_rejected_by_both_apis(self):
        for cutoff in (True, 0, -1, 100.5):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                build_snapshot(self.tx, self.catalog, cutoff, self.config)
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                build_targets(self.tx, cutoff, self.config)


if __name__ == "__main__":
    unittest.main()
