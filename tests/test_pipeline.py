import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retail_recs.contracts import DataConfig, FEATURE_COLUMNS, ID_COLUMNS, feature_schema
from retail_recs.pipeline import load_sources, prepare_data


def small_config():
    return DataConfig(max_candidates=4, repeat_quota=1, category_quota=1,
                      train_first_cutoff=84, train_last_cutoff=84,
                      validation_first_cutoff=91, validation_last_cutoff=91,
                      test_first_cutoff=98, test_last_cutoff=98)


def write_sources(root: Path):
    path = root / "data/processed"
    path.mkdir(parents=True)
    rows = [(1, 10, 1), (2, 20, 2), (1, 10, 84), (1, 10, 85), (2, 30, 91),
            (1, 20, 92), (2, 30, 98), (999, 999, 99)]
    tx = pd.DataFrame(rows, columns=["household_key", "PRODUCT_ID", "DAY"])
    tx["QUANTITY"] = 1.0
    tx["SALES_VALUE"] = 2.0
    tx["BASKET_ID"] = np.arange(len(tx), dtype="int64")
    tx.to_parquet(path / "transaction_data.parquet", index=False)
    pd.DataFrame({"PRODUCT_ID": [10, 20, 30, 999], "COMMODITY_DESC": ["A", "B", "B", "FUTURE"],
                  "BRAND": ["National"] * 4}).to_parquet(path / "product.parquet", index=False)


def test_default_exact_boundaries():
    cfg = DataConfig()
    assert len(cfg.cutoffs("train")) == 25
    assert len(cfg.cutoffs("validation")) == len(cfg.cutoffs("test")) == 12
    assert (cfg.train_last_cutoff + 7, cfg.validation_first_cutoff + 1) == (540, 541)
    assert (cfg.validation_last_cutoff + 7, cfg.test_first_cutoff + 1) == (624, 625)
    assert cfg.test_last_cutoff + 7 == 708
    assert set(ID_COLUMNS + ["relevance"]).isdisjoint(FEATURE_COLUMNS)


def test_checked_in_contracts_and_portable_dependency_lock():
    root = Path(__file__).resolve().parents[1]
    assert json.loads((root / "configs/data_v1.json").read_text()) == DataConfig().to_dict()
    assert json.loads((root / "configs/feature_schema_v1.json").read_text()) == feature_schema()
    lock = (root / "requirements-phase2.lock.txt").read_text()
    assert "file:" not in lock
    assert all("==" in line for line in lock.splitlines() if line and not line.startswith("#"))


def test_overlap_and_bad_grids_rejected():
    with pytest.raises(ValueError):
        DataConfig(validation_first_cutoff=533)
    with pytest.raises(ValueError):
        DataConfig(train_last_cutoff=532)
    with pytest.raises(ValueError):
        DataConfig(seed=True)
    with pytest.raises(ValueError):
        DataConfig(max_candidates=200.5)


def test_build_roundtrip_and_test_period_is_sealed(tmp_path):
    write_sources(tmp_path)
    cfg = small_config()
    tx, _, audit = load_sources(tmp_path, cfg)
    assert audit["loaded_day_max"] == 98
    assert 999 not in tx.household_key.values
    output = tmp_path / "artifacts/run"
    manifest = prepare_data(tmp_path, output, cfg)
    assert manifest["status"] == "complete"
    assert json.loads((output / "feature_schema.json").read_text())["feature_order"] == FEATURE_COLUMNS
    assert not (output / "INCOMPLETE").exists()
    assert not (output / "test").exists()
    assert manifest["test"]["first_target_day"] == 99
    assert len(manifest["invariance_checks"]) == 2
    assert all(c["passed"] for c in manifest["invariance_checks"])
    train = pd.read_parquet(output / "train/cutoff_84/rows.parquet")
    targets = pd.read_parquet(output / "train/cutoff_84/targets.parquet")
    groups = pd.read_parquet(output / "train/cutoff_84/groups.parquet")
    assert set(train.PRODUCT_ID) == {10, 20}
    assert set(targets.PRODUCT_ID) == {10, 30}  # future-only product remains a retrieval miss
    assert train.relevance.sum() == 1
    assert groups.row_offset.tolist() == [0, 2]
    assert groups.trainable.tolist() == [True, False]
    assert groups.group_size.sum() == len(train)
    assert all(train[c].dtype == np.dtype("float32") for c in FEATURE_COLUMNS)
    with pytest.raises(FileExistsError):
        prepare_data(tmp_path, output, cfg)


def test_incomplete_source_fails_before_writing(tmp_path):
    write_sources(tmp_path)
    cfg = DataConfig(max_candidates=4, repeat_quota=1, category_quota=1,
                     train_first_cutoff=84, train_last_cutoff=84,
                     validation_first_cutoff=98, validation_last_cutoff=98,
                     test_first_cutoff=105, test_last_cutoff=105)
    with pytest.raises(ValueError, match="final required target day"):
        prepare_data(tmp_path, tmp_path / "artifacts/run", cfg)
    assert not (tmp_path / "artifacts/run").exists()
