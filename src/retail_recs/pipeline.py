"""Build immutable train/validation artifacts; held-out test materialization is disabled."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .contracts import DataConfig, FEATURE_COLUMNS, ID_COLUMNS, PRODUCT_COLUMNS, TRANSACTION_COLUMNS, feature_schema
from .data import build_snapshot, build_targets
from .diagnostics import aggregate_summaries, summarize_snapshot


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_sources(root: Path, config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Predicate-filter transactions before pandas; no test-period rows are loaded."""
    last_day = config.validation_last_cutoff + config.horizon_days
    transaction_path = root / "data/processed/transaction_data.parquet"
    product_path = root / "data/processed/product.parquet"
    transactions = pd.read_parquet(transaction_path, columns=TRANSACTION_COLUMNS,
                                  filters=[("DAY", "<=", last_day)])
    products = pd.read_parquet(product_path, columns=PRODUCT_COLUMNS)
    if transactions.empty or transactions.DAY.max() < last_day:
        raise ValueError(f"Source does not cover the final required target day {last_day}")
    if transactions.isna().any().any():
        raise ValueError("Nulls found in required transaction columns")
    if not np.isfinite(transactions.to_numpy(dtype="float64")).all():
        raise ValueError("Transaction values must be finite")
    for column in ("household_key", "PRODUCT_ID", "DAY", "BASKET_ID"):
        if not (transactions[column] == transactions[column].astype("int64")).all():
            raise ValueError(f"Noninteger transaction identifier: {column}")
    if transactions.DAY.min() < 1 or transactions.DAY.max() > last_day:
        raise ValueError("Invalid transaction day range")
    if products.PRODUCT_ID.isna().any() or products.PRODUCT_ID.duplicated().any():
        raise ValueError("Product catalog keys must be unique and non-null")
    if not np.isfinite(products.PRODUCT_ID.to_numpy(dtype="float64")).all() or not (
            products.PRODUCT_ID == products.PRODUCT_ID.astype("int64")).all():
        raise ValueError("Product catalog keys must be finite integers")
    if not transactions.PRODUCT_ID.isin(products.PRODUCT_ID).all():
        raise ValueError("Transaction product key missing from catalog")
    audit = {
        "read_filter": f"DAY <= {last_day}", "loaded_rows": len(transactions),
        "loaded_day_min": int(transactions.DAY.min()), "loaded_day_max": int(transactions.DAY.max()),
        "nonpositive_quantity_rows_excluded_from_history_and_labels": int((transactions.QUANTITY <= 0).sum()),
        "positive_quantity_negative_sales_rows_retained": int(((transactions.QUANTITY > 0) & (transactions.SALES_VALUE < 0)).sum()),
        "catalog_rows": len(products), "test_rows_loaded": 0,
        "sources": {str(p.relative_to(root)).replace("\\", "/"): sha256(p)
                    for p in (transaction_path, product_path)},
    }
    return transactions, products, audit


def label_snapshot(snapshot: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    pair_columns = ["household_key", "PRODUCT_ID"]
    result = snapshot.copy()
    result["relevance"] = pd.MultiIndex.from_frame(result[pair_columns]).isin(
        pd.MultiIndex.from_frame(targets[pair_columns])).astype("int8")
    return result


def snapshot_groups(rows: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Offsets are relative to this snapshot shard; concatenate group sizes with shards."""
    groups = rows.groupby(["household_key", "cutoff_day"], sort=False).relevance.agg(
        group_size="size", positive_labels="sum").reset_index()
    groups["row_offset"] = groups.group_size.cumsum().shift(fill_value=0).astype("int64")
    target_counts = targets.groupby("household_key").size()
    groups["target_count"] = groups.household_key.map(target_counts).fillna(0).astype("int64")
    groups["trainable"] = (groups.positive_labels > 0) & (groups.positive_labels < groups.group_size)
    return groups


def assert_history_invariance(transactions, products, cutoff, config, snapshot) -> None:
    """Independently remove every future row and require exactly the same features."""
    historical_only = transactions.loc[transactions.DAY <= cutoff].copy()
    actual = build_snapshot(historical_only, products, cutoff, config)
    pd.testing.assert_frame_equal(snapshot, actual, check_exact=True)


def prepare_data(root: Path, output: Path, config: DataConfig,
                 limit_cutoffs: int | None = None) -> dict:
    """Build both development splits. Refuse overwrite, even after interrupted runs."""
    root, output = root.resolve(), output.resolve()
    if limit_cutoffs is not None and limit_cutoffs < 1:
        raise ValueError("limit_cutoffs must be positive")
    if output.exists():
        raise FileExistsError(f"Choose a new artifact directory; refusing overwrite: {output}")
    transactions, products, source_audit = load_sources(root, config)
    output.mkdir(parents=True)
    (output / "INCOMPLETE").write_text("Only manifest.json with status=complete marks a usable build.\n", encoding="utf-8")
    start = time.monotonic()
    schema = feature_schema()
    write_json(output / "feature_schema.json", schema)
    write_json(output / "config.json", config.to_dict())
    source_code = {str(p.relative_to(root)).replace("\\", "/"): sha256(p)
                   for p in sorted((root / "src/retail_recs").glob("*.py"))}
    git_result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    manifest = {
        "status": "building", "schema": schema, "config": config.to_dict(),
        "mode": "smoke" if limit_cutoffs else "full_development",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_base_commit": git_result.stdout.strip() if git_result.returncode == 0 else None,
        "git_working_tree_dirty_at_build": bool(git_status.stdout.strip()) if git_status.returncode == 0 else None,
        "git_status_error": git_status.stderr.strip() if git_status.returncode else None,
        "source_code_sha256": source_code, "source_audit": source_audit,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow")},
        "test": {"status": "not_materialized_before_model_selection", "cutoffs": config.cutoffs("test"),
                 "first_target_day": config.test_first_cutoff + 1,
                 "last_target_day": config.test_last_cutoff + config.horizon_days},
        "splits": {}, "invariance_checks": [],
    }
    for split in ("train", "validation"):
        cutoffs = config.cutoffs(split)
        if limit_cutoffs:
            cutoffs = cutoffs[:limit_cutoffs]
        snapshots, shards = [], []
        for cutoff in cutoffs:
            tick = time.monotonic()
            snapshot = build_snapshot(transactions, products, cutoff, config)
            eligible = np.sort(transactions.loc[(transactions.DAY <= cutoff) &
                                                (transactions.QUANTITY > 0), "household_key"].unique())
            if not np.array_equal(np.sort(snapshot.household_key.unique()), eligible):
                raise ValueError("Candidates do not cover every historically eligible household")
            if cutoff in (cutoffs[0], cutoffs[-1]):
                assert_history_invariance(transactions, products, cutoff, config, snapshot)
                manifest["invariance_checks"].append({"split": split, "cutoff": cutoff, "passed": True})
            targets = build_targets(transactions, cutoff, config, households=eligible)
            rows = label_snapshot(snapshot, targets)
            summary = summarize_snapshot(rows, targets, config.max_candidates)
            summary.update({"cutoff": cutoff, "target_start": cutoff + 1,
                            "target_end": cutoff + config.horizon_days})
            groups = snapshot_groups(rows, targets)
            summary["trainable_groups"] = int(groups.trainable.sum())
            future = transactions.loc[(transactions.DAY > cutoff) &
                                      (transactions.DAY <= cutoff + config.horizon_days) &
                                      (transactions.QUANTITY > 0)]
            summary["new_households_excluded_from_known_household_evaluation"] = int(
                future.loc[~future.household_key.isin(eligible), "household_key"].nunique())
            shard = output / split / f"cutoff_{cutoff}"
            shard.mkdir(parents=True)
            targets = targets.assign(cutoff_day=np.int64(cutoff))[ID_COLUMNS]
            files = {}
            for name, frame in (("rows", rows), ("targets", targets), ("groups", groups)):
                path = shard / f"{name}.parquet"
                frame.to_parquet(path, index=False, compression="zstd")
                if pq.ParquetFile(path).metadata.num_rows != len(frame):
                    raise ValueError(f"Parquet row count mismatch: {path}")
                files[name] = {"path": path.relative_to(output).as_posix(),
                               "rows": len(frame), "sha256": sha256(path), "bytes": path.stat().st_size}
            # Verify complete disk roundtrip on boundary shards in each split.
            if cutoff in (cutoffs[0], cutoffs[-1]):
                pd.testing.assert_frame_equal(rows, pd.read_parquet(shard / "rows.parquet"), check_exact=True)
            write_json(shard / "diagnostics.json", summary)
            snapshots.append(summary)
            shards.append({"cutoff": cutoff, "files": files, "diagnostics": summary})
            print(f"{split} cutoff={cutoff}: rows={len(rows):,}, groups={summary['groups']:,}, "
                  f"positives={summary['positive_labels']:,}, candidate_recall={summary['candidate_recall_micro']:.4f}, "
                  f"seconds={time.monotonic()-tick:.1f}", flush=True)
        aggregate = aggregate_summaries(snapshots)
        aggregate["trainable_groups"] = sum(s["trainable_groups"] for s in snapshots)
        aggregate["new_household_window_occurrences_excluded"] = sum(
            s["new_households_excluded_from_known_household_evaluation"] for s in snapshots)
        manifest["splits"][split] = {"cutoffs": cutoffs, "summary": aggregate, "shards": shards}
    manifest["elapsed_seconds"] = round(time.monotonic() - start, 3)
    manifest["status"] = "complete"
    write_json(output / "manifest.json.tmp", manifest)
    os.replace(output / "manifest.json.tmp", output / "manifest.json")
    (output / "INCOMPLETE").unlink()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("artifacts/data/retail-ranking-v1"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--limit-cutoffs", type=int, default=None, help="Smoke mode only: first N cutoffs per development split")
    args = parser.parse_args()
    config = DataConfig(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else DataConfig()
    output = args.output if args.output.is_absolute() else args.project_root / args.output
    result = prepare_data(args.project_root, output, config, args.limit_cutoffs)
    print(f"Complete: {output} ({result['elapsed_seconds']:.1f}s). Held-out test remains sealed.")


if __name__ == "__main__":
    main()
