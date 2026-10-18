"""Verify existing artifacts and load only fixed train/validation ranking data."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import DataConfig, FEATURE_COLUMNS, ID_COLUMNS, feature_schema
from .diagnostics import summarize_snapshot
from .pipeline import sha256


@dataclass
class RankingData:
    X: np.ndarray | None
    y: np.ndarray
    groups: pd.DataFrame
    ids: np.ndarray
    repeat_candidates: np.ndarray
    candidate_sha256: str

    @property
    def sizes(self):
        return self.groups.group_size.to_numpy(dtype=np.int32)


def check_manifest(root: Path, directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["mode"] != "full_development" or (directory / "INCOMPLETE").exists():
        raise ValueError("A complete full-development artifact is required")
    if manifest["config"] != DataConfig().to_dict() or manifest["schema"] != feature_schema():
        raise ValueError("Fixed v1 time boundaries and feature schema must match")
    if manifest["source_audit"]["loaded_day_max"] != 624 or manifest["test"]["status"] != "not_materialized_before_model_selection":
        raise ValueError("Held-out test isolation contract mismatch")
    if set(manifest["splits"]) != {"train", "validation"}:
        raise ValueError("Only development splits may be loaded")
    for name, digest in {**manifest["source_code_sha256"], **manifest["source_audit"]["sources"]}.items():
        if sha256(root / name) != digest:
            raise ValueError(f"Source/code hash differs from frozen data: {name}")
    for split in ("train", "validation"):
        expected = DataConfig().cutoffs(split)
        if manifest["splits"][split]["cutoffs"] != expected or [s["cutoff"] for s in manifest["splits"][split]["shards"]] != expected:
            raise ValueError(f"Incorrect cutoff sequence: {split}")
        for shard in manifest["splits"][split]["shards"]:
            for kind, file in shard["files"].items():
                path = (directory / file["path"]).resolve()
                if not path.is_relative_to((directory / split).resolve()):
                    raise ValueError("Shard path escapes its development split")
                if sha256(path) != file["sha256"]:
                    raise ValueError(f"Artifact checksum mismatch: {path}")
    return manifest


def verify_shard(rows, targets, groups, cutoff):
    if list(rows.columns) != ID_COLUMNS + FEATURE_COLUMNS + ["relevance"]:
        raise ValueError("Feature column order must match serialized schema")
    if any(rows[c].dtype != np.dtype("int64") for c in ID_COLUMNS):
        raise ValueError("Identifier dtypes must be int64")
    if not rows.cutoff_day.eq(cutoff).all() or not targets.cutoff_day.eq(cutoff).all():
        raise ValueError("Shard cutoff mismatch")
    summary = summarize_snapshot(rows, targets)
    expected = rows.groupby(["household_key", "cutoff_day"], sort=False).relevance.agg(
        group_size="size", positive_labels="sum").reset_index()
    if not np.array_equal(expected.to_numpy(), groups[expected.columns].to_numpy()):
        raise ValueError("Group IDs, sizes or positive counts do not match rows")
    offsets = np.r_[0, np.cumsum(groups.group_size.to_numpy())[:-1]]
    if not np.array_equal(offsets, groups.row_offset):
        raise ValueError("Group offsets do not match contiguous rows")
    counts = targets.groupby("household_key").size().reindex(groups.household_key, fill_value=0).to_numpy()
    if not np.array_equal(counts, groups.target_count):
        raise ValueError("Group target counts do not match independent labels")
    trainable = (groups.positive_labels > 0) & (groups.positive_labels < groups.group_size)
    if not np.array_equal(trainable, groups.trainable):
        raise ValueError("Incorrect trainable-group flag")
    return summary


def first_purchase_days(root):
    # Only development transactions; never read/evaluate test targets.
    tx = pd.read_parquet(root / "data/processed/transaction_data.parquet",
                         columns=["household_key", "PRODUCT_ID", "DAY", "QUANTITY"], filters=[("DAY", "<=", 624)])
    tx = tx.loc[tx.QUANTITY > 0]
    return tx.groupby(["household_key", "PRODUCT_ID"]).DAY.min()


def load_split(directory, manifest, split, first_days=None):
    if split not in ("train", "validation"):
        raise ValueError("Held-out test loading is disabled")
    shards = manifest["splits"][split]["shards"]
    # Determine allocation using group files; independently verify each against rows below.
    group_frames = [pd.read_parquet(directory / s["files"]["groups"]["path"]) for s in shards]
    total = sum(int(g.loc[g.trainable, "group_size"].sum()) if split == "train" else int(g.group_size.sum()) for g in group_frames)
    X = np.empty((total, len(FEATURE_COLUMNS)), dtype=np.float32)
    y = np.empty(total, dtype=np.int8)
    ids = np.empty((total, 3), dtype=np.int64) if split == "validation" else np.empty((0, 3), dtype=np.int64)
    repeat_candidates = np.empty(total, dtype=bool) if split == "validation" else np.empty(0, dtype=bool)
    kept_groups, digest, position = [], hashlib.sha256(), 0
    for shard, groups in zip(shards, group_frames):
        rows = pd.read_parquet(directory / shard["files"]["rows"]["path"])
        targets = pd.read_parquet(directory / shard["files"]["targets"]["path"])
        if any(len(frame) != shard["files"][kind]["rows"] for kind, frame in
               (("rows", rows), ("groups", groups), ("targets", targets))):
            raise ValueError("Manifest and actual shard row counts differ")
        verify_shard(rows, targets, groups, shard["cutoff"])
        if split == "train":
            rows = rows.loc[np.repeat(groups.trainable.to_numpy(), groups.group_size.to_numpy())]
            groups = groups.loc[groups.trainable].copy()
        else:
            if first_days is None:
                raise ValueError("Historical first-purchase days required for segment evaluation")
            target_first = first_days.reindex(pd.MultiIndex.from_frame(targets[["household_key", "PRODUCT_ID"]]))
            if target_first.isna().any() or (target_first.to_numpy() > shard["cutoff"] + 7).any():
                raise ValueError("Target history is inconsistent")
            is_repeat = target_first.to_numpy() <= shard["cutoff"]
            repeat_counts = targets.loc[is_repeat].groupby("household_key").size()
            groups["repeat_target_count"] = groups.household_key.map(repeat_counts).fillna(0).astype("int64")
            groups["new_target_count"] = groups.target_count - groups.repeat_target_count
        end = position + len(rows)
        X[position:end] = rows[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y[position:end] = rows.relevance.to_numpy()
        digest.update(rows[ID_COLUMNS].to_numpy(dtype=np.int64).tobytes())
        if split == "validation":
            ids[position:end] = rows[ID_COLUMNS].to_numpy(dtype=np.int64)
            repeat_candidates[position:end] = rows.pair_recency_days.to_numpy() <= shard["cutoff"]
        kept_groups.append(groups)
        position = end
        print(f"verified {split} cutoff={shard['cutoff']} retained_rows={len(rows):,}", flush=True)
    if position != total:
        raise ValueError("Allocation and verified group totals differ")
    groups = pd.concat(kept_groups, ignore_index=True)
    # Original shard offsets are checked above; training consumes these new global offsets.
    groups["row_offset"] = np.r_[0, np.cumsum(groups.group_size.to_numpy())[:-1]]
    if groups.group_size.sum() != len(y):
        raise ValueError("sum(group) must equal number of rows")
    return RankingData(X, y, groups, ids, repeat_candidates, digest.hexdigest())
