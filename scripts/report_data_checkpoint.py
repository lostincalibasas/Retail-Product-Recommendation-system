"""Verify a completed development build and render its measured checkpoint report."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retail_recs.contracts import DataConfig, FEATURE_COLUMNS, ID_COLUMNS
from retail_recs.pipeline import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/data/retail-ranking-v1"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    artifact_dir = args.artifacts if args.artifacts.is_absolute() else root / args.artifacts
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["mode"] != "full_development":
        raise ValueError("Checkpoint report requires a complete full development build")
    if (artifact_dir / "INCOMPLETE").exists() or (artifact_dir / "test").exists():
        raise ValueError("Build is incomplete or test artifacts unexpectedly exist")
    config = DataConfig(**manifest["config"])
    if manifest["schema"]["feature_order"] != FEATURE_COLUMNS:
        raise ValueError("Feature order differs from current contract")
    for file, expected in {**manifest["source_code_sha256"], **manifest["source_audit"]["sources"]}.items():
        if sha256(root / file) != expected:
            raise ValueError(f"Source changed since build: {file}")
    verified_files = total_bytes = 0
    for split in ("train", "validation"):
        if manifest["splits"][split]["cutoffs"] != config.cutoffs(split):
            raise ValueError(f"Incomplete cutoff grid: {split}")
        for shard in manifest["splits"][split]["shards"]:
            for item in shard["files"].values():
                path = artifact_dir / item["path"]
                if sha256(path) != item["sha256"] or pq.ParquetFile(path).metadata.num_rows != item["rows"]:
                    raise ValueError(f"Artifact verification failed: {path}")
                verified_files += 1
                total_bytes += item["bytes"]
            rows_path = artifact_dir / shard["files"]["rows"]["path"]
            if pq.ParquetFile(rows_path).schema_arrow.names != ID_COLUMNS + FEATURE_COLUMNS + ["relevance"]:
                raise ValueError("Disk columns differ from feature schema")
            rows = pd.read_parquet(rows_path, columns=ID_COLUMNS + ["relevance"])
            groups = pd.read_parquet(artifact_dir / shard["files"]["groups"]["path"])
            expected = rows.groupby(["household_key", "cutoff_day"], sort=False).relevance.agg(
                group_size="size", positive_labels="sum").reset_index()
            pd.testing.assert_frame_equal(expected, groups[expected.columns])
            offsets = np.r_[0, np.cumsum(groups.group_size.to_numpy())[:-1]]
            if not np.array_equal(offsets, groups.row_offset.to_numpy()):
                raise ValueError("Disk ranking group offsets are inconsistent")
    summary = {
        "artifact_path": str(artifact_dir.relative_to(root)).replace("\\", "/"),
        "manifest_sha256": sha256(artifact_dir / "manifest.json"),
        "build_git_commit": manifest["git_base_commit"],
        "build_git_dirty": manifest["git_working_tree_dirty_at_build"],
        "python": manifest["python"], "elapsed_seconds": manifest["elapsed_seconds"],
        "source_audit": manifest["source_audit"],
        "invariance_checks": manifest["invariance_checks"],
        "artifact_verification": {"parquet_files_verified": verified_files,
                                  "total_parquet_bytes": total_bytes,
                                  "all_group_offsets_verified": True,
                                  "source_and_code_hashes_match": True},
        "splits": {name: value["summary"] for name, value in manifest["splits"].items()},
        "test": manifest["test"],
    }
    write_json(root / "reports/phase2_data_checkpoint.json", summary)
    train, valid = [summary["splits"][name] for name in ("train", "validation")]
    table = []
    for title, key in [("Weekly snapshots", "snapshots"), ("Candidate rows", "candidate_rows"),
                       ("Household-week groups", "groups"), ("Positive candidate labels", "positive_labels"),
                       ("Trainable groups (both label values)", "trainable_groups"),
                       ("Zero-positive candidate groups", "zero_positive_groups"),
                       ("No-purchase groups", "no_purchase_groups"),
                       ("Purchase groups with no candidate hit", "candidate_miss_groups"),
                       ("Groups with target purchases", "groups_with_target_purchases"),
                       ("All actual household/product targets", "target_products"),
                       ("Targets missing from candidates", "completely_missed_target_products"),
                       ("Excluded new-household window occurrences", "new_household_window_occurrences_excluded")]:
        table.append(f"| {title} | {train[key]:,} | {valid[key]:,} |")
    for title, key in [("Micro candidate recall", "candidate_recall_micro"),
                       ("Macro candidate recall", "candidate_recall_macro"),
                       ("Candidate-miss rate among purchase groups", "candidate_miss_rate")]:
        table.append(f"| {title} | {train[key]:.4%} | {valid[key]:.4%} |")
    table.append(f"| Group size min / median / p95 / max | " + " | ".join(
        " / ".join(f"{s['group_size'][k]:g}" for k in ("min", "p50", "p95", "max"))
        for s in (train, valid)) + " |")
    lines = ["# Phase 2: data and feature integration checkpoint", "",
             "Generated from the completed local artifact manifest and verified Parquet files. No model was trained.", "",
             f"- Build code commit: `{manifest['git_base_commit']}`; dirty at build: `{manifest['git_working_tree_dirty_at_build']}`.",
             f"- Python: {manifest['python']}; data build time (excluding initial source load): {manifest['elapsed_seconds']:.1f} seconds.",
             f"- Artifacts: `{summary['artifact_path']}`; {verified_files} Parquet files verified by checksum and row count.",
             f"- Manifest SHA-256: `{summary['manifest_sha256']}`.",
             f"- Parquet storage: {total_bytes:,} bytes. All ranking-group offsets independently verified from disk.",
             "- 25 ordered float32 features; int64 identifiers; int8 binary labels; maximum 200 candidates per household.",
             "", "## Measured development data", "",
             "| Metric | Train | Validation |", "| --- | ---: | ---: |", *table, "",
             "## Temporal and source checks", "",
             "- Train cutoffs: 365–533, weekly; target days 366–540. Validation cutoffs: 540–617; target days 541–624.",
             f"- Transaction loader filter: `{manifest['source_audit']['read_filter']}`; {manifest['source_audit']['loaded_rows']:,} rows loaded.",
             f"- Excluded nonpositive-quantity source rows: {manifest['source_audit']['nonpositive_quantity_rows_excluded_from_history_and_labels']:,}.",
             f"- Positive-quantity rows with negative sales retained/audited: {manifest['source_audit']['positive_quantity_negative_sales_rows_retained']:,}.",
             "- First/last snapshot of each split passed exact feature/candidate equality with all future transactions removed, and complete Parquet roundtrip equality.",
             "- All snapshot labels, candidate uniqueness, finite feature dtypes, historically eligible household coverage and group sizes validated during build.",
             "- Test target days 625–708 remain sealed: no test snapshots, labels, metrics or model-selection use.", "",
             "## Interpretation", "",
             "Candidate recall measures retrieval before ranking; a ranker cannot recover missing candidates. Zero-positive groups include both no-purchase weeks and complete candidate misses. The miss rate denominator includes only groups with actual purchases. All eligible households, including inactive ones, remain in evaluation.", "",
             "The fixed 200-product budget and recency/category/popularity rules are an initial retrieval design, not a tuned optimum. Historical sales do not establish current stock. Static catalog metadata is an explicit assumption; demographics and promotions are unused.", "",
             "No NDCG@5 or Recall@5 ranking result exists yet. At the next approved checkpoint, baseline and LambdaRank must use identical candidates; end-to-end ideal DCG must count actual targets missed by retrieval. Consider candidate coverage before spending time tuning ranking.", "",
             "Environment smoke results are in `environment_smoke.json`. Contracts and agent ownership are in `docs/phase2_contract.md`. Reproduce this report with `python scripts/report_data_checkpoint.py` from the Phase 2 environment.", ""]
    (root / "reports/phase2_data_checkpoint.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
