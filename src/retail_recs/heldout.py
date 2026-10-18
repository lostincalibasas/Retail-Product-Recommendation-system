"""One frozen-policy final evaluation; no fitting, tuning or policy selection."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .contracts import DataConfig, FEATURE_COLUMNS, ID_COLUMNS, PRODUCT_COLUMNS, TRANSACTION_COLUMNS, feature_schema
from .data import build_snapshot, build_targets
from .diagnostics import aggregate_summaries
from .pipeline import label_snapshot, sha256, snapshot_groups, write_json
from .ranking_metrics import evaluate, recent_popularity_scores
from .training_data import verify_shard


APPROVED_CHECKPOINT = "b5517b4"
MODEL_SHA = "af9492f58af0b891b6f766213a5eb823bbb7fbf46aeb1719aba6c406515219ec"


def frozen_inputs(root):
    subprocess.run(["git", "merge-base", "--is-ancestor", APPROVED_CHECKPOINT, "HEAD"], cwd=root, check=True)
    model_dir = root/"artifacts/models/lambdarank-v1"
    model_manifest = json.loads((model_dir/"manifest.json").read_text())
    development_path = root/"artifacts/data/retail-ranking-v1/manifest.json"
    development = json.loads(development_path.read_text())
    if model_manifest["status"] != "complete" or (model_dir/"INCOMPLETE").exists():
        raise ValueError("Incomplete frozen model")
    if model_manifest["model_sha256"] != MODEL_SHA or model_manifest["best_iteration"] != 152:
        raise ValueError("Model differs from approved Strategy A")
    if model_manifest["schema"] != feature_schema() or development["config"] != DataConfig().to_dict():
        raise ValueError("Frozen feature/candidate contract mismatch")
    hashes = {str(development_path.relative_to(root)): model_manifest["data_manifest_sha256"],
              str((model_dir/"manifest.json").relative_to(root)): sha256(model_dir/"manifest.json"),
              **{str((model_dir/n).relative_to(root)): h for n, h in model_manifest["files_sha256"].items()},
              **model_manifest["implementation_sha256"], **development["source_audit"]["sources"]}
    for name, digest in hashes.items():
        if sha256(root/name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    return model_manifest, hashes


def load_test_sources(root):
    """Final-evaluation-only source read. Days 709 onward are excluded."""
    tx = pd.read_parquet(root/"data/processed/transaction_data.parquet", columns=TRANSACTION_COLUMNS,
                         filters=[("DAY", "<=", 708)])
    products = pd.read_parquet(root/"data/processed/product.parquet", columns=PRODUCT_COLUMNS)
    if tx.empty or tx.DAY.max() != 708 or tx.DAY.min() < 1:
        raise ValueError("Source must cover test through day 708 and contain no later rows")
    if tx.isna().any().any() or not np.isfinite(tx.to_numpy(dtype=float)).all():
        raise ValueError("Invalid transaction values")
    for col in ("household_key", "PRODUCT_ID", "DAY", "BASKET_ID"):
        if not tx[col].eq(tx[col].astype("int64")).all():
            raise ValueError("Fractional transaction identifiers")
    ids = products.PRODUCT_ID
    if ids.isna().any() or ids.duplicated().any() or not np.isfinite(ids).all() or not ids.eq(ids.astype("int64")).all():
        raise ValueError("Invalid product catalog identifiers")
    if not tx.PRODUCT_ID.isin(ids).all():
        raise ValueError("Missing catalog keys")
    return tx, products


def test_snapshot(tx, products, cutoff, config):
    """Use historical input only; compare against full-input approved builder."""
    if cutoff not in config.cutoffs("test"):
        raise ValueError("Cutoff outside the final test split")
    history = tx.loc[tx.DAY <= cutoff]
    snapshot = build_snapshot(history, products, cutoff, config)
    pd.testing.assert_frame_equal(snapshot, build_snapshot(tx, products, cutoff, config), check_exact=True)
    positive_history = history.loc[history.QUANTITY > 0]
    eligible = np.sort(positive_history.household_key.unique())
    np.testing.assert_array_equal(np.sort(snapshot.household_key.unique()), eligible)
    if not snapshot.PRODUCT_ID.isin(positive_history.PRODUCT_ID).all():
        raise ValueError("Candidate without a historical positive purchase")
    historical_pairs = pd.MultiIndex.from_frame(positive_history[["household_key", "PRODUCT_ID"]].drop_duplicates())
    repeat = pd.MultiIndex.from_frame(snapshot[["household_key", "PRODUCT_ID"]]).isin(historical_pairs)
    np.testing.assert_array_equal(repeat, snapshot.pair_recency_days <= cutoff)
    targets = build_targets(tx, cutoff, config, households=eligible)
    targets["cutoff_day"] = np.int64(cutoff)
    targets = targets[ID_COLUMNS].copy()
    targets["repeat_product"] = pd.MultiIndex.from_frame(targets[["household_key", "PRODUCT_ID"]]).isin(historical_pairs)
    rows = label_snapshot(snapshot, targets)
    groups = snapshot_groups(rows, targets)
    repeat_counts = targets.loc[targets.repeat_product].groupby("household_key").size()
    groups["repeat_target_count"] = groups.household_key.map(repeat_counts).fillna(0).astype("int64")
    groups["new_target_count"] = groups.target_count - groups.repeat_target_count
    summary = verify_shard(rows, targets, groups, cutoff)
    future = tx.loc[(tx.DAY > cutoff) & (tx.DAY <= cutoff+7) & (tx.QUANTITY > 0)]
    summary.update({"cutoff": cutoff, "history_max_day": int(history.DAY.max()),
                    "target_start": cutoff+1, "target_end": cutoff+7,
                    "history_invariance_passed": True,
                    "new_households_excluded": int(future.loc[~future.household_key.isin(eligible), "household_key"].nunique())})
    return rows, targets, groups, summary


def report_text(result):
    lines = ["# Phase 4: final held-out evaluation", "", "Strategy A was fixed before test access. No training, tuning or policy selection used these results.", "",
             "| Metric | Validation popularity | Test popularity | Change | Validation LambdaRank | Test LambdaRank | Change |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["comparison"]:
        p, l = row["popularity"], row["lambdarank"]
        lines.append(f"| {row['metric']} | {p['validation']:.8f} | {p['test']:.8f} | {p['delta']:+.8f} | {l['validation']:.8f} | {l['test']:.8f} | {l['delta']:+.8f} |")
    lines += ["", "All values are fractions. NDCG and precision average over every household-week group, including groups without purchases. Recall and candidate recall use micro denominators of all actual target products, including retrieval misses. Undefined per-group ratios are zero. Segment metrics preserve original Top 5 positions.", "",
              f"Test population: {result['data_summary']['candidate_rows']:,} candidates, {result['data_summary']['groups']:,} groups, {result['data_summary']['target_products']:,} actual targets, {result['data_summary']['positive_labels']:,} retrieved positives.",
              f"Group sizes: {json.dumps(result['data_summary']['group_size'], sort_keys=True)}.",
              f"No-purchase groups: {result['data_summary']['no_purchase_groups']:,}; zero-positive groups: {result['data_summary']['zero_positive_groups']:,}; complete candidate-miss groups: {result['data_summary']['candidate_miss_groups']:,}.", "",
              "## Temporal and artifact checks", "",
              "- Twelve weekly prediction cutoffs 624, 631, ..., 701; target days 625-708. History includes the cutoff; the target window excludes it and includes cutoff+7. Days 709-711 were not loaded.",
              "- Historical-only and full-input snapshots matched exactly at every cutoff. Original candidate/feature code and schema hashes matched the frozen model run. Every label, group offset/size and historical repeat flag passed verification.",
              "- Earlier test purchases may become history for later test cutoffs, as in weekly deployment. The model is never updated. This is rolling point-in-time evaluation, not a single forecast made on day 624 for the entire test horizon.",
              "- Both scorers rank the exact same candidate rows; ties resolve by ascending PRODUCT_ID. Every group is evaluated. No future positives are injected.",
              f"- Frozen model: `lambdarank-v1`, 152 trees, SHA-256 `{result['model_sha256']}`.",
              f"- Feature schema SHA-256: `{result['feature_schema_sha256']}`; all 25 ordered float32 features match model.feature_name().",
              f"- Test data manifest SHA-256: `{result['test_manifest_sha256']}`.",
              f"- Run Git provenance: `{result['git']['commit']}`; dirty: `{result['git']['dirty']}`. Approved predecessor: `{APPROVED_CHECKPOINT}`.",
              f"- Build/scoring/evaluation elapsed time: {result['elapsed_seconds']:.2f} s.", "",
              "## Limits and 80/20 explanation", "",
              "The model only orders the historical candidates it receives. Candidate coverage limits recall, especially discovery. Ranking gains measure prediction of observed purchases, not causal sales uplift. Inactive households stay in the denominator; historically unseen households are excluded and counted. Catalog brand/category metadata is assumed static, and stock availability is unavailable.", "",
              "This final test measures temporal generalization after validation-based selection. Repeated household-week queries are dependent; no independence-based confidence interval is claimed. The test period is now consumed for evaluation and must not be reused as fresh model-selection evidence. Results do not authorize tuning or deployment.", "",
              "Saved predictions, group metrics, schemas and checksummed manifests support audit and independent verification. Stop for approval before production integration.", ""]
    return "\n".join(lines)


def run(root):
    root = root.resolve(); start = time.monotonic()
    data_dir = root/"artifacts/data/retail-ranking-test-v1"
    output = root/"artifacts/evaluations/heldout-v1"
    if data_dir.exists() or output.exists():
        raise FileExistsError("Refusing to overwrite final evaluation or test artifacts")
    frozen, hashes = frozen_inputs(root)
    model_dir = root/"artifacts/models/lambdarank-v1"
    model = lgb.Booster(model_file=str(model_dir/"model.txt"))
    if model.feature_name() != FEATURE_COLUMNS or model.num_trees() != 152:
        raise ValueError("Model feature order or tree count mismatch")
    git = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
           "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())}
    sources = {str(p.relative_to(root)): sha256(p) for p in (root/"src/retail_recs").glob("*.py")}
    tx, products = load_test_sources(root)
    for directory in (data_dir, output):
        directory.mkdir(parents=True)
        (directory/"INCOMPLETE").write_text("Requires complete manifest before use\n")
    config = DataConfig()
    write_json(data_dir/"config.json", config.to_dict())
    write_json(data_dir/"feature_schema.json", feature_schema())
    summaries, shards, group_frames, prediction_paths = [], [], [], []
    for cutoff in config.cutoffs("test"):
        rows, targets, groups, summary = test_snapshot(tx, products, cutoff, config)
        # The model receives only schema-ordered features, never IDs or target labels.
        X = rows[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        pred = rows[ID_COLUMNS+["relevance"]].copy()
        pred["repeat_product"] = rows.pair_recency_days <= cutoff
        pred["popularity_score"] = recent_popularity_scores(X, FEATURE_COLUMNS)
        pred["lambdarank_score"] = model.predict(X, num_threads=4)
        directory = data_dir/"test"/f"cutoff_{cutoff}"
        directory.mkdir(parents=True)
        files = {}
        for name, frame in (("rows", rows), ("targets", targets), ("groups", groups)):
            path = directory/f"{name}.parquet"
            frame.to_parquet(path, index=False, compression="zstd")
            pd.testing.assert_frame_equal(frame, pd.read_parquet(path), check_exact=True)
            files[name] = {"path": path.relative_to(data_dir).as_posix(), "rows": len(frame), "sha256": sha256(path)}
        prediction_path = output/f"predictions_{cutoff}.parquet"
        pred.to_parquet(prediction_path, index=False, compression="zstd")
        prediction_paths.append(prediction_path); group_frames.append(groups)
        summaries.append(summary); shards.append({"cutoff": cutoff, "files": files, "diagnostics": summary})
        print(f"verified and scored test cutoff={cutoff} rows={len(rows):,} positives={int(rows.relevance.sum()):,}", flush=True)
    data_manifest = {"status": "complete", "mode": "final_heldout", "schema": feature_schema(), "config": config.to_dict(),
                     "git": git, "source_code_sha256": sources, "frozen_inputs_sha256": hashes,
                     "created_utc": datetime.now(timezone.utc).isoformat(),
                     "source_audit": {"read_filter": "DAY <= 708", "loaded_rows": len(tx), "day_min": int(tx.DAY.min()),
                                      "day_max": int(tx.DAY.max()), "nonpositive_quantity_rows_excluded": int((tx.QUANTITY <= 0).sum())},
                     "test": {"cutoffs": config.cutoffs("test"), "shards": shards, "summary": aggregate_summaries(summaries),
                              "new_household_window_occurrences_excluded": sum(s["new_households_excluded"] for s in summaries)}}
    write_json(data_dir/"manifest.json", data_manifest)
    (data_dir/"INCOMPLETE").unlink()
    del tx, products, X, rows, targets, pred
    predictions = pd.concat([pd.read_parquet(p) for p in prediction_paths], ignore_index=True)
    groups = pd.concat(group_frames, ignore_index=True)
    groups["row_offset"] = np.r_[0, groups.group_size.cumsum().to_numpy()[:-1]]
    sizes = groups.group_size.to_numpy(dtype=np.int32)
    metrics = {}
    for name in ("popularity", "lambdarank"):
        metrics[name], frame, _ = evaluate(predictions[name+"_score"].to_numpy(), predictions.relevance.to_numpy(), sizes,
            predictions.PRODUCT_ID.to_numpy(), groups, predictions.repeat_product.to_numpy())
        metrics[name]["precision5_all_groups"] = float((frame.hits_at_5/5).mean())
        for col in ("hits_at_5", "ndcg5_e2e", "ndcg5_candidate", "recall5"):
            groups[name+"_"+col] = frame[col].to_numpy()
    predictions.to_parquet(output/"test_predictions.parquet", index=False, compression="zstd")
    groups.to_parquet(output/"test_group_metrics.parquet", index=False, compression="zstd")
    comparisons = []
    selectors = {"End-to-end NDCG@5": lambda m:m["overall"]["ndcg5_e2e_all_groups"],
                 "Precision@5": lambda m:m["overall"]["hits_at_5"]/(5*m["overall"]["groups"]),
                 "Recall@5 micro": lambda m:m["overall"]["recall5_micro"],
                 "Candidate recall micro": lambda m:m["overall"]["candidate_recall_micro"],
                 "Repeat Recall@5 micro": lambda m:m["repeat"]["recall5_micro"],
                 "New Recall@5 micro": lambda m:m["new"]["recall5_micro"]}
    for label, select in selectors.items():
        comparison = {"metric": label}
        for name in metrics:
            val, test = select(frozen["metrics"][name]), select(metrics[name])
            comparison[name] = {"validation": val, "test": test, "delta": test-val,
                                "relative_change": (test-val)/val if val else None}
        comparisons.append(comparison)
    for name, digest in hashes.items():
        if sha256(root/name) != digest:
            raise ValueError(f"Frozen input changed during final evaluation: {name}")
    result = {"status": "complete", "policy": "approved original Top 5 Strategy A; no selection on test",
              "created_utc": datetime.now(timezone.utc).isoformat(), "git": git,
              "model_sha256": MODEL_SHA, "model_version": "lambdarank-v1", "model_trees": 152,
              "model_configuration": frozen["configuration"], "feature_schema": feature_schema(),
              "feature_schema_sha256": sha256(model_dir/"feature_schema.json"),
              "test_manifest_sha256": sha256(data_dir/"manifest.json"), "source_code_sha256": sources,
              "python": platform.python_version(), "packages": {n:importlib.metadata.version(n) for n in ("numpy","pandas","pyarrow","lightgbm")},
              "metrics": metrics, "comparison": comparisons, "data_summary": data_manifest["test"]["summary"],
              "new_household_window_occurrences_excluded": data_manifest["test"]["new_household_window_occurrences_excluded"],
              "elapsed_seconds": time.monotonic()-start}
    result["files_sha256"] = {p.name:sha256(p) for p in output.iterdir() if p.name != "INCOMPLETE"}
    write_json(output/"manifest.json", result)
    (output/"INCOMPLETE").unlink()
    write_json(root/"reports/phase4_test.json", result)
    (root/"reports/phase4_test.md").write_text(report_text(result), encoding="utf-8")
    print(json.dumps(comparisons, indent=2), flush=True)


def main():
    run(Path(__file__).resolve().parents[2])
