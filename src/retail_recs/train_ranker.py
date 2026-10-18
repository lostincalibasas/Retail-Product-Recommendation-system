"""Train one fixed LambdaRank configuration and compare on sealed-development data."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import FEATURE_COLUMNS, ID_COLUMNS, feature_schema
from .pipeline import sha256, write_json
from .ranking_metrics import evaluate, make_validation_metric, recent_popularity_scores
from .training_data import check_manifest, first_purchase_days, load_split


def process_memory():
    """Current and OS lifetime peak resident working set; excludes other processes."""
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        value = Counters(); value.cb = ctypes.sizeof(value)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"rss_bytes": value.WorkingSetSize, "peak_rss_bytes": value.PeakWorkingSetSize}
    import resource
    current = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return {"rss_bytes": current, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


class MemorySampler:
    def __init__(self):
        self.peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, process_memory()["rss_bytes"])
            self.stop.wait(0.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set(); self.thread.join()


def fit_ranker(train, valid, config, progress_path):
    history = {}
    start = time.monotonic()
    training = lgb.Dataset(train.X, label=train.y, group=train.sizes,
                           feature_name=FEATURE_COLUMNS, free_raw_data=True)
    validation = lgb.Dataset(valid.X, label=valid.y, group=valid.sizes,
                             reference=training, feature_name=FEATURE_COLUMNS, free_raw_data=True)
    def progress(env):
        if env.iteration == 0 or (env.iteration + 1) % 5 == 0:
            record = {"iteration": env.iteration + 1, "seconds": round(time.monotonic() - start, 3),
                      "validation_ndcg5_e2e": float(env.evaluation_result_list[0][2]), **process_memory()}
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    progress.order = 20
    with MemorySampler() as memory:
        model = lgb.train(config["params"], training, num_boost_round=config["num_boost_round"],
                          valid_sets=[validation], valid_names=["validation"],
                          feval=make_validation_metric(valid.y, valid.sizes, valid.ids[:, 2], valid.groups.target_count),
                          callbacks=[lgb.record_evaluation(history), progress,
                                     lgb.early_stopping(config["early_stopping_rounds"], first_metric_only=True,
                                                        verbose=True, min_delta=config["early_stopping_min_delta"])])
    elapsed = time.monotonic() - start
    return model, history, {"fit_seconds_including_binning_and_validation": elapsed,
                            "sampled_fit_peak_rss_bytes": memory.peak, **process_memory()}


def save_predictions(path, valid, baseline, scores):
    writer = None
    try:
        for start in range(0, len(scores), 250_000):
            end = min(start + 250_000, len(scores))
            frame = pd.DataFrame(valid.ids[start:end], columns=ID_COLUMNS)
            frame["relevance"] = valid.y[start:end]
            frame["repeat_product"] = valid.repeat_candidates[start:end]
            frame["popularity_score"] = baseline[start:end]
            frame["lambdarank_score"] = scores[start:end]
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def render_report(result):
    baseline, ranker = result["metrics"]["popularity"], result["metrics"]["lambdarank"]
    lines = ["# Phase 3: popularity versus LambdaRank validation", "",
             "Measured on the frozen validation candidate sets. Test targets remain sealed.", "",
             "## Overall validation metrics", "",
             "| Metric | Recent popularity | LambdaRank |", "| --- | ---: | ---: |"]
    for label, key in [("End-to-end NDCG@5, all groups", "ndcg5_e2e_all_groups"),
                       ("End-to-end NDCG@5, purchase groups", "ndcg5_e2e_purchase_groups"),
                       ("Candidate-only NDCG@5, all groups", "ndcg5_candidate_all_groups"),
                       ("Candidate-only NDCG@5, groups with positive candidates", "ndcg5_candidate_positive_groups"),
                       ("Recall@5 micro", "recall5_micro"),
                       ("Recall@5 macro, all groups", "recall5_macro_all_groups"),
                       ("Recall@5 macro, purchase groups", "recall5_macro_purchase_groups"),
                       ("Candidate recall micro", "candidate_recall_micro")]:
        lines.append(f"| {label} | {baseline['overall'][key]:.6f} | {ranker['overall'][key]:.6f} |")
    lines += ["", "## Repeat and new-product segments", "",
              "Repeat means this household bought the product before the cutoff. New means no such prior household purchase, including globally unseen products. Segment metrics retain the original top-five positions; recommendations are not reranked within a segment.", "",
              "| Segment | Targets | Retrieved | Candidate recall | Popularity Recall@5 micro | LambdaRank Recall@5 micro |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for segment in ("repeat", "new"):
        b, r = baseline[segment], ranker[segment]
        lines.append(f"| {segment} | {r['targets']:,} | {r['retrieved_targets']:,} | {r['candidate_recall_micro']:.6f} | {b['recall5_micro']:.6f} | {r['recall5_micro']:.6f} |")
    lines += ["", "## Population, selection and checks", "",
              f"- Training: {result['training']['rows']:,} rows / {result['training']['groups']:,} informative groups. Excluded from fitting only: {result['training']['excluded_groups']:,} groups without both label values.",
              f"- Validation: {result['validation']['rows']:,} rows / {ranker['overall']['groups']:,} groups, including {ranker['overall']['no_purchase_groups']:,} no-purchase and {ranker['overall']['candidate_miss_groups']:,} complete candidate-miss groups.",
              "- All-group macro metrics assign zero to no-purchase/zero-positive groups. Purchase-only and positive-candidate-only companion metrics explicitly state their denominators. Micro recall divides total top-five hits by all actual targets.",
              "- End-to-end ideal DCG includes true targets missing from the candidate set. Candidate-only ideal DCG uses positives present in the candidate set. Both use binary gains and log2 rank discounts.",
              "- Ties resolve by ascending PRODUCT_ID. Baseline and ranker score exactly the same stored candidates.",
              "- Artifact checksums, schema order, group sizes/offsets, target counts and every candidate label were checked before fitting. All validation groups retained.",
              f"- One fixed configuration; best iteration {result['best_iteration']} selected by validation end-to-end NDCG@5 across all groups. {result['rounds_executed']} rounds executed; patience {result['configuration']['early_stopping_rounds']}.",
              "- Validation is used for early stopping; these results are development estimates, not an independent test result.",
              f"- Serialized model reload reproduced all validation scores exactly: {result['model_reload_exact_match']}.", "",
              "## Runtime and memory", "",
              f"- Verification/data loading: {result['timing']['load_seconds']:.2f} s.",
              f"- Baseline evaluation: {result['timing']['baseline_evaluation_seconds']:.2f} s.",
              f"- Fit including bin construction and per-round validation: {result['training']['fit_seconds_including_binning_and_validation']:.2f} s.",
              f"- Final LambdaRank prediction: {result['timing']['prediction_seconds']:.2f} s.",
              f"- Peak process resident working set: {result['memory']['peak_rss_bytes']/2**30:.3f} GiB. Sampled fit peak: {result['training']['sampled_fit_peak_rss_bytes']/2**30:.3f} GiB (250 ms samples). These are this Python process, not system-wide memory.",
              f"- Float32 train + validation feature arrays: {result['memory']['feature_array_bytes']/2**30:.3f} GiB.", "",
              "## Reproducibility", "",
              f"- Data manifest SHA-256: `{result['data_manifest_sha256']}`.",
              f"- Validation candidate identity SHA-256: `{result['validation']['candidate_sha256']}`.",
              f"- Model: `{result['artifact_directory']}/model.txt`; SHA-256 `{result['model_sha256']}`.",
              "- Artifacts also include parameters, feature schema, full scored validation candidates, group metrics and evaluation history. Large artifacts stay outside Git.",
              "", "```json", json.dumps(result['configuration'], indent=2), "```", "",
              "## 80/20 explanation", "",
              "Retrieval is held fixed so the comparison measures ordering. Popularity gives every household the same product-popularity score within its candidate pool. LambdaRank can use household-product recency, frequency and preferences to personalize the order. No-purchase groups teach no preference ordering and are excluded from fitting, while every validation group remains represented. Candidate recall limits what either scorer can recover.", ""]
    return "\n".join(lines)


def run(root, data_directory, output, config_path):
    root, data_directory, output = root.resolve(), data_directory.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite model run: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    start = time.monotonic()
    manifest = check_manifest(root, data_directory)
    output.mkdir(parents=True)
    (output / "INCOMPLETE").write_text("Use only a run with complete manifest.json\n", encoding="utf-8")
    first_days = first_purchase_days(root)
    train = load_split(data_directory, manifest, "train")
    valid = load_split(data_directory, manifest, "validation", first_days)
    del first_days
    gc.collect()
    load_seconds = time.monotonic() - start
    feature_bytes = train.X.nbytes + valid.X.nbytes
    baseline_scores = recent_popularity_scores(valid.X, FEATURE_COLUMNS)
    tick = time.monotonic()
    baseline, baseline_groups, _ = evaluate(baseline_scores, valid.y, valid.sizes, valid.ids[:, 2], valid.groups, valid.repeat_candidates)
    baseline_seconds = time.monotonic() - tick
    write_json(output / "baseline_metrics.json", baseline)
    print("BASELINE " + json.dumps(baseline["overall"]), flush=True)
    model, history, fit_stats = fit_ranker(train, valid, config, output / "progress.jsonl")
    if model.feature_name() != FEATURE_COLUMNS:
        raise ValueError("Trained model feature order differs from schema")
    train.X = None
    gc.collect()
    tick = time.monotonic()
    scores = model.predict(valid.X, num_iteration=model.best_iteration, num_threads=config["params"]["num_threads"])
    prediction_seconds = time.monotonic() - tick
    metrics, ranker_groups, _ = evaluate(scores, valid.y, valid.sizes, valid.ids[:, 2], valid.groups, valid.repeat_candidates)
    model.save_model(str(output / "model.txt"), num_iteration=model.best_iteration)
    reloaded = lgb.Booster(model_file=str(output / "model.txt"))
    reloaded_scores = reloaded.predict(valid.X, num_threads=config["params"]["num_threads"])
    exact = np.array_equal(scores, reloaded_scores)
    if not exact:
        raise ValueError("Saved model predictions differ after reload")
    del reloaded_scores, reloaded
    save_predictions(output / "validation_predictions.parquet", valid, baseline_scores, scores)
    group_output = valid.groups.copy()
    for prefix, frame in (("popularity", baseline_groups), ("lambdarank", ranker_groups)):
        for column in ("hits_at_5", "ndcg5_e2e", "ndcg5_candidate", "recall5"):
            group_output[f"{prefix}_{column}"] = frame[column].to_numpy()
    group_output.to_parquet(output / "validation_group_metrics.parquet", index=False, compression="zstd")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True)
    result = {
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_directory": output.relative_to(root).as_posix(),
        "git_base_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_dirty_at_run": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "implementation_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in sorted((root / "src/retail_recs").glob("*.py"))},
        "data_manifest_sha256": sha256(data_directory / "manifest.json"),
        "configuration": config, "schema": feature_schema(),
        "python": platform.python_version(),
        "packages": {n: importlib.metadata.version(n) for n in ("lightgbm", "numpy", "pandas", "pyarrow", "scikit-learn")},
        "best_iteration": model.best_iteration,
        "rounds_executed": len(history["validation"]["ndcg5_e2e_all_groups"]),
        "model_reload_exact_match": exact, "model_sha256": sha256(output / "model.txt"),
        "training": {"rows": len(train.y), "groups": len(train.groups),
                     "positive_labels": int(train.y.sum()), "negative_labels": int((train.y == 0).sum()),
                     "min_group_size": int(train.sizes.min()), "max_group_size": int(train.sizes.max()),
                     "excluded_groups": manifest["splits"]["train"]["summary"]["groups"]-len(train.groups), **fit_stats},
        "validation": {"rows": len(valid.y), "groups": len(valid.groups), "candidate_sha256": valid.candidate_sha256,
                       "positive_labels": int(valid.y.sum()), "negative_labels": int((valid.y == 0).sum()),
                       "min_group_size": int(valid.sizes.min()), "max_group_size": int(valid.sizes.max())},
        "temporal_splits": {"train_cutoffs": manifest["splits"]["train"]["cutoffs"],
                            "validation_cutoffs": manifest["splits"]["validation"]["cutoffs"],
                            "test_target_days_sealed": [625, 708]},
        "timing": {"load_seconds": load_seconds, "baseline_evaluation_seconds": baseline_seconds,
                   "prediction_seconds": prediction_seconds, "total_seconds": time.monotonic()-start},
        "memory": {**process_memory(), "feature_array_bytes": feature_bytes},
        "metrics": {"popularity": baseline, "lambdarank": metrics},
        "test_status": "sealed; no test targets loaded or evaluated",
        "group_policy": "all groups retained in validation; undefined zero-denominator group metrics assigned zero; companion conditional means reported",
    }
    write_json(output / "evaluation_history.json", history)
    write_json(output / "config.json", config)
    write_json(output / "feature_schema.json", feature_schema())
    result["files_sha256"] = {p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name != "INCOMPLETE"}
    write_json(output / "manifest.json", result)
    (output / "INCOMPLETE").unlink()
    write_json(root / "reports/phase3_validation.json", result)
    (root / "reports/phase3_validation.md").write_text(render_report(result), encoding="utf-8")
    print("RESULT " + json.dumps(result["metrics"]), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data", type=Path, default=Path("artifacts/data/retail-ranking-v1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/lambdarank-v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/lambdarank_v1.json"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    run(root, root / args.data, root / args.output, root / args.config)


if __name__ == "__main__":
    main()
