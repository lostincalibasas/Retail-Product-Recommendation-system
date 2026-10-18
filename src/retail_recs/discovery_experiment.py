"""Fixed-model, validation-only comparison of a reserved fifth discovery slot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from .contracts import DataConfig, ID_COLUMNS, feature_schema
from .pipeline import sha256, write_json
from .ranking_metrics import group_metrics, summarize_metrics, top_indices
from .training_data import verify_shard


def discovery_top5(scores, group_sizes, product_ids, is_new):
    """Keep ranks 1-4; put the best unselected historical-new candidate fifth.

    Only frozen scores, IDs, group sizes and historical novelty are accepted.
    Labels and target counts are not policy inputs. Score ties use ascending ID.
    If no unselected new candidate exists, retain the original fifth candidate.
    This is not an 'at least one new item' rule: a new item in ranks 1-4 does not
    cancel the reserved fifth slot.
    """
    scores, sizes, ids, new = map(np.asarray, (scores, group_sizes, product_ids, is_new))
    if new.dtype != np.dtype(bool) or new.shape != scores.shape or (sizes < 5).any():
        raise ValueError("Boolean novelty aligned to scores and groups of at least five required")
    original = top_indices(scores, sizes, ids)
    discovery = original.copy()
    available = np.zeros(len(sizes), dtype=bool)
    offset = 0
    for i, size in enumerate(sizes):
        candidates = np.flatnonzero(new[offset:offset+size]) + offset
        candidates = candidates[~np.isin(candidates, original[i, :4])]
        if len(candidates):
            available[i] = True
            best = np.lexsort((ids[candidates], -scores[candidates]))[0]
            discovery[i, 4] = candidates[best]
        offset += size
    return original, discovery, available


def verify_recommendations(original, discovery, sizes, is_new):
    sizes = np.asarray(sizes)
    offsets = np.r_[0, sizes.cumsum()[:-1]]
    for top in (original, discovery):
        if top.shape != (len(sizes), 5) or not (
                (top >= offsets[:, None]) & (top < (offsets+sizes)[:, None])).all():
            raise ValueError("Recommendation lies outside its original query candidate set")
        if (np.diff(np.sort(top, axis=1), axis=1) == 0).any():
            raise ValueError("Duplicate recommendation within query")
    if not np.array_equal(original[:, :4], discovery[:, :4]):
        raise ValueError("Ranks 1-4 changed")
    changed = original[:, 4] != discovery[:, 4]
    if not np.asarray(is_new)[discovery[changed, 4]].all():
        raise ValueError("Replacement must be historically new")
    return changed


def evaluate_list(top, labels, sizes, groups, is_new):
    overall = group_metrics(top, labels, sizes, groups.target_count)
    repeat = group_metrics(top, labels, sizes, groups.repeat_target_count, ~is_new)
    new = group_metrics(top, labels, sizes, groups.new_target_count, is_new)
    counts = is_new[top].sum(axis=1)
    result = {"overall": summarize_metrics(overall), "repeat": summarize_metrics(repeat),
              "new": summarize_metrics(new), "precision5_all_groups": float((overall.hits_at_5/5).mean()),
              "new_slot_share": float(is_new[top].mean()), "new_slots": int(counts.sum()),
              "groups_with_new": int((counts > 0).sum())}
    frame = groups[["household_key", "cutoff_day", "target_count"]].copy()
    for column in ("hits_at_5", "ndcg5_e2e", "recall5"):
        frame[column] = overall[column].to_numpy()
    frame["precision5"] = overall.hits_at_5.to_numpy()/5
    frame["repeat_hits"] = repeat.hits_at_5.to_numpy()
    frame["new_hits"] = new.hits_at_5.to_numpy()
    frame["new_slots"] = counts
    return result, frame


def render_report(r):
    a, b = r["metrics"]["A"], r["metrics"]["B"]
    lines = ["# Phase 3.2: controlled discovery-slot experiment", "",
             "Measured validation experiment only. Strategy B is not promoted.", "",
             "A is the unchanged LambdaRank Top 5. B keeps ranks 1-4 exactly and puts the highest-scoring new-to-household candidate not already selected in position 5. This slot is reserved even when ranks 1-4 already contain new products. Ties use ascending PRODUCT_ID. If no eligible new candidate exists, the original fifth item remains.", "",
             "## Measured results", "",
             "| Metric | A: LambdaRank | B: discovery slot | B minus A | Relative change |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in r["comparison"]:
        lines.append(f"| {row['metric']} | {row['A']:.8f} | {row['B']:.8f} | {row['delta']:+.8f} | {100*row['relative_change']:+.2f}% |")
    lines += ["", "Metrics are fractions, not percentages. NDCG is the mean across all groups with full-target ideal DCG. Precision@5 is mean hits/5 across all groups (equivalently total hits / all 149,885 slots). Recall and segment recall are micro: hits / all actual targets, including candidate misses. The new-slot share is new recommendations / all recommendation slots.", "",
              "No-purchase and zero-positive-candidate groups remain in evaluation. Undefined per-group NDCG/recall is zero. The JSON also includes purchase-only and macro companion metrics with explicit denominators.", "",
              "## Exposure and affected groups", "",
              f"- Same {r['validation']['rows']:,} frozen candidates in {r['validation']['groups']:,} household-week groups; 200 candidates/group and five recommendations/group.",
              f"- Changed fifth item in **{r['affected_groups']:,} groups ({100*r['affected_group_share']:.4f}%)**; ranks 1-4 identical everywhere.",
              f"- An eligible unselected new candidate existed in {r['eligible_groups']:,} groups; fallback used in {r['fallback_groups']:,}.",
              f"- New slots: {a['new_slots']:,} -> {b['new_slots']:,}; groups with any new recommendation: {a['groups_with_new']:,} -> {b['groups_with_new']:,}.",
              f"- All-target candidate recall stays {a['overall']['candidate_recall_micro']:.8f}; no candidates or future positives were injected.", "",
              "## Hit-level accounting and regressions", "",
              f"- Total hits: {a['overall']['hits_at_5']:,} -> {b['overall']['hits_at_5']:,}.",
              f"- Repeat hits: {a['repeat']['hits_at_5']:,} -> {b['repeat']['hits_at_5']:,} (denominator {a['repeat']['targets']:,} true repeat targets).",
              f"- New hits: {a['new']['hits_at_5']:,} -> {b['new']['hits_at_5']:,} (denominator {a['new']['targets']:,} true new targets).",
              f"- Changed-group fifth-slot transitions: {json.dumps(r['changed_group_hit_transitions'], sort_keys=True)}. Keys describe old_hit -> new_hit.",
              f"- Group NDCG improved in {r['group_outcomes']['improved']:,}, regressed in {r['group_outcomes']['regressed']:,}, and was unchanged in {r['group_outcomes']['unchanged']:,} groups.", "",
              "Only position 5 changes, so a replaced repeat purchase and added new purchase trade off at the same discount. All measured regressions are reported above; increased exposure alone does not establish business value.", "",
              "## Verification and provenance", "",
              "- The selection function accepts only scores, product IDs, group sizes and historical novelty. Relevance labels and target counts are used afterwards solely for evaluation.",
              "- Historical novelty is verified against the original pair-recency feature: never purchased means pair_recency_days == cutoff + 1. Candidate history was built with DAY <= cutoff; validation targets span cutoff < DAY <= cutoff + 7.",
              f"- All {r['validation']['verified_shard_hashes']} validation row/group/target file hashes passed. Every saved prediction ID, relevance label, group count and repeat flag was matched to its frozen shard. Candidate labels were independently checked against seven-day target sets.",
              "- Both lists contain five distinct original candidates from the same query. No output can add an SKU, cross a household/week boundary, or change ranks 1-4. All changed fifth items are new products replacing repeats.",
              "- Strategy A reproduces every existing overall/repeat/new metric within 1e-12. All model-run artifacts were checksum-verified before and after the experiment and remain unchanged.",
              f"- Model SHA-256: `{r['model_sha256']}`; prediction SHA-256: `{r['prediction_sha256']}`.",
              f"- Base Git checkpoint: `{r['git_base_commit']}`. New experiment source SHA-256: `{r['experiment_source_sha256']}`.",
              f"- Experiment elapsed time through evaluation and output preparation: {r['elapsed_seconds']:.2f} seconds.",
              "- Validation cutoffs 540-617 / target days 541-624 only. No raw transaction reads, test access, fitting, feature changes, candidate changes or hyperparameter changes.", "",
              "Artifacts: `artifacts/experiments/discovery-v1/` contains both complete recommendation lists, per-group metrics, and the result JSON. Existing model files and earlier phase reports are unchanged. The experiment refuses to overwrite its output directory.", "",
              "## Decision boundary", "",
              "Keep A as the current model policy pending review. B is a development comparison, not an automatically selected or deployed strategy. Validation was already used for model selection and diagnostics; this is not independent test evidence. Review the discovery gain against repeat and overall losses before approving any next step.", ""]
    return "\n".join(lines)


def run(root, output):
    start = time.monotonic()
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {output}")
    model = root/"artifacts/models/lambdarank-v1"
    manifest = json.loads((model/"manifest.json").read_text())
    if manifest["status"] != "complete" or (model/"INCOMPLETE").exists():
        raise ValueError("Complete frozen model artifact required")
    hashes = {"manifest.json": sha256(model/"manifest.json"), **manifest["files_sha256"]}
    for name, digest in hashes.items():
        if sha256(model/name) != digest:
            raise ValueError(f"Frozen artifact checksum mismatch: {name}")
    data = root/"artifacts/data/retail-ranking-v1"
    if sha256(data/"manifest.json") != manifest["data_manifest_sha256"]:
        raise ValueError("Data manifest differs from trained model")
    dm = json.loads((data/"manifest.json").read_text())
    shards = dm["splits"]["validation"]["shards"]
    if dm["schema"] != feature_schema() or [s["cutoff"] for s in shards] != DataConfig().cutoffs("validation"):
        raise ValueError("Unexpected validation schema or cutoffs")
    pred = pd.read_parquet(model/"validation_predictions.parquet")
    groups = pd.read_parquet(model/"validation_group_metrics.parquet")
    sizes = groups.group_size.to_numpy(dtype=np.int64)
    if not np.all(sizes == 200) or not np.array_equal(groups.row_offset, np.r_[0, sizes.cumsum()[:-1]]):
        raise ValueError("Frozen query sizes/offsets mismatch")
    new = ~pred.repeat_product.to_numpy(dtype=bool)
    # Select both lists before extracting labels or opening validation target sets.
    a, b, eligible = discovery_top5(pred.lambdarank_score.to_numpy(), sizes, pred.PRODUCT_ID.to_numpy(), new)
    changed = verify_recommendations(a, b, sizes, new)
    if new[a[changed, 4]].any():
        raise ValueError("An existing fifth-position new item should never be replaced")
    offset, group_offset, checksum_count = 0, 0, 0
    for shard in shards:
        frames = {}
        for kind in ("rows", "groups", "targets"):
            record = shard["files"][kind]; path = (data/record["path"]).resolve()
            if not path.is_relative_to((data/"validation").resolve()) or sha256(path) != record["sha256"]:
                raise ValueError("Validation artifact path or checksum mismatch")
            frames[kind] = pd.read_parquet(path); checksum_count += 1
        rows, targets, group = frames["rows"], frames["targets"], frames["groups"]
        verify_shard(rows, targets, group, shard["cutoff"])
        n = len(rows); part = pred.iloc[offset:offset+n]
        for col in ID_COLUMNS+["relevance"]:
            np.testing.assert_array_equal(rows[col], part[col])
        np.testing.assert_array_equal(rows.pair_recency_days == shard["cutoff"]+1, new[offset:offset+n])
        for col in ("household_key", "cutoff_day", "group_size", "target_count", "positive_labels"):
            np.testing.assert_array_equal(group[col], groups.iloc[group_offset:group_offset+len(group)][col])
        offset += n; group_offset += len(group)
    if offset != len(pred) or group_offset != len(groups):
        raise ValueError("Rows outside the validation population")
    labels = pred.relevance.to_numpy()
    metrics, frames = {}, {}
    for name, top in (("A", a), ("B", b)):
        metrics[name], frames[name] = evaluate_list(top, labels, sizes, groups, new)
    for segment in ("overall", "repeat", "new"):
        for key, expected in manifest["metrics"]["lambdarank"][segment].items():
            if not np.isclose(metrics["A"][segment][key], expected, rtol=0, atol=1e-12):
                raise ValueError(f"Strategy A differs from frozen metrics: {segment}/{key}")
    selectors = [("End-to-end NDCG@5, all groups", lambda m: m["overall"]["ndcg5_e2e_all_groups"]),
                 ("Precision@5, all groups", lambda m: m["precision5_all_groups"]),
                 ("Recall@5, micro", lambda m: m["overall"]["recall5_micro"]),
                 ("Repeat-product Recall@5, micro", lambda m: m["repeat"]["recall5_micro"]),
                 ("New-product Recall@5, micro", lambda m: m["new"]["recall5_micro"]),
                 ("New-product Top 5 slot share", lambda m: m["new_slot_share"])]
    comparison = []
    for name, select in selectors:
        av, bv = select(metrics["A"]), select(metrics["B"])
        comparison.append({"metric": name, "A": av, "B": bv, "delta": bv-av,
                           "relative_change": (bv-av)/av if av else None})
    transitions = {f"{old}->{added}": int(((labels[a[changed, 4]] == old) & (labels[b[changed, 4]] == added)).sum())
                   for old in (0, 1) for added in (0, 1)}
    delta = frames["B"].ndcg5_e2e - frames["A"].ndcg5_e2e
    result = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "promotion_status": "not promoted; awaiting user review", "metrics": metrics, "comparison": comparison,
              "affected_groups": int(changed.sum()), "affected_group_share": float(changed.mean()),
              "eligible_groups": int(eligible.sum()), "fallback_groups": int((~eligible).sum()),
              "changed_group_hit_transitions": transitions,
              "group_outcomes": {"improved": int((delta > 0).sum()), "regressed": int((delta < 0).sum()), "unchanged": int((delta == 0).sum())},
              "validation": {"rows": len(pred), "groups": len(groups), "verified_shard_hashes": checksum_count,
                             "cutoffs": DataConfig().cutoffs("validation")},
              "test_status": "sealed; no test access", "model_sha256": hashes["model.txt"],
              "prediction_sha256": hashes["validation_predictions.parquet"],
              "model_manifest_sha256": hashes["manifest.json"], "data_manifest_sha256": manifest["data_manifest_sha256"],
              "git_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "experiment_source_sha256": sha256(Path(__file__)), "elapsed_seconds": time.monotonic()-start}
    for name, digest in hashes.items():
        if sha256(model/name) != digest:
            raise ValueError(f"Model run changed: {name}")
    output.mkdir(parents=True)
    for name, top in (("A", a), ("B", b)):
        recommendations = pred.iloc[top.ravel()][ID_COLUMNS+["lambdarank_score", "repeat_product", "relevance"]].copy()
        recommendations["source_row_index"] = top.ravel()
        recommendations["rank"] = np.tile(np.arange(1, 6), len(groups))
        recommendations.to_parquet(output/f"recommendations_{name}.parquet", index=False)
        frames[name].to_parquet(output/f"group_metrics_{name}.parquet", index=False)
    result["files_sha256"] = {p.name: sha256(p) for p in output.iterdir()}
    write_json(output/"results.json", result)
    write_json(root/"reports/phase3_2_discovery.json", result)
    (root/"reports/phase3_2_discovery.md").write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"comparison": comparison, "affected_groups": result["affected_groups"],
                      "changed_group_hit_transitions": transitions}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/discovery-v1"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    run(root, root/args.output)


if __name__ == "__main__":
    main()
