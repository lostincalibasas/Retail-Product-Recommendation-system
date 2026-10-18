"""Diagnose a frozen ranker on validation only; no fitting or feature changes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .contracts import DataConfig, FEATURE_COLUMNS, ID_COLUMNS, feature_schema
from .pipeline import sha256, write_json
from .ranking_metrics import top_indices
from .training_data import verify_shard


def describe(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite diagnostic values")
    quantiles = np.quantile(values, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    return {"count": len(values), "mean": float(values.mean()), "std": float(values.std()),
            **dict(zip(["min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"],
                       map(float, quantiles)))}


def rank_positions(scores, sizes, product_ids, is_new):
    """Full and new-only ranks, using the original deterministic tie rule."""
    sizes = np.asarray(sizes)
    if not len(sizes) or not np.all(sizes == sizes[0]):
        raise ValueError("Frozen v1 diagnostics require uniform nonempty groups")
    order = top_indices(scores, sizes, product_ids, k=int(sizes[0]))
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(1, int(sizes[0]) + 1)
    new_ranks = np.zeros(len(scores), dtype=np.int32)
    new_in_order = np.asarray(is_new)[order]
    new_ranks[order] = np.cumsum(new_in_order, axis=1) * new_in_order
    return ranks, new_ranks, order[:, :5]


def exposure(top, is_new, labels, groups):
    new = np.asarray(is_new)[top]
    hits = np.asarray(labels)[top] * new
    n = new.sum(axis=1)
    result = {"slots": int(new.size), "new_slots": int(new.sum()),
              "new_slot_share": float(new.mean()), "groups": len(top),
              "groups_with_new": int((n > 0).sum()), "group_exposure_rate": float((n > 0).mean()),
              "new_count_histogram": {str(i): int((n == i).sum()) for i in range(6)},
              "new_hits": int(hits.sum()),
              "new_slot_hit_rate": float(hits.sum()/new.sum()) if new.sum() else 0.0}
    for name, mask in {"purchase_groups": groups.target_count.to_numpy() > 0,
                       "new_purchase_groups": groups.new_target_count.to_numpy() > 0}.items():
        result[name] = {"groups": int(mask.sum()), "groups_with_new": int((n[mask] > 0).sum()),
                        "new_slots": int(n[mask].sum()),
                        "new_slot_share": float(new[mask].mean()) if mask.any() else 0.0}
    return result


def tree_gap(tree, first, second, feature_names):
    """Exact leaf-score gap assigned to the first divergent split, not causal attribution.

    Rows must be observed finite numeric features. Interactions below the split
    remain in its leaf-score difference; no additive feature-effect claim is made.
    """
    def branch(node, row):
        if node["decision_type"] != "<=" or not np.isfinite(row[node["split_feature"]]):
            raise ValueError("Only finite numeric splits are supported")
        if node.get("missing_type") == "Zero":
            raise ValueError("Zero-as-missing splits are not supported")
        return "left_child" if row[node["split_feature"]] <= float(node["threshold"]) else "right_child"

    def leaf(node, row):
        while "split_feature" in node:
            node = node[branch(node, row)]
        return float(node["leaf_value"])

    node = tree
    while "split_feature" in node:
        a, b = branch(node, first), branch(node, second)
        if a != b:
            return feature_names[node["split_feature"]], leaf(node[b], second) - leaf(node[a], first)
        node = node[a]
    return None, 0.0


def paired_score_gaps(trees, first, second, feature_names):
    by_feature = defaultdict(float)
    split_counts = Counter()
    for tree in trees:
        name, gap = tree_gap(tree["tree_structure"], first, second, feature_names)
        if name is not None:
            by_feature[name] += gap
            split_counts[name] += 1
    return dict(by_feature), dict(split_counts)


def within_group_ordering(scores, is_new, sizes):
    """Pair-weighted P(repeat > new), with half credit for exact ties."""
    wins, ties, pairs = 0, 0, 0
    means, best_gaps = [], []
    offset = 0
    for size in sizes:
        s, new = scores[offset:offset+size], is_new[offset:offset+size]
        a, b = s[~new], np.sort(s[new])
        if len(a) and len(b):
            lower = np.searchsorted(b, a, side="left")
            upper = np.searchsorted(b, a, side="right")
            wins += int(lower.sum()); ties += int((upper-lower).sum())
            pairs += len(a)*len(b)
            means.append(a.mean()-b.mean())
            best_gaps.append(b[-1]-np.sort(s)[-5])
        offset += size
    return {"repeat_new_pairs": pairs, "repeat_strictly_above_new": wins,
            "tied_pairs": ties, "prob_repeat_above_new_half_ties": (wins+.5*ties)/pairs if pairs else None,
            "mean_repeat_minus_mean_new": describe(means),
            "best_new_minus_fifth_score": describe(best_gaps)}


def run(root, model_directory, output):
    start = time.monotonic()
    root, model_directory, output = root.resolve(), model_directory.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostics: {output}")
    manifest_path = model_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete" or (model_directory / "INCOMPLETE").exists():
        raise ValueError("A complete model run is required")
    immutable = {"manifest.json": sha256(manifest_path), **manifest["files_sha256"]}
    for name, digest in immutable.items():
        if sha256(model_directory/name) != digest:
            raise ValueError(f"Model artifact checksum mismatch: {name}")
    data = root / "artifacts/data/retail-ranking-v1"
    if sha256(data/"manifest.json") != manifest["data_manifest_sha256"]:
        raise ValueError("Data manifest differs from trained model")
    dm = json.loads((data/"manifest.json").read_text())
    cutoffs = DataConfig().cutoffs("validation")
    if dm["schema"] != feature_schema() or dm["splits"]["validation"]["cutoffs"] != cutoffs:
        raise ValueError("Frozen validation schema/cutoffs mismatch")
    if [s["cutoff"] for s in dm["splits"]["validation"]["shards"]] != cutoffs:
        raise ValueError("Unexpected validation shard sequence")
    model = lgb.Booster(model_file=str(model_directory/"model.txt"))
    if model.feature_name() != FEATURE_COLUMNS or model.num_trees() != manifest["best_iteration"]:
        raise ValueError("Saved model schema/iteration mismatch")
    pred = pd.read_parquet(model_directory/"validation_predictions.parquet")
    groups = pd.read_parquet(model_directory/"validation_group_metrics.parquet")
    sizes = groups.group_size.to_numpy(dtype=np.int64)
    if sizes.sum() != len(pred) or not np.array_equal(groups.row_offset, np.r_[0, sizes.cumsum()[:-1]]):
        raise ValueError("Saved group offsets/sizes do not align")
    ids = pred.PRODUCT_ID.to_numpy()
    is_new = ~pred.repeat_product.to_numpy(dtype=bool)
    labels = pred.relevance.to_numpy()
    ranks, new_ranks, tops = {}, {}, {}
    for name in ("popularity", "lambdarank"):
        ranks[name], new_ranks[name], tops[name] = rank_positions(pred[name+"_score"].to_numpy(), sizes, ids, is_new)
    new_positive = is_new & (labels == 1)
    base_hit = new_positive & (ranks["popularity"] <= 5)
    rank_hit = new_positive & (ranks["lambdarank"] <= 5)
    lost = np.flatnonzero(base_hit & ~rank_hit)
    row_groups = np.repeat(np.arange(len(groups)), sizes)
    fifth = tops["lambdarank"][row_groups[lost], 4]
    needed = np.unique(np.r_[lost, fifth])
    selected_features = {}
    feature_summaries = {"repeat": [], "new": []}
    offset, group_offset, checksums = 0, 0, 0
    for shard in dm["splits"]["validation"]["shards"]:
        loaded = {}
        for kind in ("rows", "groups", "targets"):
            record = shard["files"][kind]
            path = (data/record["path"]).resolve()
            if not path.is_relative_to((data/"validation").resolve()) or sha256(path) != record["sha256"]:
                raise ValueError("Validation shard path/checksum mismatch")
            loaded[kind] = pd.read_parquet(path)
            checksums += 1
        rows, target, group = loaded["rows"], loaded["targets"], loaded["groups"]
        verify_shard(rows, target, group, shard["cutoff"])
        n = len(rows); part = pred.iloc[offset:offset+n]
        for col in ID_COLUMNS+["relevance"]:
            np.testing.assert_array_equal(rows[col], part[col])
        for col in ("household_key", "cutoff_day", "group_size", "target_count", "positive_labels"):
            np.testing.assert_array_equal(group[col], groups.iloc[group_offset:group_offset+len(group)][col])
        np.testing.assert_array_equal(rows.product_households_28d, part.popularity_score)
        np.testing.assert_array_equal(rows.pair_recency_days <= shard["cutoff"], part.repeat_product)
        X = rows[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        np.testing.assert_array_equal(model.predict(X, num_threads=4), part.lambdarank_score)
        subset = needed[(needed >= offset) & (needed < offset+n)]
        selected_features.update({int(i): X[i-offset].copy() for i in subset})
        # These sums describe the actual feature encoding, not importance attribution.
        for name, mask in (("new", is_new[offset:offset+n]), ("repeat", ~is_new[offset:offset+n])):
            feature_summaries[name].append((int(mask.sum()), X[mask].sum(axis=0, dtype=np.float64)))
        offset += n; group_offset += len(group)
        print(f"verified validation cutoff={shard['cutoff']} rows={n:,}", flush=True)
    if offset != len(pred) or group_offset != len(groups):
        raise ValueError("Saved predictions contain rows outside validation shards")
    gain, splits = model.feature_importance("gain"), model.feature_importance("split")
    importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "gain": gain,
                               "gain_share": gain/gain.sum(), "split_count": splits}).sort_values("gain", ascending=False)
    trees = model.dump_model()["tree_info"]
    root_counts = Counter(FEATURE_COLUMNS[t["tree_structure"]["split_feature"]] for t in trees)
    interactions = Counter()
    def visit(node, path):
        if "split_feature" not in node:
            return
        name = FEATURE_COLUMNS[node["split_feature"]]
        if name.startswith("hh_") and any(p.startswith("pair_") for p in path):
            interactions["household_splits_below_pair_history"] += 1
        if name == "category_affinity_84d" and any(p.startswith("pair_") for p in path):
            interactions["category_splits_below_pair_history"] += 1
        for child in ("left_child", "right_child"):
            visit(node[child], path+[name])
    for tree in trees:
        visit(tree["tree_structure"], [])
    cases = pred.iloc[lost][ID_COLUMNS].copy()
    cases["popularity_rank"] = ranks["popularity"][lost]
    cases["lambdarank_rank"] = ranks["lambdarank"][lost]
    cases["lambdarank_new_only_rank"] = new_ranks["lambdarank"][lost]
    cases["repeat_candidates_above"] = ranks["lambdarank"][lost]-new_ranks["lambdarank"][lost]
    cases["popularity_score"] = pred.popularity_score.to_numpy()[lost]
    cases["lambdarank_score"] = pred.lambdarank_score.to_numpy()[lost]
    cases["fifth_product_id"] = ids[fifth]
    cases["fifth_is_repeat"] = ~is_new[fifth]
    cases["fifth_score"] = pred.lambdarank_score.to_numpy()[fifth]
    cases["score_gap_to_fifth"] = cases.fifth_score - cases.lambdarank_score
    gap_sum, divergence_count, case_gaps = defaultdict(float), Counter(), {}
    for a, b in zip(lost, fifth):
        gaps, counts = paired_score_gaps(trees, selected_features[a], selected_features[b], FEATURE_COLUMNS)
        expected = pred.lambdarank_score.iloc[b]-pred.lambdarank_score.iloc[a]
        if not np.isclose(sum(gaps.values()), expected, rtol=0, atol=1e-10):
            raise ValueError("Tree traversal does not reproduce saved score gap")
        for feature, value in gaps.items():
            gap_sum[feature] += value
        divergence_count.update(counts)
        case_gaps[int(a)] = gaps
    # Fixed rank quantiles, then distinct household/cutoff queries, provide examples
    # across the lost-hit population rather than hand-picking favorable narratives.
    ordered_cases = cases.sort_values(["lambdarank_rank", *ID_COLUMNS])
    example_indices, seen = [], set()
    for q in (0, .25, .5, .75, 1):
        anchor = int(round(q * max(0, len(ordered_cases)-1)))
        for pos in sorted(range(len(ordered_cases)), key=lambda j: (abs(j-anchor), j)):
            i = int(ordered_cases.index[pos])
            key = tuple(pred.loc[i, ["household_key", "cutoff_day"]])
            if key not in seen:
                example_indices.append(i); seen.add(key); break
    examples = []
    for i in example_indices:
        group_id = row_groups[i]
        record = cases.loc[i].to_dict()
        record["features"] = dict(zip(FEATURE_COLUMNS, map(float, selected_features[i])))
        record["first_divergence_score_gap"] = case_gaps[i]
        record["fifth_features"] = dict(zip(FEATURE_COLUMNS, map(float, selected_features[tops["lambdarank"][group_id, 4]])))
        for name in tops:
            top = pred.iloc[tops[name][group_id]][ID_COLUMNS+["repeat_product", "relevance", "popularity_score", "lambdarank_score"]].copy()
            record[name+"_top5"] = top.to_dict(orient="records")
        examples.append(record)
    score_summaries = {}
    for name in tops:
        score = pred[name+"_score"].to_numpy()
        score_summaries[name] = {segment: describe(score[mask]) for segment, mask in {
            "repeat": ~is_new, "new": is_new,
            "purchased_repeat": ~is_new & (labels == 1), "purchased_new": new_positive,
            "unpurchased_repeat": ~is_new & (labels == 0), "unpurchased_new": is_new & (labels == 0)}.items()}
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "scope": "saved model and validation only; no fitting, model edits or candidate changes",
        "test_status": "sealed; no test targets or raw transactions accessed",
        "model_sha256": immutable["model.txt"], "model_manifest_sha256": immutable["manifest.json"],
        "data_manifest_sha256": manifest["data_manifest_sha256"],
        "git_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "diagnostic_source_sha256": sha256(Path(__file__)),
        "feature_order": FEATURE_COLUMNS, "trees": model.num_trees(),
        "validation_rows": len(pred), "validation_groups": len(groups),
        "verified_validation_shard_hashes": checksums, "all_saved_scores_reproduced_exactly": True,
        "importance": importance.to_dict(orient="records"), "root_split_counts": dict(root_counts),
        "tree_structure_interactions": dict(interactions),
        "candidate_populations": {name: {"rows": int(mask.sum()), "positives": int(labels[mask].sum()),
            "positive_rate": float(labels[mask].mean()), "mean_features": dict(zip(FEATURE_COLUMNS,
            map(float, sum(s for _, s in feature_summaries[name])/sum(n for n, _ in feature_summaries[name]))))}
            for name, mask in (("repeat", ~is_new), ("new", is_new))},
        "score_distributions": score_summaries,
        "within_group_ordering": {name: within_group_ordering(pred[name+"_score"].to_numpy(), is_new, sizes) for name in tops},
        "top5_exposure": {name: exposure(tops[name], is_new, labels, groups) for name in tops},
        "new_positive_ranks": {name: {"overall": describe(ranks[name][new_positive]),
            "within_new_only": describe(new_ranks[name][new_positive]),
            "new_only_top5_hits": int((new_ranks[name][new_positive] <= 5).sum())} for name in tops},
        "new_hit_transition": {"baseline": int(base_hit.sum()), "lambdarank": int(rank_hit.sum()),
            "shared": int((base_hit & rank_hit).sum()), "lost": len(lost), "gained": int((~base_hit & rank_hit).sum()),
            "lost_but_new_only_rank_le5": int((new_ranks["lambdarank"][lost] <= 5).sum()),
            "lost_with_repeat_at_fifth": int((~is_new[fifth]).sum()),
            "lost_at_equal_fifth_score": int((cases.score_gap_to_fifth == 0).sum()),
            "lost_rank_distribution": describe(ranks["lambdarank"][lost]),
            "lost_repeat_candidates_above": describe(cases.repeat_candidates_above),
            "lost_score_gap_to_fifth": describe(cases.score_gap_to_fifth)},
        "lost_gap_decomposition": {"total_score_gap": float(cases.score_gap_to_fifth.sum()),
            "signed_gap_by_first_divergence_feature": dict(sorted(gap_sum.items(), key=lambda p: -p[1])),
            "first_divergence_tree_counts": dict(divergence_count),
            "all_case_gaps_match_saved_scores": True,
            "interpretation": "Exact partition by first divergent tree split, including all downstream interactions; NOT causal or additive feature importance."},
        "example_selection": "lost-hit LambdaRank-rank quantiles 0/25/50/75/100%, nearest distinct household/cutoff",
        "examples": examples,
        "limitations": ["Gain/split importance describes training trees, not causal effects or held-out permutation importance.",
            "Pooled ranker scores are not calibrated probabilities or comparable household utility scales; within-query comparisons also reported.",
            "New-only ranking is a conditional diagnostic, not a proposed production policy or a separately trained model.",
            "Validation was used for early stopping; segment findings require future independent validation after policy selection."],
        "elapsed_seconds": time.monotonic()-start,
    }
    # Verify that every model-run file is still byte-identical before writing new outputs.
    for name, digest in immutable.items():
        if sha256(model_directory/name) != digest:
            raise ValueError(f"Frozen artifact changed during diagnostics: {name}")
    output.mkdir(parents=True)
    importance.to_csv(output/"feature_importance.csv", index=False)
    cases.to_csv(output/"lost_new_hits.csv", index=False)
    write_json(output/"diagnostics.json", result)
    write_json(root/"reports/phase3_1_diagnostics.json", result)
    print(json.dumps({"top5_exposure": result["top5_exposure"], "new_hit_transition": result["new_hit_transition"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/lambdarank-v1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/diagnostics/lambdarank-v1"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    run(root, root/args.model, root/args.output)


if __name__ == "__main__":
    main()
