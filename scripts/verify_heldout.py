"""Independently verify final test labels, predictions and metrics; never fit."""
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/"artifacts/evaluations/heldout-v1"
    data = root/"artifacts/data/retail-ranking-test-v1"
    result = json.loads((out/"manifest.json").read_text())
    manifest = json.loads((data/"manifest.json").read_text())
    assert result["status"] == manifest["status"] == "complete"
    assert not (out/"INCOMPLETE").exists() and not (data/"INCOMPLETE").exists()
    assert result == json.loads((root/"reports/phase4_test.json").read_text())
    assert digest(data/"manifest.json") == result["test_manifest_sha256"]
    for name, value in result["files_sha256"].items():
        assert digest(out/name) == value
    for name, value in {**manifest["frozen_inputs_sha256"], **manifest["source_code_sha256"]}.items():
        assert digest(root/name) == value, name
    pred = pd.read_parquet(out/"test_predictions.parquet")
    model = lgb.Booster(model_file=str(root/"artifacts/models/lambdarank-v1/model.txt"))
    features = manifest["schema"]["feature_order"]
    assert features == model.feature_name() and len(features) == 25 and model.num_trees() == 152
    tx = pd.read_parquet(root/"data/processed/transaction_data.parquet",
                         columns=["household_key", "PRODUCT_ID", "DAY", "QUANTITY"], filters=[("DAY", "<=", 708)])
    assert tx.DAY.max() == 708
    tx = tx.loc[tx.QUANTITY > 0]
    keys = ["household_key", "cutoff_day"]; pair = ["household_key", "PRODUCT_ID"]
    ids = keys+["PRODUCT_ID"]
    first = tx.groupby(pair).DAY.min()
    group_frames = []; position = 0; checked = 0
    assert manifest["test"]["cutoffs"] == list(range(624, 702, 7))
    for shard in manifest["test"]["shards"]:
        cutoff = shard["cutoff"]
        for file in shard["files"].values():
            assert digest(data/file["path"]) == file["sha256"]
            checked += 1
        rows = pd.read_parquet(data/shard["files"]["rows"]["path"])
        targets = pd.read_parquet(data/shard["files"]["targets"]["path"])
        saved_groups = pd.read_parquet(data/shard["files"]["groups"]["path"])
        history = tx.loc[tx.DAY <= cutoff]
        eligible = np.sort(history.household_key.unique())
        future = tx.loc[(tx.DAY > cutoff) & (tx.DAY <= cutoff+7) & tx.household_key.isin(eligible)]
        expected = future[pair].drop_duplicates().sort_values(pair).reset_index(drop=True)
        np.testing.assert_array_equal(expected, targets[pair])
        is_repeat_target = first.reindex(pd.MultiIndex.from_frame(expected)).to_numpy() <= cutoff
        np.testing.assert_array_equal(is_repeat_target, targets.repeat_product)
        target_pairs = pd.MultiIndex.from_frame(expected)
        y = pd.MultiIndex.from_frame(rows[pair]).isin(target_pairs)
        np.testing.assert_array_equal(y, rows.relevance)
        np.testing.assert_array_equal(np.sort(rows.household_key.unique()), eligible)
        assert rows.PRODUCT_ID.isin(history.PRODUCT_ID).all()
        segment = first.reindex(pd.MultiIndex.from_frame(rows[pair])).to_numpy() <= cutoff
        np.testing.assert_array_equal(segment, rows.pair_recency_days <= cutoff)
        assert list(rows.columns) == ids+features+["relevance"]
        chunk = pred.iloc[position:position+len(rows)]
        for col in ids+["relevance"]:
            np.testing.assert_array_equal(rows[col], chunk[col])
        np.testing.assert_array_equal(segment, chunk.repeat_product)
        np.testing.assert_array_equal(rows.product_households_28d, chunk.popularity_score)
        np.testing.assert_array_equal(model.predict(rows[features].to_numpy(dtype=np.float32), num_threads=4), chunk.lambdarank_score)
        g = rows.groupby(keys, sort=False).size().rename("group_size").reset_index()
        g["target_count"] = g.household_key.map(expected.groupby("household_key").size()).fillna(0).astype(int)
        g["repeat_target_count"] = g.household_key.map(expected.loc[is_repeat_target].groupby("household_key").size()).fillna(0).astype(int)
        g["new_target_count"] = g.target_count-g.repeat_target_count
        for col in g.columns:
            np.testing.assert_array_equal(g[col], saved_groups[col])
        np.testing.assert_array_equal(saved_groups.row_offset, np.r_[0, g.group_size.cumsum().to_numpy()[:-1]])
        group_frames.append(g); position += len(rows)
        print(f"independently verified labels, segments and model scores cutoff={cutoff}", flush=True)
    assert position == len(pred)
    groups = pd.concat(group_frames, ignore_index=True)
    gi = pd.MultiIndex.from_frame(groups[keys])
    discounts = 1/np.log2(np.arange(2,7)); ideal = np.r_[0, discounts.cumsum()]
    verification = {}
    for name in ("popularity", "lambdarank"):
        top = pred.sort_values(keys+[name+"_score", "PRODUCT_ID"], ascending=[True,True,False,True]).groupby(keys,sort=False).head(5).copy()
        top["rank"] = top.groupby(keys,sort=False).cumcount()+1
        top["dcg"] = top.relevance/np.log2(top["rank"]+1)
        per_group = top.groupby(keys)[["relevance","dcg"]].sum().reindex(gi,fill_value=0)
        denom = ideal[np.minimum(groups.target_count.to_numpy(),5)]
        ndcg = np.divide(per_group.dcg.to_numpy(),denom,out=np.zeros(len(groups)),where=denom>0)
        metrics = {"End-to-end NDCG@5":float(ndcg.mean()), "Precision@5":float(top.relevance.sum()/(5*len(groups))),
                   "Recall@5 micro":float(top.relevance.sum()/groups.target_count.sum()),
                   "Candidate recall micro":float(pred.relevance.sum()/groups.target_count.sum()),
                   "Repeat Recall@5 micro":float(top.loc[top.repeat_product,"relevance"].sum()/groups.repeat_target_count.sum()),
                   "New Recall@5 micro":float(top.loc[~top.repeat_product,"relevance"].sum()/groups.new_target_count.sum())}
        for record in result["comparison"]:
            assert np.isclose(metrics[record["metric"]],record[name]["test"],rtol=0,atol=1e-12)
        saved = pd.read_parquet(out/"test_group_metrics.parquet")
        np.testing.assert_allclose(ndcg,saved[name+"_ndcg5_e2e"],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(per_group.relevance,saved[name+"_hits_at_5"])
        verification[name] = metrics
    record = {"status":"passed", "independent_method":"Raw transaction window labels and first-purchase segmentation; pandas sorting and direct metric formulas; no production evaluator imports",
              "shard_hashes_verified":checked, "candidate_rows":len(pred), "groups":len(groups),
              "frozen_input_hashes_verified":len(manifest["frozen_inputs_sha256"]),
              "all_saved_scores_match_reloaded_model":True, "absolute_metric_tolerance":1e-12,
              "verification_source_sha256":digest(Path(__file__)), "metrics":verification}
    (root/"reports/phase4_verification.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(record,indent=2),flush=True)


if __name__ == "__main__":
    main()
