"""Integration checks use existing immutable artifacts; no synthetic fitting."""
import json

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from retail_recs.api import ROOT, SnapshotRanker, create_app, load_product_metadata
from retail_recs.contracts import FEATURE_COLUMNS


MODEL = ROOT / "artifacts/models/lambdarank-v1"
DATA = ROOT / "artifacts/data/retail-ranking-test-v1"


@pytest.fixture(scope="module")
def client():
    if not (MODEL / "model.txt").exists() or not (DATA / "manifest.json").exists():
        pytest.skip("Historical demo requires local frozen model and test snapshot artifacts")
    with TestClient(create_app()) as value:
        yield value


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == dict(status="ok", model_version="lambdarank-v1",
                                  snapshot_cutoff=701, households=2500, candidate_rows=500000)


def test_all_snapshot_scores_and_api_top5_match_saved_predictions(client):
    ranker = client.app.state.ranker
    saved = pd.read_parquet(ROOT / "artifacts/evaluations/heldout-v1/test_predictions.parquet",
                            filters=[("cutoff_day", "==", 701)])
    saved = saved.sort_values(["household_key", "cutoff_day", "PRODUCT_ID"]).reset_index(drop=True)
    assert np.array_equal(ranker.rows.PRODUCT_ID, saved.PRODUCT_ID)
    assert np.array_equal(ranker.rows.household_key, saved.household_key)
    scores = ranker.model.predict(ranker.rows[FEATURE_COLUMNS].to_numpy(), num_threads=1)
    np.testing.assert_array_equal(scores, saved.lambdarank_score)
    households = sorted(ranker.groups)
    for household in (households[0], households[len(households)//2], households[-1]):
        expected = saved.loc[saved.household_key == household].sort_values(
            ["lambdarank_score", "PRODUCT_ID"], ascending=[False, True]).head(5)
        response = client.get(f"/recommendations/{household}")
        assert response.status_code == 200
        body = response.json()
        assert body["household_id"] == household and body["snapshot_cutoff"] == 701
        assert [r["product_id"] for r in body["recommendations"]] == expected.PRODUCT_ID.tolist()
        np.testing.assert_array_equal([r["score"] for r in body["recommendations"]], expected.lambdarank_score)
        assert client.get(f"/recommendations/{household}").json() == body
    assert "relevance" not in ranker.rows
    assert "lambdarank_score" not in ranker.rows


def test_metadata_matches_original_product_table_without_changing_ranking(client):
    if not (ROOT / "data/processed/product.parquet").exists():
        pytest.skip("Optional original product metadata is absent")
    products = pd.read_parquet(ROOT / "data/processed/product.parquet").set_index("PRODUCT_ID")
    body = client.get("/recommendations/1").json()
    assert [p["product_id"] for p in body["recommendations"]] == [1082185, 1049998, 995242, 1005186, 1074612]
    for product in body["recommendations"]:
        source = products.loc[product["product_id"]]
        assert product["description"] == source.SUB_COMMODITY_DESC.strip()
        assert product["category"] == source.COMMODITY_DESC.strip()
        assert product["department"] == source.DEPARTMENT.strip()


def test_missing_or_blank_metadata_returns_nulls(tmp_path):
    assert load_product_metadata(None) == {}
    assert load_product_metadata(tmp_path / "missing.parquet") == {}
    path = tmp_path / "product.parquet"
    pd.DataFrame({"PRODUCT_ID": [1], "SUB_COMMODITY_DESC": [None],
                  "COMMODITY_DESC": [" "], "DEPARTMENT": [" GROCERY "]}).to_parquet(path)
    assert load_product_metadata(path) == {1: {"description": None, "category": None, "department": "GROCERY"}}
    with pytest.raises(ValueError, match="checksum"):
        load_product_metadata(path, "wrong-checksum")


def test_unknown_product_metadata_preserves_ids_and_scores(client, monkeypatch):
    original = client.get("/recommendations/1").json()
    monkeypatch.setattr(client.app.state, "product_metadata", {})
    result = client.get("/recommendations/1").json()
    assert [(p["product_id"], p["score"]) for p in result["recommendations"]] == [
        (p["product_id"], p["score"]) for p in original["recommendations"]]
    assert all(p["description"] is None and p["category"] is None for p in result["recommendations"])


@pytest.mark.parametrize("household,status", [(999999999, 404), (0, 422), (-1, 422), ("abc", 422)])
def test_invalid_or_unknown_households(client, household, status):
    assert client.get(f"/recommendations/{household}").status_code == status


def test_equal_scores_use_product_id_order(client, monkeypatch):
    ranker = client.app.state.ranker
    monkeypatch.setattr(ranker.model, "predict", lambda features, **kwargs: np.zeros(len(features)))
    household = min(ranker.groups)
    offset, count = ranker.groups[household]
    expected = sorted(ranker.rows.iloc[offset:offset+count].PRODUCT_ID)[:5]
    result = client.get(f"/recommendations/{household}").json()
    assert [r["product_id"] for r in result["recommendations"]] == expected


def test_corrupt_model_fails_startup(client, tmp_path):
    directory = tmp_path / "lambdarank-v1"
    directory.mkdir()
    (directory / "manifest.json").write_bytes((MODEL / "manifest.json").read_bytes())
    (directory / "model.txt").write_text("corrupt artifact")
    with pytest.raises(ValueError, match="checksum"):
        with TestClient(create_app(model_directory=directory)):
            pass


@pytest.mark.parametrize("corruption", ["hash", "path", "schema", "history", "cutoff"])
def test_snapshot_validation(client, tmp_path, corruption):
    manifest = json.loads((DATA / "manifest.json").read_text())
    shard = manifest["test"]["shards"][-1]
    if corruption == "hash":
        shard["files"]["rows"]["path"] = "rows.parquet"
        (tmp_path / "rows.parquet").write_bytes(b"corrupt snapshot")
    elif corruption == "path":
        # Resolving the original snapshot through an escaping path must fail.
        shard["files"]["rows"]["path"] = str(DATA / shard["files"]["rows"]["path"])
    elif corruption == "schema":
        manifest["schema"]["feature_order"].reverse()
    elif corruption == "history":
        shard["diagnostics"]["history_max_day"] = 702
    else:
        manifest["test"]["shards"] = []
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        SnapshotRanker(MODEL, tmp_path, 701)
