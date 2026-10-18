"""Small historical-snapshot demo serving the unchanged Strategy A ranker."""
from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Path as ApiPath
import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import BaseModel

from .contracts import FEATURE_COLUMNS, ID_COLUMNS, feature_schema
from .pipeline import sha256


ROOT = Path(__file__).resolve().parents[2]


class ProductRecommendation(BaseModel):
    product_id: int
    score: float


class Recommendations(BaseModel):
    household_id: int
    snapshot_cutoff: int
    model_version: str
    recommendations: list[ProductRecommendation]


class Health(BaseModel):
    status: str
    model_version: str
    snapshot_cutoff: int
    households: int
    candidate_rows: int


class SnapshotRanker:
    def __init__(self, model_directory: Path, data_directory: Path, cutoff: int):
        model_manifest = json.loads((model_directory / "manifest.json").read_text())
        for name in ("model.txt", "feature_schema.json"):
            if sha256(model_directory / name) != model_manifest["files_sha256"][name]:
                raise ValueError(f"Frozen model artifact checksum mismatch: {name}")
        schema = json.loads((model_directory / "feature_schema.json").read_text())
        if schema != feature_schema():
            raise ValueError("Frozen feature schema mismatch")
        manifest = json.loads((data_directory / "manifest.json").read_text())
        if (manifest["status"] != "complete" or manifest["mode"] != "final_heldout"
                or manifest["schema"] != schema):
            raise ValueError("A complete compatible historical snapshot is required")
        shard = next((s for s in manifest["test"]["shards"] if s["cutoff"] == cutoff), None)
        if shard is None:
            raise ValueError("Snapshot cutoff is unavailable")
        audit = shard["diagnostics"]
        if audit["history_max_day"] > cutoff or not audit["history_invariance_passed"]:
            raise ValueError("Snapshot point-in-time verification is missing")
        file = shard["files"]["rows"]
        snapshot_path = (data_directory / file["path"]).resolve()
        if not snapshot_path.is_relative_to(data_directory.resolve()):
            raise ValueError("Snapshot path escapes the artifact directory")
        if sha256(snapshot_path) != file["sha256"]:
            raise ValueError("Historical snapshot checksum mismatch")
        # Project away relevance: serving never loads targets, labels or saved scores.
        rows = pd.read_parquet(snapshot_path, columns=ID_COLUMNS + FEATURE_COLUMNS)
        if len(rows) != file["rows"] or rows.empty or not rows.cutoff_day.eq(cutoff).all():
            raise ValueError("Snapshot size or cutoff mismatch")
        if any(rows[c].dtype != np.dtype("int64") for c in ID_COLUMNS):
            raise ValueError("Snapshot identifiers must be int64")
        if any(rows[c].dtype != np.dtype("float32") for c in FEATURE_COLUMNS):
            raise ValueError("Snapshot features must be float32")
        if rows.duplicated(ID_COLUMNS).any() or not np.isfinite(rows[FEATURE_COLUMNS].to_numpy()).all():
            raise ValueError("Duplicate candidates or nonfinite historical features")
        rows = rows.sort_values(ID_COLUMNS).reset_index(drop=True)
        households, offsets, counts = np.unique(rows.household_key, return_index=True, return_counts=True)
        if not np.all(counts == manifest["config"]["max_candidates"]):
            raise ValueError("Snapshot household candidate counts mismatch")
        self.model = lgb.Booster(model_file=str(model_directory / "model.txt"))
        if self.model.feature_name() != FEATURE_COLUMNS:
            raise ValueError("Model feature order differs from historical snapshot schema")
        self.rows = rows
        self.groups = {int(h): (int(o), int(n)) for h, o, n in zip(households, offsets, counts)}
        self.cutoff = cutoff
        self.version = model_directory.name

    def recommend(self, household_id: int) -> Recommendations:
        if household_id not in self.groups:
            raise HTTPException(status_code=404, detail="Household is absent from the historical snapshot")
        offset, count = self.groups[household_id]
        rows = self.rows.iloc[offset:offset + count]
        scores = self.model.predict(rows[FEATURE_COLUMNS].to_numpy(), num_threads=1)
        if not np.isfinite(scores).all():
            raise RuntimeError("Model returned nonfinite scores")
        ids = rows.PRODUCT_ID.to_numpy()
        top = np.lexsort((ids, -scores))[:5]
        return Recommendations(
            household_id=household_id, snapshot_cutoff=self.cutoff, model_version=self.version,
            recommendations=[ProductRecommendation(product_id=int(ids[i]), score=float(scores[i])) for i in top],
        )


def create_app(
    model_directory: Path = ROOT / "artifacts/models/lambdarank-v1",
    data_directory: Path = ROOT / "artifacts/data/retail-ranking-test-v1",
    cutoff: int = 701,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.ranker = SnapshotRanker(Path(model_directory), Path(data_directory), cutoff)
        yield
        del app.state.ranker

    app = FastAPI(title="Retail Recommendation Demo", version="0.1.0", lifespan=lifespan)

    @app.get("/health", response_model=Health)
    def health():
        ranker = app.state.ranker
        return Health(status="ok", model_version=ranker.version, snapshot_cutoff=ranker.cutoff,
                      households=len(ranker.groups), candidate_rows=len(ranker.rows))

    @app.get("/recommendations/{household_id}", response_model=Recommendations)
    def recommendations(household_id: Annotated[int, ApiPath(gt=0)]):
        return app.state.ranker.recommend(household_id)

    return app
