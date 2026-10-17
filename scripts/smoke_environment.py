"""Exercise native dependencies without fitting a model or starting a service."""
import importlib.metadata
import json
from pathlib import Path
import platform

import fastapi
import lightgbm as lgb
import numba
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shap
from sklearn.metrics import ndcg_score


def main():
    assert platform.python_version_tuple()[:2] == ("3", "12")
    # Build a native LightGBM Dataset, without running a boosting iteration.
    features = np.arange(24, dtype=np.float32).reshape(12, 2)
    dataset = lgb.Dataset(features, label=np.tile([0, 1, 0], 4), group=[3] * 4,
                          params={"min_data_in_bin": 1, "min_data_in_leaf": 1, "verbosity": -1})
    dataset.construct()
    assert dataset.num_data() == 12
    assert ndcg_score([[1, 0, 0]], [[3, 2, 1]], k=5) == 1.0

    @numba.njit
    def total(values):
        return values.sum()

    assert total(np.array([1.0, 2.0])) == 3.0
    explanation = shap.Explanation(values=np.array([[0.2, -0.1]]), base_values=np.array([0.5]))
    assert explanation.values.shape == (1, 2)
    table = pa.Table.from_pandas(pd.DataFrame({"value": [1, 2]}), preserve_index=False)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    assert pq.read_table(pa.BufferReader(sink.getvalue())).equals(table)
    assert fastapi.FastAPI().openapi()["info"]["title"] == "FastAPI"
    result = {"status": "passed", "python": platform.python_version(), "platform": platform.platform(),
              "checks": ["LightGBM native dataset construction (no training)", "scikit-learn NDCG",
                         "Numba JIT", "SHAP Explanation", "Arrow in-memory Parquet roundtrip", "FastAPI OpenAPI"],
              "packages": {name: importlib.metadata.version(name) for name in
                           ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "lightgbm", "shap", "numba", "fastapi", "pydantic", "pytest"]}}
    root = Path(__file__).resolve().parents[1]
    (root / "reports/environment_smoke.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
