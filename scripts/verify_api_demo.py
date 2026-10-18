"""Compare a running demo (local or Docker) with the unchanged local API."""
import argparse
import json
import math
from pathlib import Path
import sys
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from retail_recs.api import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--household-id", type=int, default=1)
    args = parser.parse_args()
    if args.household_id <= 0:
        parser.error("--household-id must be positive")
    base = args.url.rstrip("/")
    with urlopen(base + "/docs", timeout=30) as response:
        if response.status != 200 or "swagger-ui" not in response.read().decode():
            raise ValueError("Interactive API documentation is unavailable")
    with urlopen(base + "/health", timeout=30) as response:
        health = json.load(response)
    path = f"/recommendations/{args.household_id}"
    with urlopen(base + path, timeout=30) as response:
        actual = json.load(response)
    with TestClient(create_app()) as local:
        if health != local.get("/health").json():
            raise ValueError("Running API health differs from local API")
        response = local.get(path)
        response.raise_for_status()
        expected = response.json()
    for key in ("household_id", "snapshot_cutoff", "model_version"):
        if actual[key] != expected[key]:
            raise ValueError(f"API metadata differs: {key}")
    got, want = actual["recommendations"], expected["recommendations"]
    if len(got) != 5 or [p["product_id"] for p in got] != [p["product_id"] for p in want]:
        raise ValueError("Top 5 product IDs or ordering differ from local API")
    differences = [abs(a["score"] - b["score"]) for a, b in zip(got, want)]
    if any(not math.isclose(a["score"], b["score"], rel_tol=0, abs_tol=1e-12)
           for a, b in zip(got, want)):
        raise ValueError("Scores differ from local API by more than 1e-12")
    print(json.dumps({"status": "passed", "url": base, "health": health,
                      "recommendations": actual, "maximum_score_difference": max(differences),
                      "score_absolute_tolerance": 1e-12}, indent=2))


if __name__ == "__main__":
    main()
