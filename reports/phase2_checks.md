# Phase 2 checkpoint validation record

Executed locally for the data implementation at commit `7de6844` on Python 3.12.15.
This record describes performed checks, not tests promised for later execution.

| Check | Actual outcome |
| --- | --- |
| `python -m pytest -q` | 34 passed in 1.63 seconds |
| `python scripts/smoke_environment.py` | All six native/library checks passed; see `environment_smoke.json` |
| `python -m pip check` | No broken requirements found |
| Interpreter/package isolation | Python prefix and pandas/LightGBM/SHAP/FastAPI imports resolve inside `.venv-phase2` |
| Phase 1 environment | Python 3.14.7, pandas 2.3.3 and pyarrow 24.0.0 still import successfully |
| Full data build | All 25 train and 12 validation snapshots completed in 277.969 seconds, excluding initial source load |
| Future-removal invariance | Exact candidates/features matched at cutoffs 365, 533, 540 and 617 |
| Artifact verification | 111 Parquet checksums and row counts passed; source/code hashes matched; all group offsets verified from disk |
| Held-out test | No test snapshots/labels/metrics generated; development loader limited to day 624 |

The 34 tests cover candidate selection, ties and quotas, 7/28/84-day history
boundaries, seven-day targets, future perturbation invariance, future-only products
and households, nonpositive quantities, schema/dtypes, malformed identifiers,
coverage denominators, group continuity/offsets, disk roundtrips, chronological
split boundaries, source completeness, overwrite refusal and portable locks.

Review corrections applied before the build:

- Added explicit serialized feature order so JSON key sorting cannot change model inputs.
- Rejected fractional/bool configuration and fractional identifiers instead of silently truncating.
- Made failed Git provenance queries explicit instead of claiming a clean working tree.
- Replaced Conda build-machine file URLs in pip freeze output with portable version pins;
  the separate Conda lock records the exact Windows runtime packages.

The Data Agent owned candidates/features and their tests. The ML Agent owned data
diagnostics and their tests and performed a read-only integration review. The
coordinator owned contracts, environment, build integration, reports and Git.
No Serving Agent was started and no model was fitted. Remaining work begins only
after review of this checkpoint: approve retrieval coverage, establish the
popularity baseline, then train/select LambdaRank using validation. Test-period
evaluation, SHAP/model integration, FastAPI and Docker follow their dependency gates.
