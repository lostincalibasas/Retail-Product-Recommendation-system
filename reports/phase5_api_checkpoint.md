# FastAPI historical demo checkpoint

The minimal API serves frozen `lambdarank-v1` Strategy A recommendations from the
existing **cutoff-701** snapshot: **2,500 households, 200 candidates each, 500,000
candidate rows**. Responses contain product IDs, ranking scores, model version,
household ID and snapshot cutoff. This is a historical demo of recommendations
for days 702-708, not a live-data service.

## Implementation

- `src/retail_recs/api.py`: application factory, lifespan model/snapshot loading,
  typed responses, `/health` and `/recommendations/{household_id}`.
- `tests/test_api.py`: artifact-backed endpoint, score parity, deterministic ties,
  request validation and startup rejection checks.
- `pyproject.toml`: optional `api` dependencies using versions already installed.
- `README.md`: startup, endpoints, local-artifact requirements and limitations.
- The completed, previously untracked `phase4_repeat_diagnostic.md` and `.json`
  are preserved and included in this local checkpoint.

Startup verifies model and feature-schema hashes against the saved model manifest,
the snapshot checksum against its data manifest, compatible ordered schema,
recorded historical-invariance verification and historical maximum day. It checks
cutoff, row count, identifier/feature dtypes, feature finiteness, unique candidates
and per-household candidate counts. Missing/inconsistent artifacts fail startup.
Only identifiers and the 25 historical features are read from Parquet. Serving
does not open target files or load relevance labels or saved predictions.

The model is loaded once. A household-to-row-offset index locates its 200
candidates; the request runs the unchanged model with explicit feature ordering
and returns descending scores, ascending product ID for ties. There is no discovery
rule or repeat-only filter. Scores are uncalibrated ranking scores. Unknown
households return 404; nonpositive/noninteger IDs return 422. `/health` is ready
only after startup completes and reports the loaded snapshot metadata.

## Executed verification

```powershell
.\.venv-phase2\python.exe -B -m pytest -q -p no:cacheprovider tests/test_api.py
.\.venv-phase2\python.exe -B -m pytest -q -p no:cacheprovider -k 'not fit_save_reload'
```

- Focused API tests: **13 passed in 8.26 seconds**.
- Regression suite: **70 passed, 1 deselected in 10.69 seconds**. The existing
  synthetic fitting test was excluded to keep this checkpoint inference-only.
- **All 500,000 cutoff-701 scores match the original saved Phase 4 scores exactly**,
  using array equality rather than a numeric tolerance.
- API Top 5 IDs, ordering and scores match saved predictions for the first,
  middle and last household in the snapshot (IDs 1, 1251 and 2500). Repeated
  requests match. Tests also cover score ties, unknown/invalid IDs, corrupt model,
  corrupt snapshot, escaping snapshot path, incompatible schema, future history
  metadata and unavailable cutoff.
- A real local Uvicorn process was started on `127.0.0.1:8765`; both endpoints
  returned HTTP 200. Household 1 returned product IDs **1082185, 1049998, 995242,
  1005186, 1074612**, led by score **1.3464024710989912**. The smoke-test server
  was stopped afterward.
- All **21 frozen input hashes** in the Phase 4 data manifest still match.
  Original training, feature/candidate generation, model/schema/configuration,
  development artifacts and original Phase 4 reports were not modified.
- `git diff --check` passed. No installation, retraining, tuning or push occurred.

The test client emits one upstream Starlette/AnyIO deprecation warning concerning
`BlockingPortal`; tests and the real Uvicorn smoke check succeed. Dependency
versions were not changed to address this warning.

## Run locally

```powershell
.\.venv-phase2\python.exe -m uvicorn retail_recs.api:create_app --factory --app-dir src --host 127.0.0.1 --port 8000
```

Visit `/docs`, `/health` or `/recommendations/1`. No environment variables, database,
external services or feature recomputation are required. Large artifacts stay
ignored by Git and must already exist locally. API integration tests skip if the
local model/data artifacts are missing; saved Phase 4 predictions are additionally
required for the score-parity test.

**80/20:** a verified historical snapshot supplies the exact model inputs, the
frozen ranker supplies the ordering, and FastAPI exposes that ordering. Keeping a
single cutoff avoids accidentally mixing feature dates or presenting historical
scores as live recommendations. The demo serves existing households only and
loads one snapshot into process memory. It has no automatic refresh, live catalog,
authentication or availability filtering. Docker, SHAP and other infrastructure
remain outside this checkpoint.

Base Git checkpoint: `57e6647` on `main`. This report is committed with the API;
the resulting commit ID is available from `git log -1`. The already-opened test
period is used only for prediction parity, with no new model-selection claim.

Frozen model SHA-256:
`af9492f58af0b891b6f766213a5eb823bbb7fbf46aeb1719aba6c406515219ec`.

Feature-schema SHA-256:
`2d319c1da73dd254e0e609c3b76f1982bef4b0a7b153dec17ac2630305adde8f`.

Cutoff-701 rows SHA-256:
`29cd69c658c13a4ada8199ad2fe2f53be2a220b141d8855935109cf625c89a17`.
