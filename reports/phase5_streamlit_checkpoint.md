# Streamlit interview demo checkpoint

Date: 2026-10-19 (Asia/Dubai). Base checkpoint: `cd91b2c` on `main`, initially clean.

The interview frontend now calls the existing FastAPI service and presents five
ordered retail product cards. The frozen 152-tree LambdaRank Strategy A, 25-feature
schema, cutoff-701 candidates, model scores, labels and evaluation results remain
unchanged. No training, tuning, new ranking policy, Docker work or push occurred.

## Scope and files

- `streamlit_app.py`: household ID input and submit button, five numbered cards,
  historical-demo explanation, expandable model/cutoff/product-ID/raw-score
  details, graceful errors and clearing previous results after failed requests.
- `.streamlit/config.toml`: restrained light retail theme and disabled usage stats.
- `src/retail_recs/demo_client.py`: HTTP-only client, 15-second timeout, response
  validation and actionable connection/unknown-household/service error messages.
- `src/retail_recs/api_models.py`: shared response types without ML imports.
- `src/retail_recs/api.py`: optional display metadata loaded once at startup and
  appended to existing recommendations after scoring and Top 5 selection.
- `tests/test_frontend.py`, `tests/test_api.py`: UI/client and metadata verification,
  while preserving the original exact 500,000-score consistency check.
- `requirements-ui.txt`, `pyproject.toml`: optional pinned frontend dependencies.
  `README.md`: exact local startup and testing commands.

The frontend reads no model, catalog or feature artifacts and performs no ranking.
The Streamlit server calls FastAPI over HTTP. Products stay in the API's original
order; no score transformation, probability interpretation, repeat-only filter or
client-side sorting is introduced. Model information and raw scores appear only
in the expandable technical details. No images, invented names, prices or charts
are used. User/catalog strings in HTML cards are escaped.

## Catalog truth and API compatibility

The existing `data/processed/product.parquet` contains **92,353 products**. It has
department/category/subcategory descriptions, but no commercial product-name or
price column. The API adds nullable fields:

| API field | Existing source column |
|---|---|
| `description` | `SUB_COMMODITY_DESC` |
| `category` | `COMMODITY_DESC` |
| `department` | `DEPARTMENT` |

Values are the actual catalog strings with surrounding whitespace removed.
Blank/missing values become null. The UI labels them as unavailable. Product IDs
and raw scores remain present and unchanged; existing endpoints and health fields
are retained. Unknown catalog products do not disappear from recommendations.

When the original catalog exists, startup checks its SHA-256 against the frozen
Phase 4 input manifest. This is a display-only lookup from the same static catalog
already used by the project, not a new feature or a future-transaction query. The
existing assumption of static catalog attributes remains. A missing catalog is
optional and yields null metadata. In particular, the existing minimal Docker
backend bundle still starts without a catalog, with null display fields; its
packaging was not expanded in this task.

Household 1's verified display values, in unchanged recommendation order:

| Rank | Product ID | Description | Category |
|---|---:|---|---|
| 1 | 1082185 | BANANAS | TROPICAL FRUIT |
| 2 | 1049998 | GUM (PACKAGED) | CANDY - PACKAGED |
| 3 | 995242 | FLUID MILK WHITE ONLY | FLUID MILK PRODUCTS |
| 4 | 1005186 | SALAD BAR FRESH FRUIT | SALAD BAR |
| 5 | 1074612 | BEANS GREEN: FS/WHL/CUT | VEGETABLES - SHELF STABLE |

The page explicitly identifies these as dataset descriptions, not commercial
product names. The leading raw model score remains **1.3464024710989912**.

## Executed checks

```powershell
.\.venv-phase2\python.exe -B -m pytest -q -p no:cacheprovider tests/test_frontend.py tests/test_api.py
.\.venv-phase2\python.exe -B -m pytest -q -p no:cacheprovider -k 'not fit_save_reload'
.\.venv-phase2\python.exe -B -m pip check
```

- Focused suite: **28 passed in 27.83 seconds**.
- Regression suite: **85 passed, 1 deselected in 10.45 seconds**. The existing
  synthetic-fitting test was excluded to keep this checkpoint inference-only.
- **All 500,000 cutoff-701 scores matched the saved Phase 4 predictions exactly**.
  First/middle/last household API Top 5 IDs and raw scores also matched.
- New checks cover exact catalog mapping, null/unknown metadata without ranking
  changes, metadata checksum rejection, client order/score preservation, invalid
  responses, HTTP 404/500, timeout and connection failures, form submission, five
  cards, expandable technical details, HTML escaping, and clearing stale results.
- All **21 frozen input hashes** still match, including the original product table,
  model, feature schema, data/configuration and training/candidate/feature code.
- Installed **Streamlit 1.64.0** and required new dependencies in the existing
  Python 3.12 environment, using the original Phase 2 lock as constraints. Network
  access required sandbox approval. Every original locked package version was
  independently checked and remained unchanged. `pip check`: **No broken
  requirements found**. The UI requirements pin Streamlit/Pydantic and constrain
  shared dependencies; they are not a complete lock of all new transitive packages.
- `git diff --check` passed. The existing Starlette/AnyIO test-client deprecation
  warning remains; it did not affect any result.

Streamlit UI assertions use its official
[AppTest framework](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
Tests skip when the optional frontend dependency is absent. Test fixture product
strings are synthetic and are never used in the running demo.

## Local smoke test

A real Uvicorn process ran on `127.0.0.1:8765`, and a real headless Streamlit
process ran on `127.0.0.1:8502`. Results:

- FastAPI `/health`: HTTP 200, cutoff **701**, model **lambdarank-v1**, **2,500
  households** and **500,000 candidate rows**.
- Streamlit `/_stcore/health`: `ok`; frontend root: HTTP **200**.
- AppTest submitted the actual frontend form to the real FastAPI server over HTTP:
  **five cards** rendered with the real descriptions/categories above, with no UI
  exceptions. Returned product IDs and ordering matched the original API.
- Submitting household **999999** showed a friendly unavailable-household message
  and removed old cards. Stopping FastAPI and resubmitting showed a friendly
  connection-error message, with no unhandled exception.
- Both temporary server processes were stopped after verification.

This verifies rendered widget content/flow and real HTTP integration. No browser
screenshot or manual visual browser review was performed; none is claimed.

## Run the interview demo

From the repository root, install the optional frontend once:

```powershell
.\.venv-phase2\python.exe -m pip install -r requirements-ui.txt
```

PowerShell terminal 1:

```powershell
.\.venv-phase2\python.exe -m uvicorn retail_recs.api:create_app --factory --app-dir src --host 127.0.0.1 --port 8000
```

PowerShell terminal 2:

```powershell
.\.venv-phase2\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`, select household 1, and click **Get Recommendations**.
Stop each process with Ctrl+C. The frontend defaults to the local backend URL;
set `$env:RETAIL_API_URL` before starting it to use another backend address.

**80/20:** the existing API remains responsible for recommendations; the UI only
requests and displays them. Actual catalog descriptions make product IDs legible,
while technical details stay available for interview discussion. This remains a
historical offline demo at day 701, not live shopping data. The test period is
already consumed; these are integration checks, not new model-selection evidence.

Frozen model SHA-256:
`af9492f58af0b891b6f766213a5eb823bbb7fbf46aeb1719aba6c406515219ec`.

Feature-schema SHA-256:
`2d319c1da73dd254e0e609c3b76f1982bef4b0a7b153dec17ac2630305adde8f`.

This report is committed with the frontend. The resulting local checkpoint is
available with `git log -1`; the working tree is checked after committing.
