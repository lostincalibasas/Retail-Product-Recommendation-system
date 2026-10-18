# Personalized Retail Recommendation Engine

An independent ML portfolio project using Dunnhumby Complete Journey to recommend
products to households over the following seven days. Phase 1 provides verified
ingestion; the first Phase 2 checkpoint implements historical candidates, features,
labels, chronological development datasets, and coverage diagnostics. Phase 3 adds
a popularity baseline and LightGBM LambdaRank training/validation. Purchases are observed
behavior, not sales-representative acceptance or causal uplift labels.

## Phase 4 final held-out evaluation

The approved, unchanged LambdaRank Strategy A has completed its final evaluation
on cutoffs 624-701 / target days 625-708. Test NDCG@5 is **0.180820** versus
**0.081921** for popularity; micro Recall@5 is **6.9470%** versus **2.9632%**.
Candidate recall is **31.7649%**. Discovery remains weak: the ranker finds 12
new-product purchases versus 318 for popularity.

See the [final comparison](reports/phase4_test.md),
[verification, validation changes and handover](reports/phase4_checkpoint.md), and
[independent numerical checks](reports/phase4_verification.json).
All 58 automated tests passed. Final data and scores are stored separately under
`artifacts/data/retail-ranking-test-v1/` and `artifacts/evaluations/heldout-v1/`.

`scripts/evaluate_heldout.py` uses the original candidate/feature functions and
frozen model, with no fitting or selection. Its outputs already exist and cannot
be overwritten. To independently verify the saved evaluation:

```powershell
.\.venv-phase2\python.exe scripts/verify_heldout.py
```

The test period is now consumed for final evaluation. It must not become tuning
data while retaining a claim of untouched test performance.

## Historical FastAPI demo

The API serves the original frozen Strategy A Top 5 from the **day 701 historical
snapshot**, predicting the next seven days. It loads the 152-tree model once at
startup and scores each requested household's 200 saved candidates. No target
labels or saved predictions enter serving. This is a historical portfolio demo,
not a live retail catalog; scores are ranking scores, not purchase probabilities.
The original policy favors repeats but does not require all five to be repeats.

The existing Python 3.12 environment already contains the API dependencies,
recorded in `requirements-phase2.lock.txt`. The `api` project extra also declares
the minimal serving dependencies. From the repository root:

```powershell
.\.venv-phase2\python.exe -m uvicorn retail_recs.api:create_app --factory --app-dir src --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`, or request:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/recommendations/1
```

`/health` reports readiness, model version, snapshot cutoff, household count and
candidate-row count after successful startup. `/recommendations/{household_id}`
returns the household ID, `snapshot_cutoff`, `model_version`, and five
`recommendations`, each with `product_id` and `score`. Equal scores use ascending
product IDs. Unknown households return 404; invalid or nonpositive IDs return 422.

Startup verifies model/schema/snapshot checksums, explicit feature order, finite
float32 features, candidate uniqueness/counts, and the snapshot's recorded
point-in-time verification. Labels stored alongside features are excluded by
Parquet column projection; target files are never opened. Missing or inconsistent
artifacts prevent startup. Required local artifacts are ignored by Git:

- `artifacts/models/lambdarank-v1/`: model, manifest and feature schema.
- `artifacts/data/retail-ranking-test-v1/`: manifest and cutoff-701 rows snapshot.

Run focused tests with:

```powershell
.\.venv-phase2\python.exe -B -m pytest -q -p no:cacheprovider tests/test_api.py
```

These integration tests require the local artifacts and additionally use frozen
Phase 4 predictions to verify every cutoff-701 model score and representative
API Top 5 responses. They skip when the model/data artifacts are absent. See
[the repeat-purchase diagnostic](reports/phase4_repeat_diagnostic.md) for the
post-hoc household-frequency comparison and [API checkpoint](reports/phase5_api_checkpoint.md)
for execution evidence. No Docker packaging is included.

## Phase 3 model training and validation

The frozen-candidate baseline/LambdaRank comparison is implemented in
`scripts/train_ranker.py`. It verifies the Phase 2 artifacts, fits informative
training groups and evaluates every validation group. Candidate generation stays
fixed; test days 625-708 were sealed during model selection and were evaluated
only after Strategy A approval in Phase 4. See the exact
[training and evaluation contract](docs/phase3_contract.md).

```powershell
.\.venv-phase2\python.exe -m pytest -q
.\.venv-phase2\python.exe scripts/train_ranker.py
```

Model artifacts are written under `artifacts/models/lambdarank-v1/`, and the run
writes measured results to `reports/phase3_validation.md` and `.json`. Existing
run directories cannot be overwritten; pass `--output artifacts/models/<new-run>`
for a deliberate new experiment. This command trains; inspecting results does
not require rerunning it.

The completed comparison achieved validation end-to-end NDCG@5 of **0.183540**
versus **0.082016** for popularity, using all validation groups. Micro Recall@5
was **7.2502%** versus **3.0206%**. This improvement is concentrated in repeat
products: new-to-household hits fell from 332 to 11. See the
[measured results](reports/phase3_validation.md) and
[verification, limitations and handover](reports/phase3_checkpoint.md).
Validation selected the boosting iteration; the final test result is recorded
separately in Phase 4 above.

Phase 3.1 [model diagnostics](reports/phase3_1_diagnostics.md) explain the discovery
regression using the saved model and validation data. New products occupy 3.10%
of LambdaRank's Top 5 slots versus 43.88% for popularity; 302 of 321 lost new hits
still rank in its top five among new candidates alone. The diagnostic command
below verifies and scores the existing model without training or changing it:

```powershell
.\.venv-phase2\python.exe scripts/diagnose_ranker.py
```

Outputs go to `artifacts/diagnostics/lambdarank-v1/`; existing output directories
cannot be overwritten. See the report for feature importance, score distributions,
representative purchases and tests. The proposed discovery experiment is evaluated
in the Phase 3.2 checkpoint below.

Phase 3.2 [controlled discovery results](reports/phase3_2_discovery.md) compare the
original Top 5 with keeping ranks 1-4 and reserving rank 5 for the best unselected
new-to-household candidate. New hits increase from 11 to 212, while total hits
fall from 23,551 to 20,542 and NDCG@5 falls from 0.183540 to 0.170303.
The [verification and decision record](reports/phase3_2_checkpoint.md) recommends
retaining the original ranking. The alternative has not been promoted.

```powershell
.\.venv-phase2\python.exe scripts/experiment_discovery.py
```

This command uses saved validation scores and never trains. Results are already
saved under `artifacts/experiments/discovery-v1/`; existing directories cannot be
overwritten. The full regression suite passed 55 tests at this checkpoint.

## Phase 2 data checkpoint

Use the isolated Python 3.12 environment (`.venv-phase2/python.exe` on this Windows
Conda setup). The original Python 3.14 Phase 1 `.venv` remains separate.

To reproduce the runtime on Windows x86-64, with Conda installed:

```powershell
conda create --prefix .venv-phase2 --file environment/conda-win-64.lock.txt --yes
.\.venv-phase2\python.exe -m pip install -r requirements-phase2.lock.txt
.\.venv-phase2\python.exe scripts/smoke_environment.py
.\.venv-phase2\python.exe -m pip check
.\.venv-phase2\python.exe -m pytest -q
.\.venv-phase2\python.exe scripts/build_training_data.py --config configs/data_v1.json
```

The explicit Conda lock captures the runtime; the pip lock captures exact resolved
Python packages. The Conda lock is Windows-specific. A Linux lock and Docker image
will be validated at the later serving checkpoint. `requirements-phase2.txt`
records selected direct dependencies. Imports/native operations are smoke-tested
without fitting a model or starting an API service.

The builder writes ignored, versioned artifacts under `artifacts/data/retail-ranking-v1/`:
rows, independent target sets, group offsets/sizes, per-cutoff diagnostics, schema,
configuration and a checksum manifest. It refuses to overwrite an existing build;
use `--output artifacts/data/<new-run-name>` for another run. `--limit-cutoffs 1`
is a marked smoke build, not the complete dataset.

Training cutoffs are 365–533 (25 weekly snapshots); validation cutoffs are 540–617
(12). History includes the cutoff day; labels cover the next seven days. The loader
excludes transactions after day 624. Test targets 625–708 are not materialized until
model selection is finished and test evaluation is approved. The Phase 2 report
contains data diagnostics; measured model results are in the Phase 3 report above.

See [the shared contract](docs/phase2_contract.md) for exact candidate selection,
25 ordered features, population/label definitions, agent ownership, and future
ranking interfaces. [Phase 1 exploration](reports/data_exploration.md) remains the
source-data baseline.

The [data checkpoint report](reports/phase2_data_checkpoint.md) records measured
coverage and group diagnostics. Regenerate it, including artifact checksum and
group-offset verification, with:

```powershell
.\.venv-phase2\python.exe scripts/report_data_checkpoint.py
```

## Dataset provenance and license

- Dataset: [Dunnhumby - The Complete Journey on Kaggle](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey)
- Source dataset license as displayed by Kaggle: [Database Contents License 1.0 (DbCL-1.0)](http://opendatacommons.org/licenses/dbcl/1.0/). Read the license before redistributing or using the database; the GitHub mirror is a download convenience, not a replacement for the source license.
- Public no-Kaggle-account download mirror: [GitHub release v1.0-data](https://github.com/bltap-plmarket/dunnhumby-complete-journey/releases/tag/v1.0-data)
- Release asset: `dunnhumby-complete-journey-data.zip` (135,859,064 bytes). Published SHA-256: `324288a8811c4ed86e85415fdee36f3692582e2cc9dd87866c0e333df237f782`.
- This workspace's archive was downloaded from that release and its SHA-256 was verified before extraction. The extracted source contains the eight CSVs listed below.

The GitHub project documents a two-year, household-level retail dataset. Its Kaggle page describes the dataset's scope and eight files. We treat those as provenance, not as measured results: row counts and quality statistics in [the exploration report](reports/data_exploration.md) are computed locally from the downloaded files.

## Quick start (Windows PowerShell)

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Download the release archive into `data/raw/`, verify its checksum, and extract it there. The archive used for this Phase 1 run is already in that folder; to repeat the download:

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
$archive = 'data\raw\dunnhumby-complete-journey-data.zip'
Invoke-WebRequest -UseBasicParsing `
  -Uri 'https://github.com/bltap-plmarket/dunnhumby-complete-journey/releases/download/v1.0-data/dunnhumby-complete-journey-data.zip' `
  -OutFile $archive
Get-FileHash -Algorithm SHA256 $archive
Expand-Archive -LiteralPath $archive -DestinationPath data\raw -Force
```

Compare the hash with the published SHA-256 above before extracting. Then run:

```powershell
.\.venv\Scripts\python scripts\ingest_data.py
```

The script expects the eight CSVs in `data/raw/`. It writes one Parquet file per table to `data/processed/` and replaces `reports/data_exploration.md` with counts, missingness, and key-relationship checks from the local files. Raw and processed data are ignored by Git because they are large; the exploration report is a small generated artifact and can be reviewed/versioned.

## Project layout

```text
data/
  raw/          Downloaded ZIP and source CSVs (not tracked)
  processed/    Generated Parquet files (not tracked)
reports/        Data exploration output
scripts/        Reproducible ingestion and validation
```

## Script: 80/20 explanation

`scripts/ingest_data.py` does three useful things: it checks that all eight expected input tables exist, converts each full CSV to Parquet without dropping or imputing rows, then computes source row/missing-value summaries plus transaction coverage and selected foreign-key checks. It writes those observations to the Markdown report. It deliberately keeps `DAY` as the dataset's relative integer day index because no calendar-date origin is supplied in these CSVs, and uses distinct transaction `PRODUCT_ID` values as the available SKU-count proxy.

The checks are practical diagnostics, not a complete formal schema contract: this extract has no store dimension, and an unmatched key is surfaced for investigation rather than silently filtered. The transaction records describe household purchases and `SALES_VALUE`; they do not identify whether a sales representative accepted a recommendation.

## Current scope boundary

Phase 4 final held-out evaluation and the repeat-purchase diagnostic are complete.
The historical FastAPI demo serves unchanged Strategy A; the discovery rule was
not promoted. SHAP and Docker remain future checkpoints. Test results must not be
used for model selection or tuning.
MLflow and Databricks remain optional extensions.
