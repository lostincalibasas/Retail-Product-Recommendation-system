# Smart Order ML: Complete Journey Phase 1

A small, reproducible data-ingestion baseline for a retail SKU recommendation interview prototype. This phase covers only source-data acquisition, CSV-to-Parquet conversion, and data validation/exploration. It does not train a model or claim customer purchases are sales-representative acceptance labels.

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

## Scope boundary

Phase 2 (features, modeling, evaluation, and deployment) is not included. We will not move into it without your confirmation.
