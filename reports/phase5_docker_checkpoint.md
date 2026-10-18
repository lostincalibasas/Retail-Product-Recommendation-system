# Minimal Docker packaging checkpoint

Date: 2026-10-19 (Asia/Dubai). Base checkpoint: `e75d9c8` on `main`.

**Status: packaging prepared and native checks passed; Docker build and container
execution remain unverified.** `docker version` failed because the command was
not found. `Get-Command docker` returned nothing, and the standard Docker Desktop
CLI location `C:\Program Files\Docker\Docker\resources\bin\docker.exe` was absent.
No image build, Linux dependency installation, container run or container parity
success is claimed. Docker was not installed as part of this task.

## Changes

- `Dockerfile`: Python 3.12 slim Debian Bookworm; Linux OpenMP runtime (`libgomp1`);
  serving-only dependencies; unchanged source; five existing artifact files;
  unprivileged runtime user; original Uvicorn application factory on port 8000.
- `.dockerignore`: default-deny allowlist excluding raw/processed datasets, other
  snapshots, saved predictions, environments, Git metadata, reports and secrets.
- `requirements-serving.txt`: eight pinned direct serving packages using the
  versions already installed in Phase 5. Training/explainability/test dependencies
  are excluded. Base-image tag and transitive dependencies are not fully locked;
  this is not a validated Linux lock or bit-for-bit reproducibility claim.
- `scripts/verify_api_demo.py`: checks a running server's `/docs`, `/health` and
  real household recommendations against the unchanged local API. IDs/order and
  metadata must match exactly; score absolute tolerance is 1e-12, relative tolerance
  zero. It requires the existing local Phase 5 environment/artifacts, including
  `httpx`, and does not run inside the image.
- `README.md`: PowerShell build, run, test, compare and stop instructions, with the
  Docker availability limitation made explicit.

## Files required inside the image

The API derives its root from `/app/src/retail_recs/api.py`. Copies preserve the
same relative paths under `/app`, so no API settings or behavior changes are needed:

```text
artifacts/models/lambdarank-v1/model.txt
artifacts/models/lambdarank-v1/manifest.json
artifacts/models/lambdarank-v1/feature_schema.json
artifacts/data/retail-ranking-test-v1/manifest.json
artifacts/data/retail-ranking-test-v1/test/cutoff_701/rows.parquet
```

These five artifacts total **8,614,321 bytes**. Expanding the explicit allowlist
on this workspace yielded **21 files / 8,745,634 bytes**, including Python source,
Docker configuration and serving requirements, before Docker metadata/overhead.
The allowlist and COPY inputs were inspected with Python; Docker's own build-
context processing has not run. Git still ignores all large artifact files.

The original row snapshot includes its existing relevance column. The unchanged
API continues to project only identifiers and the 25 historical features; no
target files or saved prediction files are included or loaded by serving. No
artifact was regenerated or stripped. Recommendations use cutoff 701 and the
original Strategy A Top 5, with ascending product-ID tie breaks. No repeat-only
filter, discovery rule, model change or evaluation change was introduced.

## Verification actually executed

```powershell
.\.venv-phase2\python.exe -B -m pytest -q -p no:cacheprovider -k 'not fit_save_reload'
```

**70 passed, 1 deselected in 18.36 seconds.** The existing synthetic fitting test
was excluded. The existing upstream Starlette/AnyIO `BlockingPortal` deprecation
warning remains; no dependency changes were made to suppress it.

- The existing API test reproduced **all 500,000 cutoff-701 saved Phase 4 scores
  exactly**, and representative API Top 5 responses matched saved predictions.
- All **21 frozen input hashes** in the Phase 4 data manifest still match.
  Frozen model, schema, training/data code, configuration and evaluated artifacts
  remain unchanged. Original Phase 4 performance reports were not edited.
- The allowlisted files were copied to a temporary directory, preserving relative
  paths and verifying copied bytes. A fresh Uvicorn process ran from that directory
  using the existing Windows Python environment, with no other datasets or
  prediction artifacts available under its application root.
- `scripts/verify_api_demo.py --url http://127.0.0.1:8765` passed against that staged
  server: `/health` and `/docs` returned HTTP 200, and household 1 returned HTTP 200
  with the exact same metadata, Top 5 IDs/order and scores as the local API.
  **Maximum score difference: 0.0**. The temporary server was stopped afterward.
- Health: `lambdarank-v1`, cutoff **701**, **2,500 households**, **500,000 rows**.
- Household 1: **1082185, 1049998, 995242, 1005186, 1074612**. Leading score:
  **1.3464024710989912**.
- `git diff --check` passed.

The staged run verifies completeness of the copied serving files and unchanged
native behavior. It does **not** verify Linux wheels, OpenMP installation, image
permissions, Docker build syntax execution or cross-platform prediction parity.
Those checks remain pending Docker availability.

## Commands to finish Docker verification

From the repository root, with Docker Desktop's Linux engine running:

```powershell
docker version
docker build --platform linux/amd64 -t retail-recs-demo:phase5 .
docker run --detach --rm --name retail-recs-demo -p 127.0.0.1:8000:8000 retail-recs-demo:phase5
docker logs retail-recs-demo
# Wait until startup completes before the following checks.
Invoke-RestMethod http://127.0.0.1:8000/health
(Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/docs).StatusCode
Invoke-RestMethod http://127.0.0.1:8000/recommendations/1 | ConvertTo-Json -Depth 5
.\.venv-phase2\python.exe -B scripts/verify_api_demo.py --url http://127.0.0.1:8000
.\.venv-phase2\python.exe -B -m pytest -q -p no:cacheprovider tests/test_api.py
docker stop retail-recs-demo
```

`--rm` removes the stopped container. If port 8000 is occupied, bind host port
8001 and use that port in verification URLs. The container requires no bind mounts
or external services, because the five immutable artifacts are bundled locally.

**80/20:** preserve the verified application and artifact paths, copy only the
files it reads, and compare its output with the local API before claiming container
parity. Docker availability is the only current execution blocker; no new ML or
deployment architecture is introduced. This checkpoint does not tune on the
already-opened test period or change its evaluation claims. No push occurred.

Frozen model SHA-256:
`af9492f58af0b891b6f766213a5eb823bbb7fbf46aeb1719aba6c406515219ec`.

Feature-schema SHA-256:
`2d319c1da73dd254e0e609c3b76f1982bef4b0a7b153dec17ac2630305adde8f`.

Cutoff-701 rows SHA-256:
`29cd69c658c13a4ada8199ad2fe2f53be2a220b141d8855935109cf625c89a17`.

This report is committed with the Docker files. The resulting local checkpoint
ID is available with `git log -1`; the working tree is verified after committing.
