FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# The Linux LightGBM wheel needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt ./
RUN pip install --no-cache-dir --only-binary=:all: -r requirements-serving.txt

COPY src/ ./src/
COPY artifacts/models/lambdarank-v1/model.txt artifacts/models/lambdarank-v1/manifest.json artifacts/models/lambdarank-v1/feature_schema.json ./artifacts/models/lambdarank-v1/
COPY artifacts/data/retail-ranking-test-v1/manifest.json ./artifacts/data/retail-ranking-test-v1/
COPY artifacts/data/retail-ranking-test-v1/test/cutoff_701/rows.parquet ./artifacts/data/retail-ranking-test-v1/test/cutoff_701/

USER 10001:10001
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "retail_recs.api:create_app", "--factory", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
