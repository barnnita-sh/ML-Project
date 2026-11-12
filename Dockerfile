# Build:  docker build -t taxi-api .        Run:  docker run -p 8000:8000 taxi-api
# Needs the artifacts/ folder (python scripts/make_artifacts.py). If artifacts/manifest.json says the models were
# trained on another Python version, change the tag below to match.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src ARTIFACTS_DIR=/app/artifacts
WORKDIR /app
COPY requirements-api.txt artifacts/versions.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt -c versions.txt
COPY src/ src/
COPY app/ app/
COPY artifacts/ artifacts/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
