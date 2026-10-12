FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DEFAULT_TIMEOUT=120

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir \
    "fastapi>=0.136.0" \
    "uvicorn[standard]>=0.44.0" \
    "google-adk>=1.31.0" \
    "google-cloud-aiplatform>=1.148.0" \
    "python-dotenv>=1.2.2"

COPY app ./app
COPY scripts/samples ./scripts/samples

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
