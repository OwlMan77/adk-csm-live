#!/usr/bin/env bash
# Deploy Qwen3-TTS on vLLM-Omni to Cloud Run with one L4, scaling to zero.
# Run from the project root.
#
# Usage:
#   ./deploy_qwen_cloudrun.sh
#   PROJECT_ID=my-project REGION=us-central1 ./deploy_qwen_cloudrun.sh
#
# Why Cloud Run and not the Vertex endpoint this replaces:
#
#   Vertex, min-replica-count=1   ~$0.75-1.20/hr around the clock (~$550-880/mo)
#                                 for a service that is idle almost always.
#   Vertex, min-replica-count=0   free at idle, but 60 s - 5 min to wake AND the
#                                 first request is dropped with a 429.
#   Cloud Run + 1x L4             free at idle, per-second billing, ~5-20 s cold
#                                 start, and the request waits rather than fails.
#
# The cold start is made survivable rather than pretended away: weights are baked
# into the image (see qwen/Dockerfile), and the frontend fires GET /warm on the
# orchestrator the moment the live agent is selected — while the start dialog is
# still on screen. That is several seconds of human time that would otherwise be
# spent staring at a silent face.
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PROJECT_ID="${PROJECT_ID:-${GOOGLE_CLOUD_PROJECT:?GOOGLE_CLOUD_PROJECT is not set}}"
REGION="${REGION:-${GOOGLE_CLOUD_LOCATION:-us-central1}}"
SERVICE_NAME="${QWEN_SERVICE_NAME:-qwen3-tts}"
REPOSITORY="${REPOSITORY:-omnivoice}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

AR_HOST="${REGION}-docker.pkg.dev"
IMAGE="${AR_HOST}/${PROJECT_ID}/${REPOSITORY}/${SERVICE_NAME}:${IMAGE_TAG}"

command -v docker >/dev/null || { echo "ERROR: docker not found"; exit 1; }
command -v gcloud >/dev/null || { echo "ERROR: gcloud not found"; exit 1; }
# No HF_TOKEN needed: Qwen3-TTS, its tokenizer and the forced aligner are all
# public and Apache-2.0. Set one anyway if anonymous rate limits bite on a
# shared IP — the build uses it when present and ignores it when not.

echo "Project  : ${PROJECT_ID}"
echo "Region   : ${REGION}"
echo "Image    : ${IMAGE}"
echo "Service  : ${SERVICE_NAME}  (L4, min-instances=0)"
echo ""

echo "▶ [1/4] Building image..."
echo "        ~7.1 GB of weights plus the voice profiles are baked in: slow once,"
echo "        then every cold start is a container start rather than a download."
DOCKER_BUILDKIT=1 docker build \
  --tag "${IMAGE}" \
  --file qwen/Dockerfile \
  --build-arg "HF_TOKEN=${HF_TOKEN:-}" \
  .

echo "▶ [2/4] Pushing to Artifact Registry..."
gcloud auth configure-docker "${AR_HOST}" --quiet
docker push "${IMAGE}"

echo "▶ [3/4] Deploying to Cloud Run..."
# --no-cpu-throttling matters: without it Cloud Run freezes the CPU between
# requests, which strands the model load and makes every request look cold.
gcloud run deploy "${SERVICE_NAME}" \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --platform=managed \
  --allow-unauthenticated \
  --port=8080 \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --cpu=8 \
  --memory=32Gi \
  --min-instances=0 \
  --max-instances=2 \
  --concurrency=4 \
  --timeout=3600 \
  --no-cpu-throttling \
  --startup-probe="tcpSocket.port=8080,initialDelaySeconds=60,periodSeconds=10,failureThreshold=60"

echo "▶ [4/4] Fetching service URL..."
URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(status.url)")

echo ""
echo "Deployment complete."
echo ""
echo "Service URL: ${URL}"
echo ""
echo "Point the orchestrator at it and switch engines:"
echo "  QWEN_TTS_URL=${URL}"
echo "  TTS_ENGINE=qwen3-tts"
echo ""
echo "Then compare the two by ear before committing to the swap:"
echo "  uv run python scripts/bench_tts.py --both"
