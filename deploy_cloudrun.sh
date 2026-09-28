#!/usr/bin/env bash
# Deploy adk-csm-live backend to Cloud Run.
# Run from the project root.
#
# Usage:
#   ./deploy_cloudrun.sh
#   PROJECT_ID=my-project REGION=us-central1 ./deploy_cloudrun.sh
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PROJECT_ID="${PROJECT_ID:-${GOOGLE_CLOUD_PROJECT:?GOOGLE_CLOUD_PROJECT is not set}}"
REGION="${REGION:-${GOOGLE_CLOUD_LOCATION:-us-central1}}"
SERVICE_NAME="${SERVICE_NAME:-adk-csm-live}"
REPOSITORY="${REPOSITORY:-omnivoice}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

AR_HOST="${REGION}-docker.pkg.dev"
IMAGE="${AR_HOST}/${PROJECT_ID}/${REPOSITORY}/${SERVICE_NAME}:${IMAGE_TAG}"

command -v docker >/dev/null || { echo "ERROR: docker not found"; exit 1; }
command -v gcloud >/dev/null || { echo "ERROR: gcloud not found"; exit 1; }

if [[ "${TTS_ENGINE:-omnivoice}" == "omnivoice" && -z "${OMNIVOICE_ENDPOINT_ID:-}" ]]; then
  echo "ERROR: TTS_ENGINE=omnivoice needs OMNIVOICE_ENDPOINT_ID"; exit 1
fi
if [[ "${GOOGLE_GENAI_USE_VERTEXAI:-true}" != "true" && -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: GOOGLE_GENAI_USE_VERTEXAI is false, so GEMINI_API_KEY is required"; exit 1
fi
# GEMINI_API_KEY and the Modal token pair go in as plain env vars, which means
# they are readable from `gcloud run services describe` and from the revision
# metadata by anyone with viewer access on the project. Fine for a personal
# project; move them to Secret Manager (--set-secrets) before anyone else has
# access to it.
if [[ "${TTS_ENGINE:-}" == "qwen3-tts" && -z "${QWEN_TTS_URL:-}" ]]; then
  echo "ERROR: TTS_ENGINE=qwen3-tts needs QWEN_TTS_URL (run ./deploy_qwen_cloudrun.sh first)"; exit 1
fi

echo "Project  : ${PROJECT_ID}"
echo "Region   : ${REGION}"
echo "Image    : ${IMAGE}"
echo "Service  : ${SERVICE_NAME}"
echo "Voice    : ${TTS_ENGINE:-omnivoice}"
echo ""

# ── 1. Build ───────────────────────────────────────────────────────────────────
echo "▶ [1/4] Building image..."
docker build --tag "${IMAGE}" --file Dockerfile .

# ── 2. Push ────────────────────────────────────────────────────────────────────
echo "▶ [2/4] Pushing to Artifact Registry..."
gcloud auth configure-docker "${AR_HOST}" --quiet
docker push "${IMAGE}"

# ── 3. Deploy to Cloud Run ─────────────────────────────────────────────────────
echo "▶ [3/4] Deploying to Cloud Run..."
# min-instances=0: this container is idle almost always, its cold start is a
# couple of seconds, and that start lands on /warm — which the frontend fires
# while the live dialog is still on screen — rather than on someone waiting to
# be answered. Set it back to 1 to buy those seconds back for a few dollars.
gcloud run deploy "${SERVICE_NAME}" \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --platform=managed \
  --allow-unauthenticated \
  --port=8080 \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=0 \
  --max-instances=10 \
  --timeout=3600 \
  --set-env-vars="^@^GOOGLE_CLOUD_PROJECT=${PROJECT_ID}@GOOGLE_CLOUD_LOCATION=${REGION}@GOOGLE_GENAI_USE_VERTEXAI=${GOOGLE_GENAI_USE_VERTEXAI:-true}@GEMINI_MODEL=${GEMINI_MODEL:-gemini-2.5-flash}@GEMINI_API_KEY=${GEMINI_API_KEY:-}@QWEN_TTS_KEY=${QWEN_TTS_KEY:-}@QWEN_TTS_SECRET=${QWEN_TTS_SECRET:-}@TTS_ENGINE=${TTS_ENGINE:-omnivoice}@OMNIVOICE_LOCATION=${OMNIVOICE_LOCATION:-us-central1}@OMNIVOICE_ENDPOINT_ID=${OMNIVOICE_ENDPOINT_ID:-}@OMNIVOICE_NUM_STEP=${OMNIVOICE_NUM_STEP:-16}@QWEN_TTS_URL=${QWEN_TTS_URL:-}"

# ── 4. Print URL ───────────────────────────────────────────────────────────────
echo "▶ [4/4] Fetching service URL..."
URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(status.url)")

echo ""
echo "Deployment complete!"
echo ""
echo "Service URL: ${URL}"
echo ""
echo "Add to your frontend .env:"
echo "  VITE_ADK_BASE_URL=${URL}"
