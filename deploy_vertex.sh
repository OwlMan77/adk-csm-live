#!/usr/bin/env bash
# Deploy OmniVoice to Vertex AI: build -> Artifact Registry -> Model Registry -> Endpoint
# Run from the adk-csm-live project root.
#
# Reads HF_TOKEN from .env automatically if present.
#
# Usage:
#   ./deploy_vertex.sh
#   PROJECT_ID=my-project REGION=us-east1 ./deploy_vertex.sh
set -euo pipefail

# ── Load .env ──────────────────────────────────────────────────────────────────
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# ── Config ─────────────────────────────────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:-${GOOGLE_CLOUD_PROJECT:?GOOGLE_CLOUD_PROJECT is not set}}"
REGION="${REGION:-${OMNIVOICE_LOCATION:-${GOOGLE_CLOUD_LOCATION:?GOOGLE_CLOUD_LOCATION is not set}}}"
REPOSITORY="${REPOSITORY:-omnivoice}"
IMAGE_NAME="omnivoice-vertex"
IMAGE_TAG="${IMAGE_TAG:-latest}"
MODEL_DISPLAY_NAME="OmniVoice"
ENDPOINT_DISPLAY_NAME="omnivoice-endpoint"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-4}"
ACCELERATOR_TYPE="nvidia-l4"
ACCELERATOR_COUNT=1

AR_HOST="${REGION}-docker.pkg.dev"
IMAGE="${AR_HOST}/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

# ── Preflight ──────────────────────────────────────────────────────────────────
: "${HF_TOKEN:?HF_TOKEN is required - set it in .env or environment}"

command -v docker >/dev/null || { echo "ERROR: docker not found"; exit 1; }
command -v gcloud >/dev/null || { echo "ERROR: gcloud not found"; exit 1; }

echo "Project  : ${PROJECT_ID}"
echo "Region   : ${REGION}"
echo "Image    : ${IMAGE}"
echo "Machine  : ${MACHINE_TYPE} + ${ACCELERATOR_COUNT}x ${ACCELERATOR_TYPE}"
echo ""

# ── 1. Build ───────────────────────────────────────────────────────────────────
echo "▶ [1/6] Checking if Docker image exists locally..."
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "   Image found locally - skipping build."
else
  echo "   Image not found - building (model weights baked in, ~10 min)..."
  docker build \
    --build-arg "HF_TOKEN=${HF_TOKEN}" \
    --tag "${IMAGE}" \
    --file vertex/Dockerfile \
    .
fi

# ── 2. Artifact Registry repo (idempotent) ─────────────────────────────────────
echo "▶ [2/6] Ensuring Artifact Registry repository '${REPOSITORY}' exists..."
gcloud artifacts repositories create "${REPOSITORY}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --description="OmniVoice serving containers" \
    --quiet 2>/dev/null || true

# ── 3. Push ────────────────────────────────────────────────────────────────────
echo "▶ [3/6] Pushing image to Artifact Registry..."
gcloud auth configure-docker "${AR_HOST}" --quiet
docker push "${IMAGE}"

# ── 4. Register model (skip if already exists) ─────────────────────────────────
echo "▶ [4/6] Checking if model '${MODEL_DISPLAY_NAME}' already exists..."
MODEL_RESOURCE=$(gcloud ai models list \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --filter="displayName=${MODEL_DISPLAY_NAME}" \
  --format="value(name)" \
  --limit=1 2>/dev/null | head -1)

if [[ -n "${MODEL_RESOURCE}" ]]; then
  echo "   Model found - skipping upload."
else
  echo "   Model not found - uploading..."
  MODEL_RESOURCE=$(gcloud ai models upload \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --display-name="${MODEL_DISPLAY_NAME}" \
    --container-image-uri="${IMAGE}" \
    --container-health-route="/health" \
    --container-predict-route="/predict" \
    --container-ports=8080 \
    --format="value(model)")
fi
echo "   Model: ${MODEL_RESOURCE}"

# ── 5. Create endpoint ─────────────────────────────────────────────────────────
echo "▶ [5/6] Creating Vertex AI endpoint..."
ENDPOINT_RESOURCE=$(gcloud ai endpoints create \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --display-name="${ENDPOINT_DISPLAY_NAME}" \
  --format="value(name)")

ENDPOINT_ID=$(basename "${ENDPOINT_RESOURCE}")
echo "   Endpoint: ${ENDPOINT_RESOURCE}"

# ── 6. Deploy model to endpoint ────────────────────────────────────────────────
echo "▶ [6/6] Deploying model (GPU provisioning takes ~10-15 min)..."
gcloud ai endpoints deploy-model "${ENDPOINT_ID}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --model="${MODEL_RESOURCE}" \
  --display-name="${MODEL_DISPLAY_NAME}" \
  --machine-type="${MACHINE_TYPE}" \
  --accelerator="type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT}" \
  --min-replica-count=1 \
  --max-replica-count=2 \
  --traffic-split=0=100

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "Deployment complete!"
echo ""
echo "Add this to your .env:"
echo "  OMNIVOICE_ENDPOINT_ID=${ENDPOINT_ID}"
echo ""
echo "Then test with:"
echo "  uv run python scripts/test_csm.py --text 'Hello, this is a test.'"
