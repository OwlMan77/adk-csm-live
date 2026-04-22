#!/usr/bin/env bash
# Deploy OmniVoice-Triton to Vertex AI.
# Wraps deploy_vertex.sh with TRITON=true.
#
# Usage:
#   ./deploy_triton.sh
#   PROJECT_ID=my-project REGION=us-east1 ./deploy_triton.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRITON=true exec "${SCRIPT_DIR}/deploy_vertex.sh" "$@"
