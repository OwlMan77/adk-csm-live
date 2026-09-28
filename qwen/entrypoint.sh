#!/usr/bin/env bash
# Start vLLM-Omni serving Qwen3-TTS, with the aligner and the baked-in voices.
set -euo pipefail

MODEL="${TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
ALIGNER="${TTS_ALIGNER:-Qwen/Qwen3-ForcedAligner-0.6B}"
CUSTOM_VOICE_DIR="${CUSTOM_VOICE_DIR:-/app/custom_voices}"
PORT="${PORT:-8080}"

# The tuned deploy config ships inside the package. Copy it rather than edit it
# in place, and point it at the voice profiles baked in at build time.
PACKAGED_CONFIG="$(python -c "import os, vllm_omni; print(os.path.join(os.path.dirname(vllm_omni.__file__), 'deploy', 'qwen3_tts.yaml'))")"
DEPLOY_CONFIG=/app/qwen3_tts.yaml
cp "${PACKAGED_CONFIG}" "${DEPLOY_CONFIG}"

if [[ -d "${CUSTOM_VOICE_DIR}" ]]; then
  printf '\ncustom_voice_dir: %s\n' "${CUSTOM_VOICE_DIR}" >> "${DEPLOY_CONFIG}"
  echo "Voice profiles: $(ls -1 "${CUSTOM_VOICE_DIR}" | tr '\n' ' ')"
else
  echo "WARNING: ${CUSTOM_VOICE_DIR} not found — requests must carry ref_audio." >&2
fi

echo "Serving ${MODEL} on :${PORT} (aligner: ${ALIGNER})"

# --forced-aligner is what makes word_timestamps available on the streaming
# WebSocket. Without it the server answers 400 to word_timestamps=true and the
# face falls back to pure audio analysis, which still works — it just loses
# anticipation, the one thing analysis structurally cannot do.
exec vllm serve "${MODEL}" \
  --omni \
  --deploy-config "${DEPLOY_CONFIG}" \
  --forced-aligner "${ALIGNER}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  "$@"
