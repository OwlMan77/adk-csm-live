# Qwen3-TTS-12Hz-1.7B-Base on vLLM-Omni, for Cloud Run + 1x L4.
#
# Everything the first request needs is baked in: model weights, the tokenizer,
# the forced aligner, and the precomputed voice profiles. The service scales to
# zero, so a cold start is a container start, not a 3 GB download — that is the
# difference between a ~15 s wake and a two-minute one.
#
# Pinned to a release tag, not a nightly. Commit-pinned `nightly-<sha>` tags look
# like the safer choice but are pruned to the newest handful — about an eleven-day
# window — so a pin that reproduces today stops existing within a fortnight. The
# streaming and word-timestamp path this image depends on is present in the
# tagged releases, so nothing is given up by pinning to one.
ARG VLLM_OMNI_IMAGE=vllm/vllm-omni:v0.30.0rc1
FROM ${VLLM_OMNI_IMAGE}

# Must match the image above: the precompute script is fetched from the same ref,
# and it has to agree with the server about the voice-profile format.
ARG VLLM_OMNI_REF=v0.30.0rc1

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    TTS_TOKENIZER=Qwen/Qwen3-TTS-Tokenizer-12Hz \
    TTS_ALIGNER=Qwen/Qwen3-ForcedAligner-0.6B \
    CUSTOM_VOICE_DIR=/app/custom_voices

WORKDIR /app

RUN pip install --no-cache-dir "huggingface_hub[hf_transfer]"

# ── weights, at build time ────────────────────────────────────────────────────
# All three repos are public and Apache-2.0, so this needs no credentials. The
# build arg exists only for anonymous rate limits on a shared IP. Deliberately
# not promoted to ENV: that would bake the token into the published image, and
# this image goes to Artifact Registry.
ARG HF_TOKEN=""

RUN python -c "\
import os;\
from huggingface_hub import snapshot_download;\
token = os.environ.get('HF_TOKEN') or None;\
[snapshot_download(repo_id=r, token=token) for r in (\
    os.environ['TTS_MODEL'], os.environ['TTS_TOKENIZER'], os.environ['TTS_ALIGNER'])]"

# ── voice profiles, also at build time ────────────────────────────────────────
# Encoding a reference clip runs the speaker encoder and the codec. Doing it per
# request is pure waste: the reference never changes. One profile per mood, since
# the -Base checkpoint has no `instruct` parameter and the clip *is* the emotion.
ADD https://raw.githubusercontent.com/vllm-project/vllm-omni/${VLLM_OMNI_REF}/examples/online_serving/text_to_speech/qwen3_tts/precompute_custom_voice.py \
    /app/precompute_custom_voice.py

COPY scripts/samples /app/samples
COPY qwen/precompute_voices.py /app/precompute_voices.py
RUN python /app/precompute_voices.py \
      --samples-dir /app/samples \
      --output-dir "${CUSTOM_VOICE_DIR}" \
      --model "${TTS_MODEL}"

COPY qwen/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Cloud Run injects PORT; 8091 is the vLLM-Omni convention for local runs.
ENV PORT=8080

# No EXPOSE here on purpose. It is documentation-only in Docker -- Cloud Run
# routes on $PORT and Modal on the port named in its decorator, so neither reads
# it -- and Modal's Dockerfile builder rejects it outright. Dropping it is what
# lets one image serve both deploy targets (see qwen/modal_app.py).

ENTRYPOINT ["/app/entrypoint.sh"]
