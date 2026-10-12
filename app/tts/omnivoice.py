"""OmniVoice on a Vertex AI endpoint — the engine that is deployed today.

Kept behind the same interface as Qwen3-TTS so the two can be compared by
measurement rather than by argument (scripts/bench_tts.py), and so falling back
is a one-line env change rather than a revert.

Three things it structurally cannot do, which is why there is a second engine:
no emphasis markup beyond a closed tag list, no word or phoneme timings, and no
intra-utterance streaming — its attention is fully bidirectional, so every
decoding step is a prefill of the whole utterance and the first byte arrives
only once the last one has. Its weights are also CC-BY-NC.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import wave
from collections.abc import AsyncIterator
from functools import lru_cache

import google.cloud.aiplatform as aiplatform

from app.tts.base import AudioChunk, Utterance

logger = logging.getLogger(__name__)

# 32 is the library default. 16 is the documented fast setting and roughly
# halves the wait. The client used to send this as "decoding_steps" inside the
# instance while the server read "num_step" from the parameters block, so it was
# silently ignored on every request; the server now accepts both spellings.
NUM_STEP = int(os.environ.get("OMNIVOICE_NUM_STEP", "16"))

TIMEOUT_S = float(os.environ.get("OMNIVOICE_TIMEOUT_S", "25"))


@lru_cache(maxsize=1)
def _endpoint() -> aiplatform.Endpoint:
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("OMNIVOICE_LOCATION", "us-central1")
    endpoint_id = os.environ["OMNIVOICE_ENDPOINT_ID"]

    aiplatform.init(project=project, location=location)
    return aiplatform.Endpoint(
        endpoint_name=f"projects/{project}/locations/{location}/endpoints/{endpoint_id}"
    )


def _wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    """Unwrap a RIFF container. The rate is whatever the server actually used."""
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        return wf.readframes(wf.getnframes()), wf.getframerate()


class OmniVoiceEngine:
    name = "omnivoice"

    async def synthesize(self, utterance: Utterance) -> AsyncIterator[AudioChunk]:
        """Yields exactly one chunk: this engine has no intermediate state to give."""
        instances = [{"text": utterance.text, "num_step": NUM_STEP}]

        loop = asyncio.get_running_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: _endpoint().predict(instances=instances)),
            timeout=TIMEOUT_S,
        )

        audio_b64: str = response.predictions[-1]["audio"]
        if not audio_b64:
            return

        pcm, sample_rate = _wav_to_pcm(base64.b64decode(audio_b64))
        yield AudioChunk(
            pcm=pcm,
            sample_rate=sample_rate,
            utterance=utterance,
            first=True,
            final=True,
        )

    async def warm(self) -> bool:
        """Vertex runs at min-replica-count>=1, so there is nothing to wake."""
        return True

    async def aclose(self) -> None:
        return None
