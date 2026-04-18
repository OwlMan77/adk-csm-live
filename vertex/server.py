"""Vertex AI custom prediction server for k2-fsa/OmniVoice.

Endpoint contract (matches existing csm_tts.py client):
  POST /predict
    request:  {"instances": [{"text": "...", ...}, ...]}
    response: {"predictions": [{"audio": "<base64-wav>"}, ...]}

  GET /health  →  200 {"status": "healthy"}

The reference voice is loaded once at startup from /app/samples (audio1.wav +
audio1.txt).  All synthesis requests use that fixed voice — no per-request
reference audio needed.
"""

import base64
import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", "/app/samples"))

_model = None
_ref_audio_path: str = ""
_ref_text: str = ""


def _pick_reference() -> tuple[str, str]:
    """Return (wav_path, transcript) for the first complete pair in SAMPLES_DIR."""
    if not SAMPLES_DIR.exists():
        logger.warning("SAMPLES_DIR %s not found — voice cloning unavailable", SAMPLES_DIR)
        return "", ""

    for txt_path in sorted(SAMPLES_DIR.glob("*.txt")):
        wav_path = txt_path.with_suffix(".wav")
        if wav_path.exists():
            transcript = txt_path.read_text(encoding="utf-8").strip()
            logger.info("Reference voice: %s | %d chars", wav_path.name, len(transcript))
            return str(wav_path), transcript

    logger.warning("No .wav/.txt pairs found in %s", SAMPLES_DIR)
    return "", ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _ref_audio_path, _ref_text

    logger.info("Loading OmniVoice on %s …", DEVICE)
    from omnivoice import OmniVoice

    _model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=DEVICE,
        dtype=torch.float16,
    )
    _ref_audio_path, _ref_text = _pick_reference()
    logger.info("OmniVoice ready. Reference: %s", _ref_audio_path or "none")
    yield


app = FastAPI(lifespan=lifespan)


class PredictRequest(BaseModel):
    instances: list[dict[str, Any]]
    parameters: dict[str, Any] = {}


def _ndarray_to_b64_wav(audio: np.ndarray, sample_rate: int = 24000) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def _synthesize_one(
    text: str,
    override_audio_b64: str = "",
    override_ref_text: str = "",
    num_step: int = 32,
    speed: float = 1.0,
    duration: float | None = None,
) -> str:
    """Run OmniVoice inference for a single text, return base64 WAV."""
    kwargs: dict[str, Any] = {
        "text": text,
        "num_step": num_step,
        "speed": speed,
    }
    if duration is not None:
        kwargs["duration"] = duration

    if override_audio_b64 and override_ref_text:
        audio_bytes = base64.b64decode(override_audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(audio_bytes)
        try:
            kwargs["ref_audio"] = tmp_path
            kwargs["ref_text"] = override_ref_text
            arrays = _model.generate(**kwargs)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    elif _ref_audio_path:
        kwargs["ref_audio"] = _ref_audio_path
        kwargs["ref_text"] = _ref_text
        arrays = _model.generate(**kwargs)
    else:
        arrays = _model.generate(**kwargs)

    return _ndarray_to_b64_wav(arrays[0])


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(request: PredictRequest):
    import asyncio

    loop = asyncio.get_running_loop()
    predictions = []

    for instance in request.instances:
        text = instance.get("text", "").strip()
        if not text:
            predictions.append({"audio": ""})
            continue

        # Optional per-request override — falls back to the baked-in reference.
        override_audio = instance.get("ref_audio", "")
        override_text = instance.get("ref_text", "")

        # Generation tuning from request.parameters (applies to all instances).
        num_step: int = request.parameters.get("num_step", 32)
        speed: float = request.parameters.get("speed", 1.0)
        duration: float | None = request.parameters.get("duration")

        audio_b64 = await loop.run_in_executor(
            None,
            _synthesize_one,
            text,
            override_audio,
            override_text,
            num_step,
            speed,
            duration,
        )
        predictions.append({"audio": audio_b64})

    return {"predictions": predictions}
