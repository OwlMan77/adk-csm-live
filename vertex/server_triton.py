"""Vertex AI prediction server for OmniVoice using omnivoice-triton.

Endpoint contract (matches the app/tts/omnivoice.py client):
  POST /predict
    request:  {"instances": [{"text": "...", "decoding_steps": 8}, ...]}
    response: {"predictions": [{"audio": "<base64-wav>"}, ...]}

  GET /health  →  200 {"status": "healthy"}

The reference voice is loaded once at startup from /app/samples.
Hybrid mode (Triton kernels + CUDA graph capture) gives ~3.4x speedup
over standard PyTorch inference.
"""

import asyncio
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
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", "/app/samples"))
DEFAULT_STEPS = int(os.environ.get("DECODING_STEPS", "8"))

_runner = None
_ref_audio_path: str = ""
_ref_text: str = ""
_ready = False


def _pick_reference() -> tuple[str, str]:
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


async def _load_model():
    global _runner, _ref_audio_path, _ref_text, _ready
    try:
        logger.info("Downloading k2-fsa/OmniVoice weights …")
        from huggingface_hub import snapshot_download
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: snapshot_download(
            repo_id="k2-fsa/OmniVoice",
            token=os.environ.get("HF_TOKEN") or None,
        ))
        logger.info("Weights ready. Loading OmniVoice-Triton (triton mode) on %s …", DEVICE)
        from omnivoice_triton import create_runner
        _runner = create_runner("triton")
        await loop.run_in_executor(None, _runner.load_model)
        _ref_audio_path, _ref_text = _pick_reference()
        _ready = True
        logger.info("OmniVoice-Triton ready. Reference: %s", _ref_audio_path or "none")
    except Exception:
        logger.exception("Model loading failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_load_model())
    yield
    if _runner:
        _runner.unload_model()


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
    decoding_steps: int = DEFAULT_STEPS,
) -> str:
    if override_audio_b64 and override_ref_text:
        audio_bytes = base64.b64decode(override_audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(audio_bytes)
        try:
            result = _runner.generate_voice_clone(
                text=text,
                ref_audio=tmp_path,
                ref_text=override_ref_text,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    elif _ref_audio_path:
        result = _runner.generate_voice_clone(
            text=text,
            ref_audio=_ref_audio_path,
            ref_text=_ref_text,
        )
    else:
        result = _runner.generate(text=text, decoding_steps=decoding_steps)

    return _ndarray_to_b64_wav(result["audio"], result["sample_rate"])


@app.get("/health")
def health():
    if not _ready:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "healthy"}


@app.post("/predict")
async def predict(request: PredictRequest):
    loop = asyncio.get_running_loop()
    predictions = []

    for instance in request.instances:
        text = instance.get("text", "").strip()
        if not text:
            predictions.append({"audio": ""})
            continue

        override_audio = instance.get("ref_audio", "")
        override_text = instance.get("ref_text", "")
        decoding_steps: int = instance.get(
            "decoding_steps",
            instance.get("steps", request.parameters.get("decoding_steps", DEFAULT_STEPS)),
        )

        audio_b64 = await loop.run_in_executor(
            None,
            _synthesize_one,
            text,
            override_audio,
            override_text,
            decoding_steps,
        )
        predictions.append({"audio": audio_b64})

    return {"predictions": predictions}
