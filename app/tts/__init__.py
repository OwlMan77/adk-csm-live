"""Speech synthesis, one engine at a time, chosen by `TTS_ENGINE`.

    TTS_ENGINE=omnivoice   k2-fsa/OmniVoice on a Vertex AI endpoint (default)
    TTS_ENGINE=qwen3-tts   Qwen3-TTS-12Hz-1.7B-Base on vLLM-Omni

The switch stays until Qwen3-TTS has won on both first-byte latency *and* the
voice actually sounding like the reference — decided by ear against
scripts/bench_tts.py, not by argument. If it wins on latency and loses on the
voice, that is a real finding and the cascade stays where it is.
"""

from __future__ import annotations

import logging

from app.tts.base import (
    DEFAULT_MOOD,
    MOODS,
    AudioChunk,
    TTSEngine,
    Utterance,
    WordTiming,
    engine_name,
)

logger = logging.getLogger(__name__)

_engine: TTSEngine | None = None


def get_engine() -> TTSEngine:
    """The process-wide engine. Built on first use so importing is cheap and so
    a missing endpoint id fails at the first request rather than at import."""
    global _engine
    if _engine is not None:
        return _engine

    name = engine_name()
    if name in ("qwen3", "qwen3-tts", "qwen"):
        from app.tts.qwen3 import Qwen3TTSEngine

        _engine = Qwen3TTSEngine()
    else:
        if name != "omnivoice":
            logger.warning("Unknown TTS_ENGINE %r — falling back to omnivoice", name)
        from app.tts.omnivoice import OmniVoiceEngine

        _engine = OmniVoiceEngine()

    logger.info("TTS engine: %s", _engine.name)
    return _engine


async def close_engine() -> None:
    global _engine
    if _engine is not None:
        await _engine.aclose()
        _engine = None


__all__ = [
    "DEFAULT_MOOD",
    "MOODS",
    "AudioChunk",
    "TTSEngine",
    "Utterance",
    "WordTiming",
    "close_engine",
    "get_engine",
]
