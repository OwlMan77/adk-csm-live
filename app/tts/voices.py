"""Mood → reference voice.

Qwen3-TTS's cloning checkpoint (`-Base`) has no `instruct` parameter: natural
language prosody control lives in the CustomVoice and VoiceDesign checkpoints,
and neither of those clones. So on the cloning path the reference clip *is* the
emotion control, and picking a clip per mood is how a sigh sounds different from
a warning.

The mapping lives in `scripts/samples/voices.json` so it can be corrected by ear
without touching code.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.tts.base import DEFAULT_MOOD, MOODS

logger = logging.getLogger(__name__)

SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", str(Path(__file__).parents[2] / "scripts" / "samples")))
MANIFEST = SAMPLES_DIR / "voices.json"

# Voice profiles are precomputed and baked into the serving image under these
# names (see scripts/precompute_voices.py).
VOICE_PREFIX = os.environ.get("TTS_VOICE_PREFIX", "hero")


@dataclass(frozen=True, slots=True)
class Reference:
    mood: str
    name: str
    """The precomputed voice name the server knows, e.g. "hero_urgent"."""

    wav: Path
    text: str


@lru_cache(maxsize=1)
def references() -> dict[str, Reference]:
    """Load the mood → clip map. Falls back to a single neutral voice."""
    mapping: dict[str, str] = {}
    if MANIFEST.exists():
        try:
            data = json.loads(MANIFEST.read_text(encoding="utf-8"))
            for mood, entry in (data.get("voices") or {}).items():
                clip = entry.get("clip") if isinstance(entry, dict) else entry
                if mood in MOODS and clip:
                    mapping[mood] = str(clip)
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning("Could not read %s (%s) — falling back to one voice", MANIFEST, exc)

    out: dict[str, Reference] = {}
    for mood, clip in mapping.items():
        wav = SAMPLES_DIR / f"{clip}.wav"
        txt = SAMPLES_DIR / f"{clip}.txt"
        if not (wav.exists() and txt.exists()):
            logger.warning("Reference clip %s for mood %r is missing — skipping", clip, mood)
            continue
        out[mood] = Reference(
            mood=mood,
            name=f"{VOICE_PREFIX}_{mood}",
            wav=wav,
            text=txt.read_text(encoding="utf-8").strip(),
        )

    if not out:
        # No manifest, or nothing in it resolved. Take the first complete pair
        # so the thing still speaks rather than failing on a config problem.
        for txt in sorted(SAMPLES_DIR.glob("*.txt")):
            wav = txt.with_suffix(".wav")
            if wav.exists():
                out[DEFAULT_MOOD] = Reference(
                    mood=DEFAULT_MOOD,
                    name=f"{VOICE_PREFIX}_{DEFAULT_MOOD}",
                    wav=wav,
                    text=txt.read_text(encoding="utf-8").strip(),
                )
                break

    return out


def for_mood(mood: str) -> Reference | None:
    """The reference for `mood`, or the neutral one, or nothing at all."""
    refs = references()
    return refs.get(mood) or refs.get(DEFAULT_MOOD) or next(iter(refs.values()), None)
