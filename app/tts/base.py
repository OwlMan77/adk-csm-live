"""The contract every TTS engine speaks.

One shape, two consumers. The browser gets PCM to play; the face gets word
timings and cue tags to animate from. Both ride the same chunk so they cannot
drift apart.

Synthesis is an *async generator* rather than a coroutine returning bytes.
OmniVoice can only produce a whole utterance at once — it yields exactly one
chunk and that is fine — but Qwen3-TTS emits its first audio packet long before
the sentence is finished, and that difference is most of the perceived latency.
Keeping the generator shape for both means the pipeline above never has to know
which engine is underneath.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# The mood vocabulary. Small on purpose: it has to be reliably pickable by an
# LLM, mappable to a reference clip, and expressible as a face pose.
MOODS = ("neutral", "warm", "urgent", "amused", "concerned")
DEFAULT_MOOD = "neutral"


@dataclass(frozen=True, slots=True)
class WordTiming:
    """One word, in milliseconds relative to the start of its own sentence."""

    word: str
    start_ms: int
    end_ms: int

    def as_dict(self) -> dict[str, object]:
        return {"word": self.word, "startMs": self.start_ms, "endMs": self.end_ms}


@dataclass(frozen=True, slots=True)
class Utterance:
    """One sentence on its way to the speaker, with everything known about it."""

    text: str
    """What is actually spoken — cue tags already stripped."""

    mood: str = DEFAULT_MOOD
    tags: tuple[str, ...] = ()
    """Non-verbal cues found in the sentence, in the order they appeared."""

    index: int = 0
    """Sentence index within this reply. Also the chunk id the client sees."""


@dataclass(slots=True)
class AudioChunk:
    """A piece of one utterance's audio, plus whatever metadata is known by now."""

    pcm: bytes
    """Mono signed 16-bit little-endian. Empty when the chunk exists only to
    carry timings — engines emit one of those at the end of a sentence."""

    sample_rate: int
    utterance: Utterance
    words: tuple[WordTiming, ...] = ()
    """Sentence-relative word timings. Empty until the aligner has run, which
    for a streaming engine is after the last audio chunk of the sentence."""

    first: bool = False
    """True on the first chunk of an utterance — the one worth timing."""

    final: bool = False


@runtime_checkable
class TTSEngine(Protocol):
    """What `handle_message` is allowed to assume about a voice."""

    name: str

    def synthesize(self, utterance: Utterance) -> AsyncIterator[AudioChunk]: ...

    async def warm(self) -> bool:
        """Pull the backend out of a cold start. Cheap, idempotent, best effort."""
        ...

    async def aclose(self) -> None: ...


def engine_name() -> str:
    return os.environ.get("TTS_ENGINE", "omnivoice").strip().lower()
