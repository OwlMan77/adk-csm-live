"""The wire frame, and the engine contract underneath it.

The frontend is the other half of this and lives in a different repo, so the
shape is pinned here rather than trusted.
"""

import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
os.environ.setdefault("OMNIVOICE_ENDPOINT_ID", "test-endpoint")

from app.main import audio_frame
from app.tts.base import AudioChunk, Utterance, WordTiming
from app.tts.voices import for_mood, references

SPOKEN = Utterance(text="Hello there.", mood="warm", tags=("laughter",), index=2)


def test_audio_frame_carries_both_halves():
    frame = json.loads(audio_frame(AudioChunk(
        pcm=b"\x01\x02\x03\x04",
        sample_rate=24000,
        utterance=SPOKEN,
        words=(WordTiming("Hello", 0, 320), WordTiming("there", 320, 610)),
        first=True,
    )))

    inline = frame["content"]["parts"][0]["inlineData"]
    assert inline["mimeType"] == "audio/pcm;rate=24000"
    assert base64.b64decode(inline["data"]) == b"\x01\x02\x03\x04"

    cue = frame["cue"]
    assert cue["chunkId"] == 2
    assert cue["mood"] == "warm"
    assert cue["tags"] == ["laughter"]
    assert cue["first"] is True
    assert cue["words"][1] == {"word": "there", "startMs": 320, "endMs": 610}


def test_a_timings_only_chunk_carries_no_audio_part():
    # The aligner runs on the finished sentence, so its timings arrive in a
    # trailing empty chunk. That frame must not look like silence to play.
    frame = json.loads(audio_frame(AudioChunk(
        pcm=b"",
        sample_rate=24000,
        utterance=SPOKEN,
        words=(WordTiming("Hello", 0, 320),),
    )))
    assert "content" not in frame
    assert frame["cue"]["words"]


def test_mime_type_follows_the_engine_not_a_constant():
    # The client used to hardcode 24 kHz. The rate the engine actually used has
    # to survive to the browser or playback runs at the wrong speed.
    frame = json.loads(audio_frame(AudioChunk(
        pcm=b"\x00\x00", sample_rate=16000, utterance=SPOKEN,
    )))
    assert frame["content"]["parts"][0]["inlineData"]["mimeType"] == "audio/pcm;rate=16000"
    assert frame["cue"]["sampleRate"] == 16000


def test_every_mood_resolves_to_a_reference():
    from app.tts.base import MOODS

    for mood in MOODS:
        assert for_mood(mood) is not None, mood


def test_an_unknown_mood_still_gets_a_voice():
    assert for_mood("nonsense") is not None


def test_reference_names_are_unique_per_mood():
    refs = references()
    assert len({r.name for r in refs.values()}) == len(refs)


def test_engine_yields_chunks_not_a_blob():
    """The pipeline consumes an async iterator; a coroutine returning bytes
    would deadlock it. Checked structurally because neither engine can run
    without a GPU behind it."""
    from app.tts.omnivoice import OmniVoiceEngine
    from app.tts.qwen3 import Qwen3TTSEngine

    for engine in (OmniVoiceEngine(), Qwen3TTSEngine()):
        gen = engine.synthesize(SPOKEN)
        assert hasattr(gen, "__anext__"), engine.name
