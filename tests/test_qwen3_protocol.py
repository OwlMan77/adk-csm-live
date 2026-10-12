"""The vLLM-Omni streaming protocol, against a fake socket.

This is the highest-risk code in the swap and the code that cannot be exercised
without a GPU behind it, so the frame handling is pinned here instead. The shapes
below are taken from vLLM-Omni's own `serving_speech_stream.py` protocol
docstring: binary frames when word timestamps are off, JSON `audio.chunk` frames
carrying base64 PCM when they are on, and a trailing empty chunk holding the
sentence's alignment.
"""

import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
os.environ.setdefault("OMNIVOICE_ENDPOINT_ID", "test-endpoint")

from app.tts.base import Utterance
from app.tts.qwen3 import Qwen3TTSEngine, _Session

SPOKEN = Utterance(text="Hello there.", mood="warm", tags=("laughter",), index=0)


class FakeSocket:
    """Replays a scripted server side and records what the client sent."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.sent: list[dict] = []
        self.closed = False

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    async def recv(self):
        if not self._frames:
            raise AssertionError("client read past the end of the script")
        return self._frames.pop(0)

    async def close(self):
        self.closed = True


def drain(engine, session, utterance):
    async def go():
        return [chunk async for chunk in engine._run(session, utterance)]

    return asyncio.run(go())


def chunk_frame(**fields):
    base = {
        "type": "audio.chunk",
        "utterance_index": 0,
        "sentence_index": 0,
        "chunk_id": 0,
        "sample_rate": 24000,
        "audio_b64": "",
        "timestamps": None,
    }
    base.update(fields)
    return json.dumps(base)


def test_json_audio_chunks_are_decoded_in_order():
    ws = FakeSocket([
        json.dumps({"type": "audio.start", "sentence_text": "Hello there.",
                    "format": "pcm", "sample_rate": 24000}),
        chunk_frame(chunk_id=0, audio_b64=base64.b64encode(b"\x01\x02").decode()),
        chunk_frame(chunk_id=1, audio_b64=base64.b64encode(b"\x03\x04").decode()),
        json.dumps({"type": "audio.done"}),
        json.dumps({"type": "session.done", "total_sentences": 1}),
    ])
    chunks = drain(Qwen3TTSEngine(), _Session(ws=ws), SPOKEN)

    audio = [c for c in chunks if c.pcm]
    assert b"".join(c.pcm for c in audio) == b"\x01\x02\x03\x04"
    assert audio[0].first is True
    assert audio[1].first is False
    assert chunks[-1].final is True


def test_binary_frames_are_accepted_when_timings_are_off():
    # With word_timestamps=false the server sends raw binary instead of JSON.
    ws = FakeSocket([
        json.dumps({"type": "audio.start", "format": "pcm", "sample_rate": 24000}),
        b"\x05\x06\x07\x08",
        json.dumps({"type": "session.done"}),
    ])
    session = _Session(ws=ws, word_timestamps=False)
    engine = Qwen3TTSEngine()
    engine._aligned = False
    chunks = drain(engine, session, SPOKEN)

    assert b"".join(c.pcm for c in chunks) == b"\x05\x06\x07\x08"


def test_the_trailing_empty_chunk_carries_the_alignment():
    # The aligner runs on the finished sentence, so its timings arrive after all
    # the audio. That frame must still reach the face.
    ws = FakeSocket([
        json.dumps({"type": "audio.start", "format": "pcm", "sample_rate": 24000}),
        chunk_frame(audio_b64=base64.b64encode(b"\x01\x02").decode()),
        chunk_frame(chunk_id=1, audio_b64="", timestamps=[
            {"word": "Hello", "start_ms": 0, "end_ms": 320},
            {"word": "there", "start_ms": 320, "end_ms": 610},
        ]),
        json.dumps({"type": "session.done"}),
    ])
    chunks = drain(Qwen3TTSEngine(), _Session(ws=ws), SPOKEN)

    timed = [c for c in chunks if c.words]
    assert len(timed) == 1
    assert not timed[0].pcm
    assert [w.word for w in timed[0].words] == ["Hello", "there"]
    assert timed[0].words[1].start_ms == 320


def test_failed_alignment_is_dropped_rather_than_forwarded_as_silence():
    # timestamps: null means the aligner failed. An empty chunk with no timings
    # carries nothing and would look like a gap in playback.
    ws = FakeSocket([
        json.dumps({"type": "audio.start", "format": "pcm", "sample_rate": 24000}),
        chunk_frame(audio_b64=base64.b64encode(b"\x01\x02").decode()),
        chunk_frame(chunk_id=1, audio_b64="", timestamps=None),
        json.dumps({"type": "session.done"}),
    ])
    chunks = drain(Qwen3TTSEngine(), _Session(ws=ws), SPOKEN)
    assert sum(1 for c in chunks if not c.pcm and not c.words and not c.final) == 0


def test_the_sample_rate_the_server_reports_is_the_one_that_propagates():
    ws = FakeSocket([
        json.dumps({"type": "audio.start", "format": "pcm", "sample_rate": 8000}),
        chunk_frame(sample_rate=8000, audio_b64=base64.b64encode(b"\x01\x02").decode()),
        json.dumps({"type": "session.done"}),
    ])
    chunks = drain(Qwen3TTSEngine(), _Session(ws=ws), SPOKEN)
    assert all(c.sample_rate == 8000 for c in chunks)


def test_the_mood_selects_the_voice_in_the_session_config():
    ws = FakeSocket([
        json.dumps({"type": "audio.start", "format": "pcm", "sample_rate": 24000}),
        json.dumps({"type": "session.done"}),
    ])
    drain(Qwen3TTSEngine(), _Session(ws=ws), Utterance(text="Careful.", mood="urgent"))

    config = next(m for m in ws.sent if m["type"] == "session.config")
    assert config["voice"] == "hero_urgent"
    assert config["stream_audio"] is True
    # We split sentences upstream; letting the server split again only fragments
    # what is already one unit.
    assert config["split_granularity"] == "none"
    assert [m["type"] for m in ws.sent[-2:]] == ["input.text", "input.done"]


def test_config_is_not_resent_when_the_mood_has_not_moved():
    # The session config is sticky across flushes. Resending it every sentence
    # would be a wasted round trip on the latency path.
    ws = FakeSocket([
        json.dumps({"type": "audio.start", "format": "pcm", "sample_rate": 24000}),
        json.dumps({"type": "session.done"}),
    ])
    session = _Session(ws=ws)
    engine = Qwen3TTSEngine()
    drain(engine, session, Utterance(text="One.", mood="warm"))

    ws._frames = [
        json.dumps({"type": "audio.start", "format": "pcm", "sample_rate": 24000}),
        json.dumps({"type": "session.done"}),
    ]
    before = len([m for m in ws.sent if m["type"] == "session.config"])
    drain(engine, session, Utterance(text="Two.", mood="warm", index=1))
    after = len([m for m in ws.sent if m["type"] == "session.config"])
    assert after == before


def test_a_missing_aligner_disables_timings_instead_of_killing_the_voice():
    ws = FakeSocket([
        json.dumps({
            "type": "error",
            "message": "word_timestamps=true but the server was launched without --forced-aligner",
        }),
    ])
    engine = Qwen3TTSEngine()
    with pytest.raises(RuntimeError):
        drain(engine, _Session(ws=ws), SPOKEN)

    # The next sentence asks for audio only; the face falls back to analysis,
    # which is why it was built to work without timings.
    assert engine._aligned is False


def test_no_credentials_means_no_headers():
    # A local vLLM-Omni wants none, and sending empty ones would be worse than
    # sending nothing.
    import app.tts.qwen3 as q

    assert q._auth_headers() == {}


def test_modal_proxy_auth_is_sent_as_a_header_pair(monkeypatch):
    # @app.server defaults to unauthenticated=False, so Modal's proxy answers
    # 401 to anonymous callers. Without this the whole voice path is dead, and
    # it fails looking exactly like a cold start.
    import app.tts.qwen3 as q

    monkeypatch.setattr(q, "AUTH_BEARER", "")
    monkeypatch.setattr(q, "AUTH_KEY", "wk-abc")
    monkeypatch.setattr(q, "AUTH_SECRET", "ws-xyz")
    assert q._auth_headers() == {"Modal-Key": "wk-abc", "Modal-Secret": "ws-xyz"}


def test_half_a_credential_pair_is_not_sent(monkeypatch):
    # A key with no secret is a misconfiguration. Sending it would earn a 401
    # that looks identical to sending nothing, so fail the same way either way.
    import app.tts.qwen3 as q

    monkeypatch.setattr(q, "AUTH_BEARER", "")
    monkeypatch.setattr(q, "AUTH_KEY", "wk-abc")
    monkeypatch.setattr(q, "AUTH_SECRET", "")
    assert q._auth_headers() == {}


def test_bearer_token_gets_its_scheme_added_once(monkeypatch):
    import app.tts.qwen3 as q

    monkeypatch.setattr(q, "AUTH_BEARER", "wk-abc.ws-xyz")
    assert q._auth_headers() == {"Authorization": "Bearer wk-abc.ws-xyz"}

    monkeypatch.setattr(q, "AUTH_BEARER", "Bearer wk-abc.ws-xyz")
    assert q._auth_headers() == {"Authorization": "Bearer wk-abc.ws-xyz"}


def test_an_https_endpoint_upgrades_the_websocket_scheme(monkeypatch):
    # Modal and Cloud Run both serve https; ws:// against them fails at the
    # handshake rather than anywhere informative.
    import app.tts.qwen3 as q

    monkeypatch.setattr(q, "BASE_URL", "https://ws--qwen3-tts-server.modal.run")
    assert q._ws_url() == "wss://ws--qwen3-tts-server.modal.run/v1/audio/speech/stream"

    monkeypatch.setattr(q, "BASE_URL", "http://localhost:8091")
    assert q._ws_url() == "ws://localhost:8091/v1/audio/speech/stream"
