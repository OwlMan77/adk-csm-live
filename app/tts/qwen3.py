"""Qwen3-TTS-12Hz-1.7B-Base served by vLLM-Omni.

Why this engine: Apache-2.0 weights (OmniVoice's are CC-BY-NC), cloning from a
3-second reference, a first audio packet long before the sentence is finished,
and — the part the face needs — word-level timings from the forced aligner.

Why the WebSocket and not `POST /v1/audio/speech`: on the REST path
`word_timestamps` and `stream=true` are mutually exclusive, and we want both.
`/v1/audio/speech/stream` is the only transport that streams audio *and* aligns
it, which is exactly the hybrid the face was built for — timings for when a word
starts, spectrum for what the mouth is doing.

One caveat worth stating plainly: the aligner runs on the finished sentence, so
its timings arrive in a trailing empty chunk *after* the last audio chunk. That
is still early enough to be useful, because synthesis runs faster than playback
and the browser holds a ring buffer — the timings land while the sentence is
still queued, not while it is being heard. This is the whole reason the face
schedules on the audio clock rather than on arrival time.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import websockets

from app.tts.base import AudioChunk, Utterance, WordTiming
from app.tts.voices import for_mood

logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("QWEN_TTS_URL", "http://localhost:8091").rstrip("/")
MODEL = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
LANGUAGE = os.environ.get("QWEN_TTS_LANGUAGE", "English")
MAX_NEW_TOKENS = int(os.environ.get("QWEN_TTS_MAX_NEW_TOKENS", "2048"))

# One socket per concurrent sentence. The pipeline above synthesises sentence
# N+1 while N is still sending, so a single sticky session would interleave two
# flushes on one connection.
POOL_SIZE = int(os.environ.get("QWEN_TTS_POOL_SIZE", "2"))

# A cold Cloud Run GPU instance takes 5-20 s to answer; a warm one answers in
# tens of milliseconds. The long timeout only ever applies to the first request
# after a scale-up.
CONNECT_TIMEOUT_S = float(os.environ.get("QWEN_TTS_CONNECT_TIMEOUT_S", "60"))
CHUNK_TIMEOUT_S = float(os.environ.get("QWEN_TTS_CHUNK_TIMEOUT_S", "30"))

# Credentials for a private endpoint. A GPU box on the open internet is a
# standing invitation to burn someone else's credits, so the Modal deployment
# keeps proxy auth on (`@app.server` defaults to `unauthenticated=False`) and
# rejects anonymous callers with 401 — which the orchestrator has to satisfy or
# nothing works at all.
#
# Two shapes, both from `modal token new`:
#   QWEN_TTS_KEY / QWEN_TTS_SECRET -> Modal-Key / Modal-Secret headers
#   QWEN_TTS_BEARER                -> Authorization: Bearer wk-<id>.ws-<secret>
# The bearer form also covers a Cloud Run service behind IAM, where the value is
# an identity token. Unset for a local vLLM-Omni, which wants no auth.
AUTH_KEY = os.environ.get("QWEN_TTS_KEY", "").strip()
AUTH_SECRET = os.environ.get("QWEN_TTS_SECRET", "").strip()
AUTH_BEARER = os.environ.get("QWEN_TTS_BEARER", "").strip()


def _auth_headers() -> dict[str, str]:
    if AUTH_BEARER:
        return {"Authorization": AUTH_BEARER if AUTH_BEARER.startswith("Bearer ") else f"Bearer {AUTH_BEARER}"}
    if AUTH_KEY and AUTH_SECRET:
        return {"Modal-Key": AUTH_KEY, "Modal-Secret": AUTH_SECRET}
    return {}


def _ws_url() -> str:
    scheme = "wss" if BASE_URL.startswith("https://") else "ws"
    host = BASE_URL.split("://", 1)[-1]
    return f"{scheme}://{host}/v1/audio/speech/stream"


@dataclass
class _Session:
    """One open socket and the config it is currently carrying."""

    ws: websockets.ClientConnection
    voice: str | None = None
    word_timestamps: bool = True


class Qwen3TTSEngine:
    name = "qwen3-tts"

    def __init__(self) -> None:
        self._idle: list[_Session] = []
        self._open = 0
        self._slots = asyncio.Semaphore(POOL_SIZE)
        self._lock = asyncio.Lock()
        # Flipped off permanently if the server was launched without
        # --forced-aligner. The face then falls back to pure audio analysis,
        # which is why it was built to work without timings in the first place.
        self._aligned = True

    # ── connection pool ───────────────────────────────────────────────────────

    async def _acquire(self) -> _Session:
        async with self._lock:
            if self._idle:
                return self._idle.pop()
            self._open += 1
        try:
            ws = await asyncio.wait_for(
                websockets.connect(
                    _ws_url(),
                    max_size=None,
                    additional_headers=_auth_headers() or None,
                ),
                timeout=CONNECT_TIMEOUT_S,
            )
        except Exception:
            async with self._lock:
                self._open -= 1
            raise
        return _Session(ws=ws, word_timestamps=self._aligned)

    async def _release(self, session: _Session) -> None:
        async with self._lock:
            self._idle.append(session)

    async def _discard(self, session: _Session) -> None:
        async with self._lock:
            self._open -= 1
        with contextlib.suppress(Exception):
            await session.ws.close()

    # ── synthesis ─────────────────────────────────────────────────────────────

    async def synthesize(self, utterance: Utterance) -> AsyncIterator[AudioChunk]:
        await self._slots.acquire()
        session: _Session | None = None
        healthy = False
        try:
            session = await self._acquire()
            async for chunk in self._run(session, utterance):
                yield chunk
            healthy = True
        finally:
            self._slots.release()
            if session is not None:
                # A socket abandoned mid-utterance still has frames in flight;
                # reusing it would deliver the previous sentence's tail into the
                # next one. Barge-in takes this path, so it has to be the safe one.
                if healthy:
                    await self._release(session)
                else:
                    await self._discard(session)

    async def _run(self, session: _Session, utterance: Utterance) -> AsyncIterator[AudioChunk]:
        want_timestamps = session.word_timestamps and self._aligned
        ref = for_mood(utterance.mood)
        voice = ref.name if ref else None

        # The session config is sticky across flushes, so only resend it when
        # the mood has actually moved the voice.
        if voice != session.voice or want_timestamps != session.word_timestamps:
            config: dict[str, object] = {
                "type": "session.config",
                "model": MODEL,
                "response_format": "pcm",
                "language": LANGUAGE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "stream_audio": True,
                "word_timestamps": want_timestamps,
                # We split sentences upstream so each flush is already one unit;
                # letting the server split again would only fragment it further.
                "split_granularity": "none",
            }
            if voice:
                config["voice"] = voice
            await session.ws.send(json.dumps(config))
            session.voice = voice
            session.word_timestamps = want_timestamps

        await session.ws.send(json.dumps({"type": "input.text", "text": utterance.text}))
        await session.ws.send(json.dumps({"type": "input.done"}))

        sample_rate = 24000
        first = True

        while True:
            raw = await asyncio.wait_for(session.ws.recv(), timeout=CHUNK_TIMEOUT_S)

            if isinstance(raw, (bytes, bytearray)):
                # Binary frame: raw PCM, the shape used when timings are off.
                yield AudioChunk(
                    pcm=bytes(raw),
                    sample_rate=sample_rate,
                    utterance=utterance,
                    first=first,
                )
                first = False
                continue

            msg = json.loads(raw)
            kind = msg.get("type")

            if kind == "audio.start":
                sample_rate = int(msg.get("sample_rate") or sample_rate)

            elif kind == "audio.chunk":
                sample_rate = int(msg.get("sample_rate") or sample_rate)
                pcm = base64.b64decode(msg.get("audio_b64") or "")
                stamps = msg.get("timestamps")
                words = tuple(
                    WordTiming(
                        word=str(w.get("word", "")),
                        start_ms=int(w.get("start_ms", 0)),
                        end_ms=int(w.get("end_ms", 0)),
                    )
                    for w in stamps
                    if isinstance(w, dict)
                ) if isinstance(stamps, list) else ()

                if not pcm and not words:
                    # Alignment failed (timestamps: null) or the sentence was
                    # silence. Nothing to send.
                    continue

                yield AudioChunk(
                    pcm=pcm,
                    sample_rate=sample_rate,
                    utterance=utterance,
                    words=words,
                    first=first and bool(pcm),
                )
                if pcm:
                    first = False

            elif kind == "session.done":
                yield AudioChunk(
                    pcm=b"",
                    sample_rate=sample_rate,
                    utterance=utterance,
                    final=True,
                )
                return

            elif kind == "error":
                message = str(msg.get("message", ""))
                if want_timestamps and "aligner" in message.lower():
                    # The server was launched without --forced-aligner. Say so
                    # once, then run without timings for the rest of the process.
                    logger.warning(
                        "vLLM-Omni has no forced aligner; word timings disabled "
                        "for this session (%s)", message,
                    )
                    self._aligned = False
                raise RuntimeError(f"vLLM-Omni: {message}")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def warm(self) -> bool:
        """Fired while the user is still reading the live dialog, so the 5-20 s
        cold start happens during human time rather than in the first reply."""
        async with httpx.AsyncClient(timeout=CONNECT_TIMEOUT_S, headers=_auth_headers()) as client:
            for path in ("/health", "/v1/audio/voices"):
                try:
                    response = await client.get(f"{BASE_URL}{path}")
                except httpx.HTTPError as exc:
                    logger.debug("warm %s failed: %s", path, exc)
                    continue
                if response.status_code in (401, 403):
                    # Worth saying out loud: this looks exactly like a cold start
                    # from the outside, and the face would sit in its waking pose
                    # forever waiting for a voice that is never coming.
                    logger.error(
                        "%s rejected the warm ping with %d — the endpoint needs "
                        "credentials. Set QWEN_TTS_KEY/QWEN_TTS_SECRET (from "
                        "`modal token new`), or deploy it unauthenticated.",
                        BASE_URL, response.status_code,
                    )
                    return False
                if response.status_code < 400:
                    return True
        return False

    async def aclose(self) -> None:
        async with self._lock:
            sessions, self._idle, self._open = self._idle, [], 0
        for session in sessions:
            with contextlib.suppress(Exception):
                await session.ws.send(json.dumps({"type": "session.close"}))
                await session.ws.close()
