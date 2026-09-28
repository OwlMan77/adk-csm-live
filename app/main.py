"""FastAPI server: bidi-streaming voice agent pipeline.

WebSocket bidi flow (no Gemini Live API required):
  1. Browser AudioWorklet streams raw 16kHz Int16 PCM continuously.
  2. Server VAD buffers chunks; silence after speech triggers an utterance.
  3. Utterance PCM is sent to gemini-2.5-flash via run_async().
  4. Streaming ADK events are forwarded to the browser as they arrive.
  5. Agent text is split at sentence boundaries; each sentence has its cue
     tags and mood extracted and is queued for synthesis.
  6. Audio chunks are forwarded to the browser as they arrive, each carrying the
     cues, the mood and (where the engine provides them) word timings, so the
     face animates off the same sentence the ear is hearing.

REST endpoints support text chat via SSE for non-live agents.
"""

import asyncio
import base64
import io
import json
import logging
import os
import re
import struct
import uuid
import warnings
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel

load_dotenv(Path(__file__).parent.parent / ".env")

from app.csm_agent.agent import agent, create_agent
from app.car_seller_agent.agent import root_agent as car_seller_agent
from app.cues import extract
from app.tts import AudioChunk, Utterance, close_engine, get_engine

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

APP_NAME = "adk-csm-live"
CAR_SELLER_APP_NAME = "car-seller"

@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await close_engine()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
session_service = InMemorySessionService()
runner = Runner(app_name=APP_NAME, agent=agent, session_service=session_service)
car_seller_runner = Runner(app_name=CAR_SELLER_APP_NAME, agent=car_seller_agent, session_service=session_service)

_RUNNERS: dict[str, Runner] = {
    APP_NAME: runner,
    CAR_SELLER_APP_NAME: car_seller_runner,
}

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# VAD tuning
_VAD_SAMPLE_RATE = 16000
_VAD_ENERGY_THRESHOLD = 600      # Int16 RMS; below = silence
_VAD_SILENCE_MS = 400            # ms of silence that closes an utterance
_VAD_MIN_SPEECH_MS = 150         # ignore clips shorter than this
_VAD_SILENCE_BYTES = int(_VAD_SILENCE_MS * _VAD_SAMPLE_RATE / 1000) * 2
_VAD_MIN_BYTES = int(_VAD_MIN_SPEECH_MS * _VAD_SAMPLE_RATE / 1000) * 2

# How long to wait for the next audio chunk of a sentence already in flight.
# Generous: it has to cover a cold GPU on the very first sentence.
_TTS_CHUNK_TIMEOUT_S = float(os.environ.get("TTS_CHUNK_TIMEOUT_S", "45"))


def _rms(pcm_bytes: bytes) -> float:
    count = len(pcm_bytes) // 2
    if count == 0:
        return 0.0
    samples = struct.unpack_from(f"<{count}h", pcm_bytes)
    return (sum(s * s for s in samples) / count) ** 0.5


def _split_sentences(text: str) -> tuple[list[str], str]:
    parts = _SENTENCE_BOUNDARY.split(text)
    if len(parts) <= 1:
        return [], text
    return parts[:-1], parts[-1]


def audio_frame(chunk: AudioChunk) -> str:
    """One wire frame. Audio for the ear, cues and timings for the face.

    The cue block is a sibling of `content`, not a part of it, so a client
    that only knows about audio ignores it without noticing."""
    payload: dict[str, Any] = {
        "cue": {
            "chunkId": chunk.utterance.index,
            "mood": chunk.utterance.mood,
            "tags": list(chunk.utterance.tags),
            "text": chunk.utterance.text,
            "sampleRate": chunk.sample_rate,
            "first": chunk.first,
            "final": chunk.final,
            "words": [w.as_dict() for w in chunk.words],
        }
    }
    if chunk.pcm:
        payload["content"] = {
            "parts": [{
                "inlineData": {
                    "mimeType": f"audio/pcm;rate={chunk.sample_rate}",
                    "data": base64.b64encode(chunk.pcm).decode(),
                }
            }]
        }
    return json.dumps(payload)


@app.get("/warm")
async def warm() -> dict[str, object]:
    """Wake the speech backend before anyone needs it.

    The frontend fires this the moment the live agent is selected — while the
    start dialog is still on screen and the user is reading it and granting mic
    access. That is several seconds of human time that would otherwise be spent
    waiting for a GPU to come up, and it covers most of a Cloud Run cold start.
    Safe to call repeatedly; safe to ignore the answer."""
    engine = get_engine()
    try:
        ready = await engine.warm()
    except Exception as exc:  # noqa: BLE001 — warming must never fail a request
        logger.debug("warm failed: %s", exc)
        ready = False
    return {"engine": engine.name, "ready": ready}


@app.get("/list-apps")
async def list_apps() -> list[dict]:
    return [
        {"name": APP_NAME, "live": True},
        {"name": CAR_SELLER_APP_NAME, "live": False},
    ]


@app.post("/apps/{app_name}/users/{user_id}/sessions")
async def create_session(app_name: str, user_id: str) -> dict[str, str]:
    session_id = str(uuid.uuid4())
    resolved = app_name if app_name in _RUNNERS else APP_NAME
    await session_service.create_session(
        app_name=resolved, user_id=user_id, session_id=session_id
    )
    return {"id": session_id}


class MessagePart(BaseModel):
    text: str | None = None
    inlineData: dict[str, Any] | None = None


class NewMessage(BaseModel):
    role: str
    parts: list[MessagePart]


class RunRequest(BaseModel):
    appName: str
    userId: str
    sessionId: str
    newMessage: NewMessage
    streaming: bool = True


@app.post("/run_sse")
async def run_sse(request: RunRequest) -> StreamingResponse:
    parts = []
    for p in request.newMessage.parts:
        if p.text:
            parts.append(types.Part(text=p.text))
        elif p.inlineData:
            parts.append(types.Part(
                inline_data=types.Blob(
                    mime_type=p.inlineData["mimeType"],
                    data=base64.b64decode(p.inlineData["data"]),
                )
            ))

    message = types.Content(role=request.newMessage.role, parts=parts)
    sse_runner = _RUNNERS.get(request.appName, runner)

    async def event_stream():
        # Keepalive: ping every 15s so the connection doesn't drop during long Gemini calls
        async def _run_agent(queue: asyncio.Queue) -> None:
            try:
                async for event in sse_runner.run_async(
                    user_id=request.userId,
                    session_id=request.sessionId,
                    new_message=message,
                ):
                    await queue.put(("event", event))
            except Exception as exc:
                await queue.put(("error", exc))
            finally:
                await queue.put(("done", None))

        q: asyncio.Queue = asyncio.Queue()
        agent_task = asyncio.create_task(_run_agent(q))

        try:
            while True:
                try:
                    kind, payload = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                if kind == "done":
                    break
                if kind == "error":
                    logger.error("SSE stream error: %s", payload, exc_info=True)
                    break

                event = payload
                event_json = event.model_dump_json(exclude_none=True, by_alias=True)
                yield f"data: {event_json}\n\n"

                data = json.loads(event_json)
                parts_data = (
                    data.get("content", {}).get("parts", [])
                    if isinstance(data.get("content"), dict) else []
                )
                text = "".join(p.get("text", "") for p in parts_data if isinstance(p, dict))
        finally:
            agent_task.cancel()

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    session_id: str,
) -> None:
    await websocket.accept()
    logger.debug("WebSocket accepted: user=%s session=%s", user_id, session_id)

    session = await session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    if not session:
        await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )

    ws_agent = create_agent()
    ws_runner = Runner(app_name=APP_NAME, agent=ws_agent, session_service=session_service)
    logger.debug("Model: %s", ws_agent.model)

    # Queue of Content objects to process sequentially.
    msg_queue: asyncio.Queue[types.Content | None] = asyncio.Queue()

    # ── VAD state ─────────────────────────────────────────────────────────────

    audio_buf = bytearray()
    silence_bytes = 0
    has_speech = False

    def vad_feed(chunk: bytes) -> types.Content | None:
        """Feed a PCM chunk; returns a Content when an utterance is complete."""
        nonlocal audio_buf, silence_bytes, has_speech

        energy = _rms(chunk)

        if energy > _VAD_ENERGY_THRESHOLD:
            has_speech = True
            silence_bytes = 0
            audio_buf.extend(chunk)
            return None

        if has_speech:
            audio_buf.extend(chunk)
            silence_bytes += len(chunk)
            if silence_bytes >= _VAD_SILENCE_BYTES:
                pcm = bytes(audio_buf)
                audio_buf.clear()
                silence_bytes = 0
                has_speech = False
                if len(pcm) < _VAD_MIN_BYTES:
                    return None
                wav_buf = io.BytesIO()
                with wave.open(wav_buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(_VAD_SAMPLE_RATE)
                    wf.writeframes(pcm)
                return types.Content(role="user", parts=[
                    types.Part(inline_data=types.Blob(
                        mime_type="audio/wav",
                        data=wav_buf.getvalue(),
                    ))
                ])

        return None

    # ── Message handler ────────────────────────────────────────────────────────

    async def handle_message(content: types.Content) -> None:
        engine = get_engine()
        tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
        buffer = ""

        # Pipeline: a draining task is started the moment a sentence arrives, so
        # synthesis for sentence N+1 overlaps with sending sentence N. Each task
        # writes into its own chunk queue, which keeps playback order intact
        # while letting the engines run concurrently.
        Job = tuple[Utterance, asyncio.Queue | None, asyncio.Task | None]
        job_queue: asyncio.Queue[Job | None] = asyncio.Queue()

        async def drain(utterance: Utterance, chunks: asyncio.Queue) -> None:
            try:
                async for chunk in engine.synthesize(utterance):
                    await chunks.put(chunk)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — reported on the consumer side
                await chunks.put(exc)
            finally:
                await chunks.put(None)

        async def tts_producer() -> None:
            index = 0
            while True:
                text = await tts_queue.get()
                if text is None:
                    await job_queue.put(None)
                    break

                utterance = extract(text, index=index)
                index += 1

                if not utterance.text:
                    # A sentence that was nothing but cues — "[laughter]" on its
                    # own. There is no audio to make, but the face should still
                    # get it, in order, where it happened.
                    if utterance.tags:
                        await job_queue.put((utterance, None, None))
                    continue

                chunks: asyncio.Queue = asyncio.Queue()
                await job_queue.put(
                    (utterance, chunks, asyncio.create_task(drain(utterance, chunks)))
                )

        async def tts_task() -> None:
            producer = asyncio.create_task(tts_producer())
            started: list[asyncio.Task] = []
            try:
                while True:
                    job = await job_queue.get()
                    if job is None:
                        break
                    utterance, chunks, task = job

                    if chunks is None:
                        # Cue-only: no audio, but the face still gets its moment.
                        await websocket.send_text(audio_frame(AudioChunk(
                            pcm=b"",
                            sample_rate=0,
                            utterance=utterance,
                            first=True,
                            final=True,
                        )))
                        continue

                    if task is not None:
                        started.append(task)

                    while True:
                        try:
                            item = await asyncio.wait_for(chunks.get(), timeout=_TTS_CHUNK_TIMEOUT_S)
                        except asyncio.TimeoutError:
                            logger.error("TTS stalled on sentence %d", utterance.index)
                            break
                        if item is None:
                            break
                        if isinstance(item, Exception):
                            logger.error("%s synthesis failed: %s", engine.name, item)
                            break
                        await websocket.send_text(audio_frame(item))
            finally:
                producer.cancel()
                # Barge-in cancels this task mid-sentence. Sentences the producer
                # had already started but nobody has consumed are still running
                # on the GPU and still holding an engine connection, so they have
                # to be cancelled too — not just the one being sent.
                while not job_queue.empty():
                    queued = job_queue.get_nowait()
                    if queued and queued[2] is not None:
                        started.append(queued[2])
                for task in started:
                    task.cancel()
                for pending in (producer, *started):
                    try:
                        await pending
                    except (asyncio.CancelledError, Exception):
                        pass

        tts = asyncio.create_task(tts_task())

        _interrupted = False
        try:
            async for event in ws_runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content,
            ):
                event_json = event.model_dump_json(exclude_none=True, by_alias=True)
                try:
                    await websocket.send_text(event_json)
                except RuntimeError:
                    break

                if event.author != "user":
                    event_dict = json.loads(event_json)
                    content_data = event_dict.get("content")
                    if isinstance(content_data, dict):
                        for part in content_data.get("parts", []):
                            if isinstance(part, dict) and (t := part.get("text", "")):
                                buffer += t
                                sentences, buffer = _split_sentences(buffer)
                                for s in sentences:
                                    if s.strip():
                                        await tts_queue.put(s.strip())

                    if not event_dict.get("partial", True) and buffer.strip():
                        await tts_queue.put(buffer.strip())
                        buffer = ""
        except asyncio.CancelledError:
            _interrupted = True
            raise
        finally:
            if _interrupted:
                # Interrupted by a new utterance — kill TTS immediately
                tts.cancel()
            else:
                # Normal completion — flush buffer and wait for TTS to finish
                if buffer.strip():
                    tts_queue.put_nowait(buffer.strip())
                tts_queue.put_nowait(None)
                try:
                    await asyncio.wait_for(tts, timeout=30.0)
                except (asyncio.TimeoutError, Exception):
                    tts.cancel()

    # ── Process loop (interruptible) ──────────────────────────────────────────

    async def process_loop() -> None:
        current: asyncio.Task | None = None
        while True:
            content = await msg_queue.get()
            if content is None:
                if current and not current.done():
                    current.cancel()
                    try:
                        await current
                    except (asyncio.CancelledError, Exception):
                        pass
                break

            # Drain queue — skip to the latest utterance if several queued up
            while not msg_queue.empty():
                next_item = msg_queue.get_nowait()
                if next_item is None:
                    content = None
                    break
                content = next_item
            if content is None:
                break

            # Cancel in-flight response
            if current and not current.done():
                current.cancel()
                try:
                    await current
                except (asyncio.CancelledError, Exception):
                    pass

            current = asyncio.create_task(_safe_handle(content))

    async def _safe_handle(content: types.Content) -> None:
        try:
            await handle_message(content)
        except asyncio.CancelledError:
            logger.debug("handle_message interrupted by new utterance")
            raise
        except Exception as exc:
            logger.error("handle_message error: %s", exc, exc_info=True)

    processor = asyncio.create_task(process_loop())

    # ── Receive loop ───────────────────────────────────────────────────────────

    try:
        while True:
            message = await websocket.receive()

            # The low-level receive() *returns* a disconnect message rather than
            # raising WebSocketDisconnect -- only the typed receive_bytes() and
            # receive_text() helpers raise. Without this branch the next
            # iteration calls receive() again, Starlette raises RuntimeError,
            # and an ordinary hang-up lands in the generic handler below as an
            # ERROR with a traceback. Closing a tab is not an error.
            if message["type"] == "websocket.disconnect":
                logger.debug("Client disconnected: user=%s session=%s", user_id, session_id)
                break

            if "bytes" in message:
                chunk = message["bytes"]
                rms = _rms(chunk)
                if rms > 50:  # log any non-trivial audio
                    logger.debug("PCM chunk: %d bytes, RMS=%.1f", len(chunk), rms)
                utterance = vad_feed(chunk)
                if utterance:
                    logger.debug("VAD: utterance detected (%d bytes)", len(utterance.parts[0].inline_data.data))
                    await msg_queue.put(utterance)

            elif "text" in message:
                payload = json.loads(message["text"])
                if payload.get("type") == "text":
                    await msg_queue.put(types.Content(role="user", parts=[
                        types.Part(text=payload["text"])
                    ]))
                elif payload.get("type") == "image":
                    await msg_queue.put(types.Content(role="user", parts=[
                        types.Part(inline_data=types.Blob(
                            mime_type=payload.get("mimeType", "image/jpeg"),
                            data=base64.b64decode(payload["data"]),
                        ))
                    ]))

    except WebSocketDisconnect:
        logger.debug("Client disconnected: user=%s session=%s", user_id, session_id)
    except Exception as exc:
        logger.error("WebSocket error: %s", exc, exc_info=True)
    finally:
        await msg_queue.put(None)
        await processor
        logger.debug("WebSocket closed: user=%s session=%s", user_id, session_id)
