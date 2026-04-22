"""FastAPI server: bidi-streaming voice agent pipeline.

WebSocket bidi flow (no Gemini Live API required):
  1. Browser AudioWorklet streams raw 16kHz Int16 PCM continuously.
  2. Server VAD buffers chunks; silence after speech triggers an utterance.
  3. Utterance PCM is sent to gemini-2.5-flash via run_async().
  4. Streaming ADK events are forwarded to the browser as they arrive.
  5. Agent text is split at sentence boundaries and queued for CSM TTS.
  6. PCM audio chunks are forwarded to the browser as they complete.

REST endpoints support text chat via SSE for non-live agents.
"""

import asyncio
import base64
import io
import json
import logging
import re
import struct
import uuid
import warnings
import wave
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
from app.csm_tts import synthesize

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

APP_NAME = "adk-csm-live"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
session_service = InMemorySessionService()
runner = Runner(app_name=APP_NAME, agent=agent, session_service=session_service)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=,)\s+|(?<=;)\s+")

# VAD tuning
_VAD_SAMPLE_RATE = 16000
_VAD_ENERGY_THRESHOLD = 600      # Int16 RMS; below = silence
_VAD_SILENCE_MS = 800            # ms of silence that closes an utterance
_VAD_MIN_SPEECH_MS = 150         # ignore clips shorter than this
_VAD_SILENCE_BYTES = int(_VAD_SILENCE_MS * _VAD_SAMPLE_RATE / 1000) * 2
_VAD_MIN_BYTES = int(_VAD_MIN_SPEECH_MS * _VAD_SAMPLE_RATE / 1000) * 2


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


@app.get("/list-apps")
async def list_apps() -> list[dict]:
    return [{"name": APP_NAME, "live": True}]


@app.post("/apps/{app_name}/users/{user_id}/sessions")
async def create_session(app_name: str, user_id: str) -> dict[str, str]:
    session_id = str(uuid.uuid4())
    await session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
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

    async def event_stream():
        async for event in runner.run_async(
            user_id=request.userId,
            session_id=request.sessionId,
            new_message=message,
        ):
            data = event.model_dump(exclude_none=True, by_alias=True)
            yield f"data: {json.dumps(data)}\n\n"

            parts_data = (
                data.get("content", {}).get("parts", [])
                if isinstance(data.get("content"), dict) else []
            )
            text = "".join(p.get("text", "") for p in parts_data if isinstance(p, dict))
            if not data.get("partial", True) and text.strip():
                try:
                    audio_bytes = await synthesize(text.strip())
                    audio_b64 = base64.b64encode(audio_bytes).decode()
                    yield f"data: {json.dumps({'type': 'audio', 'data': audio_b64})}\n\n"
                except Exception as exc:
                    logger.error("CSM synthesis failed: %s", exc)

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
        tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
        buffer = ""

        # Pipeline: futures are created immediately as sentences arrive so
        # synthesis for sentence N+1 overlaps with sending sentence N.
        fut_queue: asyncio.Queue[asyncio.Future[bytes] | None] = asyncio.Queue()

        async def tts_producer() -> None:
            while True:
                text = await tts_queue.get()
                if text is None:
                    await fut_queue.put(None)
                    break
                fut = asyncio.ensure_future(synthesize(text))
                await fut_queue.put(fut)

        async def tts_task() -> None:
            producer = asyncio.create_task(tts_producer())
            try:
                while True:
                    item = await fut_queue.get()
                    if item is None:
                        break
                    try:
                        wav_bytes = await item
                        with wave.open(io.BytesIO(wav_bytes)) as wf:
                            sample_rate = wf.getframerate()
                            pcm = wf.readframes(wf.getnframes())
                        await websocket.send_text(json.dumps({
                            "content": {
                                "parts": [{
                                    "inlineData": {
                                        "mimeType": f"audio/pcm;rate={sample_rate}",
                                        "data": base64.b64encode(pcm).decode(),
                                    }
                                }]
                            }
                        }))
                    except Exception as exc:
                        logger.error("CSM synthesis failed: %s", exc)
            finally:
                await producer

        tts = asyncio.create_task(tts_task())

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
        finally:
            if buffer.strip():
                await tts_queue.put(buffer.strip())
            await tts_queue.put(None)
            await tts

    # ── Process loop (sequential) ──────────────────────────────────────────────

    async def process_loop() -> None:
        while True:
            content = await msg_queue.get()
            if content is None:
                break
            try:
                await handle_message(content)
            except Exception as exc:
                logger.error("handle_message error: %s", exc, exc_info=True)

    processor = asyncio.create_task(process_loop())

    # ── Receive loop ───────────────────────────────────────────────────────────

    try:
        while True:
            message = await websocket.receive()

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
