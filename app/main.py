"""FastAPI server: real-time voice agent pipeline.

WebSocket flow:
  1. Browser streams raw 16kHz PCM from the microphone.
  2. RMS-based VAD detects end of speech and flushes the buffer.
  3. Gemini Flash transcribes the audio chunk to text.
  4. ADK agent (Gemini Flash) generates a text response.
  5. OmniVoice (Vertex AI) synthesises the response to WAV.
  6. WAV bytes are sent back to the browser over the WebSocket.

REST endpoints support text chat via SSE for non-live agents.
"""

import asyncio
import base64
import json
import logging
import uuid
import warnings
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

from app.csm_agent import agent
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

            # Synthesize TTS on completed agent turns.
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
runner = Runner(app_name=APP_NAME, agent=agent, session_service=session_service)


import io
import struct
import wave as wave_module

import google.genai as genai

_genai_client = None

def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        import os
        _genai_client = genai.Client(
            vertexai=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    return _genai_client


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave_module.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


async def _transcribe(pcm_bytes: bytes) -> str:
    wav_bytes = _pcm_to_wav(pcm_bytes)
    client = _get_genai_client()
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part(
                    inline_data=types.Blob(mime_type="audio/wav", data=wav_bytes)
                ),
                types.Part(text="Transcribe the speech in this audio. Reply with only the transcript, nothing else."),
            ],
        ),
    )
    return response.text.strip()


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

    SAMPLE_RATE = 16000
    BYTES_PER_SAMPLE = 2
    SILENCE_THRESHOLD = 300        # RMS below this = silence
    SILENCE_DURATION = 0.8         # seconds of silence before flush
    MIN_SPEECH_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * 0.3  # min 0.3s of speech

    SILENCE_SAMPLES = int(SAMPLE_RATE * SILENCE_DURATION)
    silent_samples = 0
    has_speech = False
    pcm_buffer = bytearray()

    def rms(chunk: bytes) -> float:
        samples = struct.unpack_from(f"<{len(chunk)//2}h", chunk)
        return (sum(s * s for s in samples) / len(samples)) ** 0.5

    async def handle_audio(pcm: bytes) -> None:
        transcript = await _transcribe(pcm)
        if not transcript:
            return
        logger.debug("Transcript: %r", transcript[:80])
        await websocket.send_text(json.dumps({"type": "transcript", "text": transcript}))
        await run_agent(transcript)

    async def run_agent(text: str) -> None:
        message = types.Content(role="user", parts=[types.Part(text=text)])
        text_buffer = ""
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            event_dict = event.model_dump(exclude_none=True, by_alias=True)
            await websocket.send_text(json.dumps(event_dict))

            parts = (
                event_dict.get("content", {}).get("parts", [])
                if isinstance(event_dict.get("content"), dict) else []
            )
            for part in parts:
                if isinstance(part.get("text"), str):
                    text_buffer += part["text"]

        if text_buffer.strip():
            logger.debug("Synthesizing: %r", text_buffer[:80])
            try:
                audio_bytes = await synthesize(text_buffer.strip())
                await websocket.send_bytes(audio_bytes)
            except Exception as exc:
                logger.error("CSM synthesis failed: %s", exc)

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                chunk = message["bytes"]
                pcm_buffer.extend(chunk)

                energy = rms(chunk)
                num_samples = len(chunk) // BYTES_PER_SAMPLE

                if energy >= SILENCE_THRESHOLD:
                    has_speech = True
                    silent_samples = 0
                else:
                    if has_speech:
                        silent_samples += num_samples
                        if silent_samples >= SILENCE_SAMPLES and len(pcm_buffer) >= MIN_SPEECH_BYTES:
                            pcm_chunk = bytes(pcm_buffer)
                            pcm_buffer.clear()
                            silent_samples = 0
                            has_speech = False
                            asyncio.create_task(handle_audio(pcm_chunk))

            elif "text" in message:
                payload = json.loads(message["text"])
                if payload.get("type") == "text":
                    await run_agent(payload["text"])
                elif payload.get("type") == "flush":
                    if pcm_buffer:
                        pcm_chunk = bytes(pcm_buffer)
                        pcm_buffer.clear()
                        asyncio.create_task(handle_audio(pcm_chunk))
                elif payload.get("type") == "image":
                    image_bytes = base64.b64decode(payload["data"])
                    await run_agent(f"[Image received: {payload.get('mimeType', 'image/jpeg')}]")

    except WebSocketDisconnect:
        logger.debug("Client disconnected")
    except Exception as exc:
        logger.error("Streaming error: %s", exc, exc_info=True)
