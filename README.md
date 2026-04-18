# adk-csm-live

A real-time voice agent that uses **Gemini 2.5 Flash** for speech-to-text and conversation, and **OmniVoice** (deployed on Vertex AI) for voice synthesis. Built with Google ADK and FastAPI, designed to work with [adk-chat](https://github.com/OwlMan77/ui-adk-chat) as the frontend.

## Architecture

```
Browser mic
    │  PCM 16kHz (WebSocket binary)
    ▼
FastAPI WebSocket server
    │  VAD detects end of speech
    ▼
Gemini 2.5 Flash          ← transcribes audio to text
    │  transcript
    ▼
ADK Agent (Gemini 2.5 Flash)   ← generates response
    │  text response
    ▼
OmniVoice (Vertex AI endpoint) ← synthesizes speech
    │  WAV audio (WebSocket binary)
    ▼
Browser speaker
```

- **VAD**: simple RMS energy-based silence detection — flushes audio after ~0.8s of silence
- **OmniVoice**: voice-cloned TTS using reference samples baked into the container at build time
- **ui-adk-chat**: companion frontend that connects via WebSocket when a live agent is selected

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- Docker (for building the OmniVoice container)
- Google Cloud project with Vertex AI enabled
- Hugging Face account with access to [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)

## Setup

**1. Install dependencies**

```bash
uv sync
```

**2. Configure environment**

Copy and fill in `.env`:

```bash
# Google Cloud
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=true

# Gemini model
GEMINI_MODEL=gemini-2.5-flash

# OmniVoice (set after deploying)
OMNIVOICE_ENDPOINT_ID=your-endpoint-id
OMNIVOICE_LOCATION=us-central1

# Hugging Face (for building the container)
HF_TOKEN=hf_...
```

**3. Deploy OmniVoice to Vertex AI**

```bash
# bash
./deploy_vertex.sh

# PowerShell
.\deploy_vertex.ps1
```

The deploy script:
- Checks for an existing local Docker image before building
- Creates the Artifact Registry repo if it doesn't exist
- Skips model upload if the model is already registered
- Deploys to a `g2-standard-4` instance with 1× NVIDIA L4 GPU

After deployment, add the printed `OMNIVOICE_ENDPOINT_ID` to your `.env`.

**4. Add voice reference samples**

Place audio/transcript pairs in `scripts/samples/`:

```
scripts/samples/
  audio1.wav
  audio1.txt
  audio2.wav
  audio2.txt
  ...
```

These are baked into the OmniVoice container at build time and used as the reference voice for synthesis.

## Running

```bash
uv run uvicorn app.main:app --port 8000
```

The server exposes:

| Endpoint | Description |
|---|---|
| `GET /list-apps` | Lists available agents (used by adk-chat) |
| `POST /apps/{app}/users/{user}/sessions` | Creates a session |
| `POST /run_sse` | Text chat via SSE |
| `WebSocket /ws/{user}/{session}` | Live voice session |

## Testing the TTS endpoint

```bash
uv run python scripts/test_csm.py
uv run python scripts/test_csm.py --text "What a wonderful day."
```

Output is saved to `audio/csm_output.wav`.

## Project Structure

```
app/
  main.py          — FastAPI server, WebSocket handler, VAD, transcription pipeline
  csm_tts.py       — OmniVoice Vertex AI client
  csm_agent/
    agent.py       — ADK agent definition

vertex/
  Dockerfile       — OmniVoice container (weights baked in at build time)
  server.py        — Vertex AI prediction server
  download_model.py

scripts/
  test_csm.py      — Test the OmniVoice endpoint
  samples/         — Reference voice audio + transcripts
  download_context_audio.py
  remove_music.py

deploy_vertex.sh   — Bash deploy script
deploy_vertex.ps1  — PowerShell deploy script
```

## TODOS

Make a terraform template for this project
