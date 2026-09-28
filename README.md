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
- A Gemini API key from [AI Studio](https://aistudio.google.com) — that is the
  whole requirement for text chat and transcription. No Google Cloud project, no
  billing account, no quota grant.
- For voice output, somewhere to run the TTS GPU: a Modal account, or a Google
  Cloud project with an L4 quota grant. See **Where the GPU runs** below.
- Docker, only if you are building a serving container yourself

## Setup

**1. Install dependencies**

```bash
uv sync
```

**2. Configure environment**

```bash
cp .env.example .env
```

For text chat and transcription the only value that matters is `GEMINI_API_KEY`,
from [AI Studio](https://aistudio.google.com). The default
`GOOGLE_GENAI_USE_VERTEXAI=false` reaches Gemini with that key alone: no project,
no billing account, no quota grant.

Voice output additionally needs a TTS server, which is the next step.

**3. Deploy a speech engine** (only needed for voice output)

```bash
modal deploy qwen/modal_app.py     # Modal
./deploy_qwen_cloudrun.sh          # Cloud Run + L4
```

Paste the printed URL into `QWEN_TTS_URL`. Without one, everything except
synthesis still runs, and the failure is confined to the first synthesis request.

To use OmniVoice instead, run `deploy_vertex.sh` (or the PowerShell twin), then
set `OMNIVOICE_ENDPOINT_ID`, `GOOGLE_CLOUD_PROJECT` and `TTS_ENGINE=omnivoice`.
Note its weights are CC-BY-NC, so that deployment is non-commercial.

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
| `GET /warm` | Wakes the speech backend from a cold start |

## Speech engines

`TTS_ENGINE` picks the voice:

| | `omnivoice` (default) | `qwen3-tts` |
|---|---|---|
| Model | k2-fsa/OmniVoice on Vertex AI | Qwen3-TTS-12Hz-1.7B-Base on vLLM-Omni |
| Weights | **CC-BY-NC** (only the code is Apache-2.0) | Apache-2.0 |
| Cloning | yes, from a reference clip | yes, from 3 s |
| Streaming | no — bidirectional attention makes the first byte wait for the last | yes |
| Word timings | none | yes, via `Qwen3-ForcedAligner-0.6B` |
| Emphasis | 13 fixed tags, closed `instruct` vocabulary | a reference clip per mood; the `-Base` checkpoint has no `instruct` |

The two are not swapped on preference. Run them side by side, read the number
*and* listen to the files:

```bash
uv run python scripts/bench_tts.py --both
```

It reports time to first audio byte, total wall time and realtime factor, and
writes both engines' WAVs to `audio/bench/` for an A/B listen. Latency is half
the question; whether it still sounds like the reference voice is the other half,
and that one is decided by ear.

## Where the GPU runs

Qwen3-TTS needs an L4-class GPU. There are two deploy targets and the same image
serves both — `qwen/Dockerfile` is the single source of truth, so a switch is a
deploy command, not a rebuild.

| | Cloud Run + L4 | Modal |
|---|---|---|
| Deploy | `./deploy_qwen_cloudrun.sh` | `modal deploy qwen/modal_app.py` |
| Rate | ~$0.67/hr | ~$0.80/hr |
| Idle | $0 — scales to zero | $0 — scales to zero |
| Included | — | **$30/month ≈ 37 L4-hours** |
| First run needs | a project, billing, **and an L4 quota grant** | an account |

The hourly rates are close enough to ignore. The GPU only runs while a sentence
is being synthesised — not for the length of a call, never while idle — so a few
GPU-hours a month is the realistic figure, which lands inside Modal's included
credits and bills nothing. Cloud Run's per-hour edge is worth pennies against
that, and it is the target that needs a quota grant before it will serve a byte.

Nothing here is locked in: keep whichever is deployed and the other stays a
command away.

## Cues

The agent marks non-verbal moments inline — `[laughter]`, `[sigh]`,
`[question-en]` — and opens each reply with one `[mood:*]` tag. `app/cues.py`
extracts both once, and that single result feeds three places that must never
disagree:

* the **engine**, where the mood selects which reference clip the sentence is
  cloned from (`scripts/samples/voices.json` holds that map — correct it by ear);
* the **client**, where the tags, the mood and any word timings ride alongside
  the audio frame, so the face reacts to the sentence the ear is hearing;
* the **transcript**, with the tags stripped, so `[sigh]` stops being rendered as
  literal text.

Because the tags are removed before synthesis they are never spoken, so the
vocabulary no longer depends on which engine is loaded.

## Project Structure

```
app/
  main.py          — FastAPI server, WebSocket handler, VAD, synthesis pipeline
  cues.py          — [tag] and [mood:*] extraction; one label, three consumers
  tts/
    base.py        — the streaming AudioChunk contract every engine speaks
    omnivoice.py   — k2-fsa/OmniVoice on Vertex AI
    qwen3.py       — Qwen3-TTS on vLLM-Omni (streaming + word timings)
    voices.py      — mood -> reference clip
  csm_agent/
    agent.py       — ADK agent definition

vertex/            — OmniVoice serving container (weights baked in at build time)
  Dockerfile
  server.py
  download_model.py

qwen/              — Qwen3-TTS serving container
  Dockerfile       — weights, aligner and voice profiles all baked in
  entrypoint.sh
  precompute_voices.py
  modal_app.py     — the same image, deployed to Modal instead of Cloud Run

scripts/
  bench_tts.py     — first-byte latency + A/B listening harness
  samples/         — reference voice audio, transcripts, and voices.json
  download_context_audio.py
  remove_music.py

tests/             — cue extraction, wire frames, vLLM-Omni protocol handling

deploy_cloudrun.sh       — orchestrator (CPU, scales to zero)
deploy_qwen_cloudrun.sh  — Qwen3-TTS on Cloud Run + L4 (scales to zero)
deploy_vertex.sh         — OmniVoice on Vertex AI
deploy_vertex.ps1        — the same, in PowerShell
```

## TODOS

Make a terraform template for this project
