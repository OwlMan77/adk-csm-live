"""
Qwen3-TTS on Modal — the scale-to-zero deployment, as an alternative to Cloud Run.

## Why this exists alongside deploy_qwen_cloudrun.sh

Both targets scale to zero and both cost nothing idle, so the hourly rate is
almost irrelevant: Cloud Run's L4 is ~$0.67/hr against Modal's ~$0.80/hr, and at
the handful of GPU-hours a month this workload actually burns, that difference is
under a dollar.

What is *not* irrelevant is Modal's $30/month of included credits, which at the
L4 rate is ~37 GPU-hours. This workload only consumes GPU time while a sentence
is being synthesised — not for the length of a call, and never while idle — so
personal use realistically lands inside the free tier and bills zero.

The other reason is setup cost. The GCP project this repo was originally built
against is no longer reachable from the current account, so the Cloud Run path
now means a fresh project, a billing link, *and* an L4 quota grant before the
first byte is served. Modal needs an account and `modal deploy`.

Neither file replaces the other. The container is identical — this runs the very
same `qwen/Dockerfile`, so whatever is true of the image on Cloud Run is true of
it here, and switching back is a deploy command.

## Deploying

    pip install modal && modal setup
    modal deploy qwen/modal_app.py        # from the repository root

The build runs the reference-clip precompute, which needs a GPU, so the *build*
is billed too (a few minutes of L4). It only happens when the image changes.

Then point the orchestrator at the printed URL:

    TTS_ENGINE=qwen3-tts
    QWEN_TTS_URL=https://<workspace>--qwen3-tts-server.modal.run

## A note on who can call it

This endpoint is left authenticated on purpose — `@app.server` defaults to
`unauthenticated=False`. An open GPU endpoint is a standing invitation to burn
someone else's credits, and the blast radius of a public TTS box is real money
rather than mere noise. Modal issues token pairs (`modal token new`); anonymous
callers get a 401 from Modal's proxy before a container is ever involved.

So the orchestrator has to carry them, or nothing works:

    QWEN_TTS_KEY=wk-...
    QWEN_TTS_SECRET=ws-...

`app/tts/qwen3.py` turns those into `Modal-Key` / `Modal-Secret` headers on both
the warm ping and the WebSocket handshake. Worth knowing the failure mode if they
are missing: a 401 is indistinguishable from a cold start from the outside, so
the face sits in its waking pose waiting for a voice that is never coming. The
warm ping logs that case explicitly rather than letting it look like slowness.

If you would rather have it open, add `unauthenticated=True` to the decorator
below. That is a deliberate choice, not a default worth inheriting silently.

## Still to verify on the first deploy

Modal documents WebSocket support for `@modal.web_server`, `@modal.asgi_app` and
`@modal.wsgi_app`. The Servers guide does not mention WebSockets either way, and
this whole path rides on one (`/v1/audio/speech/stream` is the only transport that
streams audio *and* returns word timings). A Server is described as "most directly
analogous to" a `@modal.web_server` Function, so it very probably works — but
check it before assuming, because the fallback is a real cost: no timings means no
anticipation in the face, and the REST path cannot stream and align at once.

    uv run python scripts/bench_tts.py --engine qwen3-tts --text "Testing."

A 401 means the credentials above are missing. A handshake failure on the
WebSocket, with the warm ping succeeding, means Servers do not carry WebSockets
and this needs `@modal.web_server` on a Function instead.
"""

import subprocess

import modal

MINUTES = 60

# Must match the Dockerfile's ENV PORT, which is what entrypoint.sh binds.
PORT = 8080

# `context_dir` is the repository root because the Dockerfile copies
# `scripts/samples` and `qwen/precompute_voices.py` from there.
#
# `gpu` is load-bearing and easy to miss: the image's precompute step runs the
# speaker encoder and the codec over each reference clip at *build* time, so the
# builder itself needs a GPU. Without this the build fails inside
# precompute_voices.py rather than at deploy, which is a confusing place to
# discover it.
#
# `.entrypoint([])` clears the image's ENTRYPOINT. Modal supplies its own, and a
# lingering ENTRYPOINT shadows it -- the container would start vLLM twice, once
# without Modal's supervision.
qwen_image = modal.Image.from_dockerfile(
    "qwen/Dockerfile",
    context_dir=".",
    gpu="L4",
    build_args={"HF_TOKEN": ""},
).entrypoint([])

app = modal.App("qwen3-tts")


@app.server(
    image=qwen_image,
    gpu="L4",
    port=PORT,
    # How long a warm container survives after the last request. Long enough
    # that the pauses inside a conversation never pay a cold start, short
    # enough that hanging up stops the meter within a couple of minutes.
    scaledown_window=2 * MINUTES,
    # vLLM loading Qwen3-TTS plus the forced aligner is minutes, not seconds,
    # even with the weights baked into the image. The default startup timeout
    # is far too tight for this and fails the first request of a cold start.
    startup_timeout=10 * MINUTES,
    # Sentences are pipelined -- N+1 synthesises while N is still sending -- so
    # a replica must take several at once or the pipeline serialises and the
    # latency win is given straight back.
    target_concurrency=4,
)
class Server:
    @modal.enter()
    def start(self):
        # entrypoint.sh is reused verbatim rather than re-spelling the vllm
        # command here: it is the one place that knows to copy the packaged
        # deploy config, append `custom_voice_dir`, and pass --forced-aligner,
        # which is what makes word_timestamps available on the WebSocket.
        # Duplicating that here is how the two deployment paths would drift.
        self.process = subprocess.Popen(["/app/entrypoint.sh"])

    @modal.exit()
    def stop(self):
        self.process.terminate()
