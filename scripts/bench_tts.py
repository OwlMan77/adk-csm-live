"""Decide the engine swap with a measurement, not a preference.

Same sentences, same reference voice, both engines. Records time to first audio
byte and total wall time, and writes the WAVs side by side so the clone fidelity
can be judged by ear. Two numbers and two files, because the two halves of this
decision are not the same question:

  * **First byte** is the one the user feels. OmniVoice's attention is fully
    bidirectional, so it cannot emit anything until the whole utterance is
    decoded; Qwen3-TTS streams. That gap should be large and obvious.
  * **Does it sound like the reference** cannot be measured from here, so this
    only puts the files next to each other and gets out of the way.

If Qwen3-TTS wins on latency and loses on the voice, that is a real finding and
the cascade stays on OmniVoice with its ceilings — decided by ear, on the record.

Usage:
    uv run python scripts/bench_tts.py                 # whatever TTS_ENGINE says
    uv run python scripts/bench_tts.py --both          # A/B, the point of this
    uv run python scripts/bench_tts.py --engine qwen3-tts --runs 5
    uv run python scripts/bench_tts.py --text "One sentence to try."
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
import wave
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.cues import extract  # noqa: E402
from app.tts.base import MOODS  # noqa: E402

OUTPUT_DIR = ROOT / "audio" / "bench"

# Deliberately varied: a plain statement, a question, something with a cue in it,
# and something long enough that a streaming engine's advantage has room to show.
SENTENCES = [
    "Hello, it's a pleasure to meet you.",
    "[question-en] Are you quite sure that's what he told you?",
    "[sigh] I suppose we'll have to start the whole thing again.",
    "The train leaves at a quarter past four, which gives us just enough time to "
    "collect the tickets, find the right platform, and still buy a coffee before "
    "we board.",
]


class Timing:
    __slots__ = ("engine", "sentence", "mood", "first_byte_ms", "total_ms", "bytes", "words", "rate")

    def __init__(self, engine: str, sentence: str, mood: str) -> None:
        self.engine = engine
        self.sentence = sentence
        self.mood = mood
        self.first_byte_ms = float("nan")
        self.total_ms = float("nan")
        self.bytes = 0
        self.words = 0
        self.rate = 0


def build_engine(name: str):
    """Build one engine by name, bypassing the process-wide singleton so --both
    can hold two at once."""
    if name in ("qwen3", "qwen3-tts", "qwen"):
        from app.tts.qwen3 import Qwen3TTSEngine

        return Qwen3TTSEngine()
    from app.tts.omnivoice import OmniVoiceEngine

    return OmniVoiceEngine()


def save_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate or 24000)
        wf.writeframes(pcm)


async def run_one(engine, text: str, index: int, out: Path | None) -> Timing:
    utterance = extract(text, index=index)
    timing = Timing(engine.name, utterance.text, utterance.mood)

    pcm = bytearray()
    started = time.perf_counter()

    async for chunk in engine.synthesize(utterance):
        if chunk.pcm and timing.bytes == 0:
            timing.first_byte_ms = (time.perf_counter() - started) * 1000
        pcm.extend(chunk.pcm)
        timing.bytes += len(chunk.pcm)
        timing.words += len(chunk.words)
        if chunk.sample_rate:
            timing.rate = chunk.sample_rate

    timing.total_ms = (time.perf_counter() - started) * 1000

    if out is not None and pcm:
        save_wav(out, bytes(pcm), timing.rate)

    return timing


async def bench(engine_name: str, sentences: list[str], runs: int, save: bool) -> list[Timing]:
    engine = build_engine(engine_name)
    results: list[Timing] = []
    try:
        for run in range(runs):
            for i, text in enumerate(sentences):
                # Only the first run writes files; later runs exist to measure.
                out = OUTPUT_DIR / f"{engine.name}_{i:02d}.wav" if (save and run == 0) else None
                try:
                    timing = await run_one(engine, text, i, out)
                except Exception as exc:  # noqa: BLE001 — a failed engine is a result
                    print(f"  ! {engine.name} failed on sentence {i}: {exc}", file=sys.stderr)
                    continue
                results.append(timing)
                seconds = timing.bytes / 2 / (timing.rate or 24000)
                print(
                    f"  [{engine.name}] #{i} {timing.first_byte_ms:7.0f} ms to first byte"
                    f" | {timing.total_ms:7.0f} ms total"
                    f" | {seconds:5.2f} s audio"
                    f" | {timing.rate or '?'} Hz"
                    f" | mood={timing.mood}"
                    f" | {timing.words} word timings"
                )
    finally:
        await engine.aclose()
    return results


def summarise(results: list[Timing]) -> None:
    if not results:
        print("\nNothing to summarise — every request failed.")
        return

    by_engine: dict[str, list[Timing]] = {}
    for t in results:
        by_engine.setdefault(t.engine, []).append(t)

    print("\n" + "=" * 74)
    print(f"{'engine':<14}{'first byte':>16}{'total':>14}{'realtime':>12}{'timings':>10}")
    print("-" * 74)

    for name, rows in by_engine.items():
        firsts = [r.first_byte_ms for r in rows if r.first_byte_ms == r.first_byte_ms]
        totals = [r.total_ms for r in rows]
        audio_s = sum(r.bytes / 2 / (r.rate or 24000) for r in rows)
        wall_s = sum(totals) / 1000
        rtf = wall_s / audio_s if audio_s else float("nan")
        timings = sum(r.words for r in rows)
        median_first = statistics.median(firsts) if firsts else float("nan")
        print(
            f"{name:<14}{median_first:>13.0f} ms{statistics.median(totals):>11.0f} ms"
            f"{rtf:>12.2f}x{timings:>10}"
        )

    print("=" * 74)
    print("first byte = median time until any audio could start playing")
    print("realtime   = wall time / audio produced; below 1.0 means faster than playback")
    print("timings    = word timings returned; 0 means the face falls back to analysis")

    if len(by_engine) > 1:
        names = list(by_engine)
        firsts = {
            n: statistics.median(
                [r.first_byte_ms for r in by_engine[n] if r.first_byte_ms == r.first_byte_ms] or [float("nan")]
            )
            for n in names
        }
        best = min(names, key=lambda n: firsts[n])
        other = [n for n in names if n != best]
        if other and firsts[other[0]] == firsts[other[0]]:
            saved = firsts[other[0]] - firsts[best]
            print(f"\n{best} reaches audio {saved:.0f} ms sooner than {other[0]}.")
        print(f"\nNow listen to both before deciding: {OUTPUT_DIR}")
        print("The number above is half the question. The other half is whether it")
        print("still sounds like the reference voice.")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", help="omnivoice | qwen3-tts. Defaults to $TTS_ENGINE.")
    parser.add_argument("--both", action="store_true", help="Run both engines and compare.")
    parser.add_argument("--text", action="append", help="Sentence to synthesise; repeatable.")
    parser.add_argument("--runs", type=int, default=1, help="Repeat the set N times.")
    parser.add_argument("--no-save", action="store_true", help="Measure only, write no WAVs.")
    parser.add_argument(
        "--mood",
        choices=MOODS,
        help="Force one mood instead of taking it from the text's cues.",
    )
    args = parser.parse_args()

    sentences = args.text or SENTENCES
    if args.mood:
        sentences = [f"[mood:{args.mood}] {s}" for s in sentences]

    engines = ["omnivoice", "qwen3-tts"] if args.both else [args.engine or os.environ.get("TTS_ENGINE", "omnivoice")]

    results: list[Timing] = []
    for name in engines:
        print(f"\n▶ {name}")
        results.extend(await bench(name, sentences, args.runs, not args.no_save))

    summarise(results)
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
