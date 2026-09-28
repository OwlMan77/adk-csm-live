"""Download a clip from a YouTube video and export as 24kHz mono WAV for CSM context.

Also extracts the matching transcript from YouTube's auto-generated captions so you
have the exact text that corresponds to the audio — required for CSM context segments.

Usage:
    uv run python scripts/download_context_audio.py <youtube_url> [options]

Examples:
    uv run python scripts/download_context_audio.py "https://youtu.be/..." --start 2:30 --duration 10
    uv run python scripts/download_context_audio.py "https://youtu.be/..." --start 150 --duration 15

Requirements:
    ffmpeg must be installed and on PATH.
    Windows: winget install ffmpeg

Outputs (alongside --output path):
    context_audio.wav      — 24kHz mono WAV for CSM context audio
    context_audio.txt      — exact transcript lines for the chosen time window
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not found on PATH.")
        print("  Windows: winget install ffmpeg  or  choco install ffmpeg")
        sys.exit(1)


def parse_timestamp(ts: str) -> float:
    """Convert M:SS, HH:MM:SS, or raw seconds to float seconds."""
    parts = ts.split(":")
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Unrecognised timestamp: {ts!r}")


def vtt_time_to_seconds(vtt_ts: str) -> float:
    """Convert VTT timestamp (HH:MM:SS.mmm or MM:SS.mmm) to seconds."""
    vtt_ts = vtt_ts.strip()
    parts = vtt_ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(vtt_ts)


def parse_vtt(vtt_path: Path, start: float, end: float) -> str:
    """Extract transcript lines whose cue window overlaps [start, end]."""
    text = vtt_path.read_text(encoding="utf-8")

    # Each VTT cue: timestamp line followed by text lines.
    cue_pattern = re.compile(
        r"(\d[\d:\.]+)\s+-->\s+(\d[\d:\.]+)[^\n]*\n((?:(?!-->)[^\n]+\n?)+)",
        re.MULTILINE,
    )

    lines: list[str] = []
    seen: set[str] = set()

    for match in cue_pattern.finditer(text):
        cue_start = vtt_time_to_seconds(match.group(1))
        cue_end = vtt_time_to_seconds(match.group(2))

        # Include cues that overlap the requested window.
        if cue_end < start or cue_start > end:
            continue

        # Strip VTT tags (<c>, <b>, timestamps embedded in lines).
        raw = re.sub(r"<[^>]+>", "", match.group(3)).strip()
        if raw and raw not in seen:
            lines.append(raw)
            seen.add(raw)

    return " ".join(lines)


def download_clip(
    url: str,
    start: float | None,
    duration: float,
    output: Path,
) -> None:
    end = (start or 0) + duration
    transcript_path = output.with_suffix(".txt")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_audio = tmp_path / "raw.%(ext)s"

        # --- Audio ---
        print("Downloading audio…")
        subprocess.run(
            [
                "yt-dlp",
                "--extract-audio",
                "--audio-format", "wav",
                "--audio-quality", "0",
                "--output", str(raw_audio),
                "--no-playlist",
                url,
            ],
            check=True,
        )

        src_files = list(tmp_path.glob("raw.*"))
        if not src_files:
            print("ERROR: yt-dlp produced no audio file.")
            sys.exit(1)
        src = src_files[0]

        # --- Subtitles ---
        print("Downloading captions…")
        sub_result = subprocess.run(
            [
                "yt-dlp",
                "--write-auto-subs",
                "--sub-lang", "en",
                "--sub-format", "vtt",
                "--skip-download",
                "--output", str(tmp_path / "subs.%(ext)s"),
                "--no-playlist",
                url,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        vtt_files = list(tmp_path.glob("*.vtt"))
        transcript = ""
        if vtt_files:
            transcript = parse_vtt(vtt_files[0], start or 0, end)
            transcript_path.write_text(transcript, encoding="utf-8")
            print(f"Transcript saved to: {transcript_path}")
            print(f"  \"{transcript}\"")
        else:
            print("WARNING: No captions found. Transcript will be empty.")
            print(f"  yt-dlp stderr: {sub_result.stderr.strip()}")

        # --- Clip + convert ---
        print(f"\nConverting to 24kHz mono WAV → {output}")
        ff_cmd = ["ffmpeg", "-y", "-i", str(src)]
        if start is not None:
            ff_cmd += ["-ss", str(start)]
        ff_cmd += [
            "-t", str(duration),
            "-ar", "24000",
            "-ac", "1",
            "-sample_fmt", "s16",
            str(output),
        ]
        subprocess.run(ff_cmd, check=True)

    print(f"\nDone.")
    print(f"  Audio:      {output}")
    print(f"  Transcript: {transcript_path}")
    print(f"  Duration:   {duration}s | 24kHz mono")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a YouTube clip as 24kHz mono WAV + exact transcript for CSM context."
    )
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument(
        "--start",
        default=None,
        help="Start timestamp (e.g. 2:30 or 150.5). Defaults to beginning.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Clip length in seconds (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("audio/context_audio.wav"),
        help="Output WAV path (default: audio/context_audio.wav).",
    )
    args = parser.parse_args()

    check_ffmpeg()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    download_clip(
        url=args.url,
        start=parse_timestamp(args.start) if args.start else None,
        duration=args.duration,
        output=args.output,
    )


if __name__ == "__main__":
    main()
