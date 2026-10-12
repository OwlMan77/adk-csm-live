"""Build one vLLM-Omni voice profile per mood, at image build time.

Qwen3-TTS's cloning checkpoint has no `instruct` parameter — natural-language
prosody control lives in the CustomVoice and VoiceDesign checkpoints, neither of
which clones. So on this path the reference clip *is* the emotion control, and
the mood → clip map in `scripts/samples/voices.json` is the whole expression
system for the voice.

Encoding a clip is not free and does not depend on the text, so it happens once,
here, and the result is baked into the image as a `.safetensors` profile the
server loads at startup. At request time the mood picks a voice by name.

Wraps vLLM-Omni's own `precompute_custom_voice.py` rather than reimplementing
the encode, so the profile format stays whatever the pinned server expects.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PRECOMPUTE = Path("/app/precompute_custom_voice.py")
VOICE_PREFIX_DEFAULT = "hero"


def load_mapping(samples_dir: Path) -> dict[str, str]:
    """mood -> clip stem, from voices.json, or a single neutral fallback."""
    manifest = samples_dir / "voices.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        mapping = {
            mood: (entry.get("clip") if isinstance(entry, dict) else entry)
            for mood, entry in (data.get("voices") or {}).items()
        }
        mapping = {m: c for m, c in mapping.items() if c}
        if mapping:
            return mapping

    for txt in sorted(samples_dir.glob("*.txt")):
        if txt.with_suffix(".wav").exists():
            return {"neutral": txt.stem}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix", default=VOICE_PREFIX_DEFAULT)
    parser.add_argument(
        "--precompute-script",
        type=Path,
        default=PRECOMPUTE,
        help="vLLM-Omni's precompute_custom_voice.py",
    )
    args = parser.parse_args()

    mapping = load_mapping(args.samples_dir)
    if not mapping:
        print(f"No reference clips found in {args.samples_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    built: list[str] = []
    for mood, clip in sorted(mapping.items()):
        wav = args.samples_dir / f"{clip}.wav"
        txt = args.samples_dir / f"{clip}.txt"
        if not (wav.exists() and txt.exists()):
            print(f"  skip {mood}: {clip} is missing a .wav/.txt pair", file=sys.stderr)
            continue

        name = f"{args.prefix}_{mood}"
        # icl mode conditions on the transcript as well as the speaker embedding.
        # It is the higher-quality path and the reason each clip has a .txt.
        command = [
            sys.executable,
            str(args.precompute_script),
            "--model", args.model,
            "--voice-name", name,
            "--ref-audio", str(wav),
            "--ref-text", txt.read_text(encoding="utf-8").strip(),
            "--mode", "icl",
            "--output-dir", str(args.output_dir),
        ]
        print(f"  {name}  <-  {wav.name}")
        subprocess.run(command, check=True)
        built.append(name)

    if not built:
        print("No voice profiles were built.", file=sys.stderr)
        return 1

    print(f"Built {len(built)} voice profiles: {', '.join(built)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
