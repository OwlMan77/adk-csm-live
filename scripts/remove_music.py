"""Remove background music from an audio clip, keeping only the vocals/speech.

Uses the Demucs htdemucs model for separation but bypasses torchaudio I/O
entirely (loading and saving via soundfile instead), avoiding the torchcodec
DLL requirement on Windows.

Usage:
    uv run python scripts/remove_music.py <input_wav> [options]

Examples:
    uv run python scripts/remove_music.py scripts/context_audio.wav
    uv run python scripts/remove_music.py scripts/context_audio.wav --output scripts/vocals.wav

Output is 24kHz mono WAV, ready for CSM context audio.
On first run, Demucs downloads htdemucs model weights (~80MB).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not found on PATH.")
        print("  Windows: winget install ffmpeg")
        sys.exit(1)


def separate_vocals(input_wav: Path, output_wav: Path) -> None:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    print("Loading htdemucs model (downloads ~80MB on first run)…")
    model = get_model("htdemucs")
    model.eval()

    target_sr = model.samplerate          # 44100
    target_ch = model.audio_channels      # 2

    # Load with soundfile — no torchaudio/torchcodec involved.
    print(f"Loading: {input_wav}")
    audio_np, file_sr = sf.read(str(input_wav), dtype="float32", always_2d=True)
    # soundfile gives (samples, channels) — convert to (channels, samples)
    audio_np = audio_np.T

    # Resample to model rate if needed (input is 24kHz from download script).
    if file_sr != target_sr:
        import subprocess as sp
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        # Write current audio at original rate, then resample via ffmpeg.
        sf.write(str(tmp_path), audio_np.T, file_sr, subtype="PCM_16")
        resampled = tmp_path.with_suffix(".rs.wav")
        sp.run(
            ["ffmpeg", "-y", "-i", str(tmp_path),
             "-ar", str(target_sr), "-ac", str(target_ch), str(resampled)],
            check=True, capture_output=True,
        )
        audio_np, _ = sf.read(str(resampled), dtype="float32", always_2d=True)
        audio_np = audio_np.T
        tmp_path.unlink(missing_ok=True)
        resampled.unlink(missing_ok=True)

    # Ensure stereo.
    if audio_np.shape[0] == 1:
        audio_np = np.repeat(audio_np, target_ch, axis=0)

    wav_tensor = torch.from_numpy(audio_np).unsqueeze(0)  # (1, channels, samples)

    print("Separating vocals (CPU inference — may take a minute)…")
    with torch.no_grad():
        sources = apply_model(model, wav_tensor, device="cpu", progress=True)[0]

    vocals_idx = model.sources.index("vocals")
    vocals_np = sources[vocals_idx].numpy().T  # (samples, channels)

    # Save to a temp WAV at model rate, then resample to 24kHz mono via ffmpeg.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    sf.write(str(tmp_path), vocals_np, target_sr, subtype="PCM_16")

    print(f"Resampling to 24kHz mono → {output_wav}")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp_path),
         "-ar", "24000", "-ac", "1", "-sample_fmt", "s16", str(output_wav)],
        check=True,
    )
    tmp_path.unlink(missing_ok=True)

    print(f"\nDone. Vocals saved to: {output_wav}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove background music from a WAV clip using Demucs."
    )
    parser.add_argument("input", type=Path, help="Input WAV file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output WAV path. Defaults to <input>_vocals.wav.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    check_ffmpeg()

    output = args.output or Path("audio") / (args.input.stem + "_vocals.wav")
    output.parent.mkdir(parents=True, exist_ok=True)

    separate_vocals(args.input, output)


if __name__ == "__main__":
    main()
