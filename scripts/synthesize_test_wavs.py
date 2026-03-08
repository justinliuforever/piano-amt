"""Synthesize WAV files from MAESTRO MIDI for testing.

Since full MAESTRO WAVs (~130GB) are impractical to download,
this script renders a subset of MIDI files to WAV using pretty_midi's
built-in FluidSynth/sine-wave synthesizer.

Usage:
    python scripts/synthesize_test_wavs.py              # default: 20 shortest test files
    python scripts/synthesize_test_wavs.py --n 10        # only 10 files
    python scripts/synthesize_test_wavs.py --split test   # specific split
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
import soundfile as sf

DATA_DIR = Path(__file__).parent.parent / "data" / "maestro"
SAMPLE_RATE = 16000


def synthesize_subset(n: int = 20, split: str = "test") -> None:
    metadata_path = DATA_DIR / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found. Run: python scripts/download_maestro.py --midi-only"
        )

    df = pd.read_csv(metadata_path)
    subset = df[df["split"] == split].copy()
    subset["duration"] = subset["duration"].astype(float)
    subset = subset.sort_values("duration").head(n)

    total_duration = 0.0
    total_size = 0
    synthesized = 0

    for _, row in subset.iterrows():
        midi_path = DATA_DIR / row["midi_path"]
        wav_path = DATA_DIR / row["audio_path"]

        if not midi_path.exists():
            print(f"  SKIP (no MIDI): {midi_path}")
            continue

        if wav_path.exists():
            print(f"  EXISTS: {wav_path.name}")
            total_size += wav_path.stat().st_size
            total_duration += row["duration"]
            synthesized += 1
            continue

        wav_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            audio = pm.synthesize(fs=SAMPLE_RATE).astype(np.float32)

            # Normalize to prevent clipping
            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio / peak * 0.9

            sf.write(str(wav_path), audio, SAMPLE_RATE)
            fsize = wav_path.stat().st_size
            total_size += fsize
            total_duration += row["duration"]
            synthesized += 1
            print(f"  OK: {wav_path.name} ({fsize / 1e6:.1f} MB, {row['duration']:.0f}s)")
        except Exception as e:
            print(f"  ERROR: {wav_path.name}: {e}")

    print(f"\nDone: {synthesized}/{len(subset)} files")
    print(f"Total duration: {total_duration / 60:.1f} min")
    print(f"Total size: {total_size / 1e6:.0f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize WAV from MAESTRO MIDI")
    parser.add_argument("--n", type=int, default=20, help="Number of files (default: 20)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (default: test)")
    args = parser.parse_args()

    print(f"Synthesizing {args.n} shortest '{args.split}' files at {SAMPLE_RATE} Hz...\n")
    synthesize_subset(n=args.n, split=args.split)


if __name__ == "__main__":
    main()
