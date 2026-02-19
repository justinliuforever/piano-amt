"""Download MAESTRO v3 dataset (MIDI-only or with WAV).

Usage:
    python scripts/download_maestro.py --midi-only          # MIDI files only (~56MB)
    python scripts/download_maestro.py --split test          # test split WAVs

Downloads are placed in data/maestro/ with the original MAESTRO directory structure.
A unified metadata.csv is generated for use with `piano-amt evaluate`.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

MAESTRO_CSV_URL = (
    "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/"
    "maestro-v3.0.0.csv"
)
MAESTRO_MIDI_URL = (
    "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/"
    "maestro-v3.0.0-midi.zip"
)

DATA_DIR = Path(__file__).parent.parent / "data" / "maestro"


def download_csv() -> Path:
    """Download MAESTRO metadata CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / "maestro-v3.0.0.csv"
    if dest.exists():
        print(f"CSV already exists: {dest}")
        return dest

    print("Downloading MAESTRO metadata CSV...")
    urllib.request.urlretrieve(MAESTRO_CSV_URL, str(dest))
    print(f"Saved to {dest}")
    return dest


def download_midi() -> None:
    """Download and extract MAESTRO MIDI files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    marker = DATA_DIR / ".midi_downloaded"
    if marker.exists():
        print("MIDI files already downloaded.")
        return

    print("Downloading MAESTRO MIDI files (~56 MB)...")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        urllib.request.urlretrieve(MAESTRO_MIDI_URL, tmp.name)
        tmp_path = tmp.name

    print("Extracting...")
    with zipfile.ZipFile(tmp_path, "r") as zf:
        for member in zf.namelist():
            # Strip the top-level directory (maestro-v3.0.0/)
            parts = member.split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue
            rel_path = parts[1]
            target = DATA_DIR / rel_path
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())

    os.unlink(tmp_path)
    marker.touch()
    print("MIDI files extracted.")


def generate_metadata_csv(csv_path: Path) -> Path:
    """Generate a unified metadata.csv from the MAESTRO CSV.

    Converts MAESTRO's original CSV format to our standard format with
    relative paths from data/maestro/.
    """
    import csv as csv_mod

    output = DATA_DIR / "metadata.csv"

    rows = []
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            rows.append({
                "audio_path": row["audio_filename"],
                "midi_path": row["midi_filename"],
                "split": row["split"],
                "duration": row["duration"],
                "composer": row["canonical_composer"],
                "title": row["canonical_title"],
                "tags": "",
            })

    with open(output, "w", newline="") as f:
        writer = csv_mod.DictWriter(
            f,
            fieldnames=["audio_path", "midi_path", "split", "duration", "composer", "title", "tags"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {output} ({len(rows)} entries)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MAESTRO v3 dataset")
    parser.add_argument(
        "--midi-only", action="store_true", help="Download MIDI files only (~56MB)"
    )
    parser.add_argument(
        "--split", type=str, help="Download WAV files for a specific split (train/validation/test)"
    )
    args = parser.parse_args()

    # Always download CSV and generate metadata
    csv_path = download_csv()
    generate_metadata_csv(csv_path)

    if args.midi_only or (not args.split):
        download_midi()
        print("\nDone! MIDI files are in data/maestro/")
        if not args.split:
            print("To also download WAV files, use: --split test")
    else:
        download_midi()
        print(f"\nNote: WAV download for split '{args.split}' requires the full dataset.")
        print("Full MAESTRO dataset is ~130GB.")
        print("Download manually from: https://magenta.tensorflow.org/datasets/maestro")
        print(f"Place WAV files in: {DATA_DIR}/")


if __name__ == "__main__":
    main()
