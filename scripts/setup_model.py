"""Download the pretrained O&V checkpoint and export to ONNX.

Usage:
    python scripts/setup_model.py              # download + export
    python scripts/setup_model.py --download    # download only
    python scripts/setup_model.py --export      # export only (needs checkpoint)
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

CHECKPOINT_URL = (
    "https://github.com/andres-fr/iamusica_demo/raw/main/"
    "iamusica_demo/assets/"
    "OnsetsAndVelocities_2023_03_04_09_53_53.289step%3D43500_f1%3D0.9675__0.9480.torch"
)
CHECKPOINT_FILENAME = "ov_checkpoint.torch"
ONNX_FILENAME = "ov_model.onnx"
MODELS_DIR = Path(__file__).parent.parent / "models"


def download_checkpoint() -> Path:
    """Download the pretrained checkpoint."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / CHECKPOINT_FILENAME
    if dest.exists():
        print(f"Checkpoint already exists: {dest}")
        return dest

    print(f"Downloading checkpoint to {dest} ...")
    urllib.request.urlretrieve(CHECKPOINT_URL, str(dest))
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"Downloaded {size_mb:.1f} MB")
    return dest


def export_onnx(checkpoint_path: Path | None = None) -> Path:
    """Export checkpoint to ONNX format."""
    if checkpoint_path is None:
        checkpoint_path = MODELS_DIR / CHECKPOINT_FILENAME

    if not checkpoint_path.exists():
        print("Checkpoint not found. Run with --download first.")
        raise SystemExit(1)

    output_path = MODELS_DIR / ONNX_FILENAME

    from piano_amt.model.export import export_to_onnx

    return export_to_onnx(checkpoint_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up O&V model (download + ONNX export)")
    parser.add_argument("--download", action="store_true", help="Download checkpoint only")
    parser.add_argument("--export", action="store_true", help="Export ONNX only")
    args = parser.parse_args()

    # If neither flag specified, do both
    do_download = args.download or not args.export
    do_export = args.export or not args.download

    if do_download:
        download_checkpoint()
    if do_export:
        export_onnx()

    print("\nModel setup complete!")
    print(f"  ONNX model: {MODELS_DIR / ONNX_FILENAME}")


if __name__ == "__main__":
    main()
