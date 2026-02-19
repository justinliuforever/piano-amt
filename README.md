# Piano AMT

Automatic piano transcription: audio → MIDI using the [Onsets & Velocities](https://github.com/andres-fr/iamusica_training) model.

## Model

Uses the pretrained **Onsets & Velocities** checkpoint (O&V, 2023) exported to ONNX. The model takes log-Mel spectrograms (229 bands, 16kHz) and outputs per-frame onset probabilities + velocity estimates for 88 piano keys.

## Quick Start

```bash
pip install -e ".[dev]"

# Download checkpoint + export ONNX (~50MB)
pip install "torch>=2.0" "onnx>=1.14" onnxscript parse
python scripts/setup_model.py

# Transcribe
piano-amt transcribe input.wav -o output.mid
```

## Evaluate

```bash
# Evaluate on bundled test data (2 Chopin preludes)
piano-amt evaluate data/smd/

# Evaluate on MAESTRO test set
piano-amt evaluate data/maestro/
```

## Data

```bash
# Download MAESTRO v3 MIDI files (~56MB)
python scripts/download_maestro.py --midi-only
```

WAV files must be downloaded separately from [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) (~130GB full dataset). Place them in `data/maestro/` matching the paths in `metadata.csv`.
