# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Offline piano automatic music transcription (audio → MIDI) using the Onsets & Velocities deep learning model. Inference runs via ONNX Runtime (no PyTorch needed at runtime).

## Commands

```bash
# Install (editable, with dev tools)
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests excluding model-dependent ones
pytest tests/ -v -m "not slow"

# Run a single test
pytest tests/test_transcribe.py::test_name -v

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Model setup (requires export extras: torch, onnx)
pip install "torch>=2.0" "onnx>=1.14" onnxscript parse
python scripts/setup_model.py
```

## Architecture

The pipeline flows: **CLI → transcribe → ONNX model → MIDI output**.

- **`src/piano_amt/cli.py`** — Entry point (`piano-amt` command). Two subcommands: `transcribe` and `evaluate`.
- **`src/piano_amt/transcribe.py`** — Core pipeline: `_load_audio()` → `_compute_mel()` → `_run_model()` → `_track_notes()` → `save_midi()`. All audio processing constants live here (SAMPLE_RATE=16000, N_MELS=229, HOP_LENGTH=384).
- **`src/piano_amt/evaluate.py`** — Evaluation against reference MIDI using mir_eval. `evaluate_pair()` for single files, `evaluate_dataset()` for batch. Reports onset F1, onset+offset F1, velocity MAE.
- **`src/piano_amt/model/architecture.py`** — PyTorch model definition (OnsetsAndVelocities). Only needed for export, not runtime.
- **`src/piano_amt/model/export.py`** — Converts PyTorch checkpoint to ONNX format.

## Data

- Data lives in `data/` (gitignored). See `notes/data.md` for full layout and schema.
- `metadata.csv` schema: `audio_path, midi_path, split, duration, composer, title, tags` — all datasets share this format.
- MAESTRO MIDI: `python scripts/download_maestro.py --midi-only`
- Synthesize test WAVs from MIDI: `python scripts/synthesize_test_wavs.py --n 20`
- Custom data: create `data/custom/` with same `metadata.csv` schema.

## Project Notes

Ongoing progress, decisions, and TODO tracking in `notes/progress.md`.

## Code Conventions

- Python 3.10+, type hints throughout (`from __future__ import annotations`)
- Ruff for linting, line length 100
- Private functions prefixed with `_`
- `@pytest.mark.slow` for tests requiring the ONNX model file
- Model files (*.onnx, *.pt) are gitignored; stored in `models/`
