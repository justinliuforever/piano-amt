"""Behavior tests for the transcription pipeline.

All tests require the ONNX model (marked slow).
Run with: pytest tests/ -v
Skip model tests: pytest tests/ -v -m "not slow"
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data" / "smd"


@pytest.fixture
def model_path():
    p = MODELS_DIR / "ov_model.onnx"
    if not p.exists():
        pytest.skip("ONNX model not found. Run: python scripts/setup_model.py")
    return p


@pytest.fixture
def smd_audio():
    p = DATA_DIR / "chopin_op28_1.wav"
    if not p.exists():
        pytest.skip("SMD test data not found")
    return p


@pytest.fixture
def smd_midi():
    p = DATA_DIR / "chopin_op28_1.mid"
    if not p.exists():
        pytest.skip("SMD test data not found")
    return p


@pytest.mark.slow
def test_transcribe_produces_notes(model_path, smd_audio):
    """Transcribing real piano audio should produce notes."""
    from piano_amt.transcribe import transcribe

    notes = transcribe(smd_audio, model_path)
    assert len(notes) > 0
    for note in notes:
        assert 21 <= note.pitch <= 108
        assert note.end > note.start
        assert 1 <= note.velocity <= 127


@pytest.mark.slow
def test_transcribe_f1_above_threshold(model_path, smd_audio, smd_midi):
    """Onset F1 on SMD test data should be reasonable (> 50%)."""
    from piano_amt.evaluate import evaluate_pair

    metrics = evaluate_pair(smd_audio, smd_midi, model_path)
    assert metrics["onset_f1"] > 0.5, f"onset_f1={metrics['onset_f1']:.3f} too low"


@pytest.mark.slow
def test_save_midi_roundtrip(model_path, smd_audio):
    """save_midi writes a file that pretty_midi can read back correctly."""
    import pretty_midi

    from piano_amt.transcribe import save_midi, transcribe

    notes = transcribe(smd_audio, model_path)
    assert len(notes) > 0

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        tmp_path = f.name

    save_midi(notes, tmp_path)

    # Read back
    midi = pretty_midi.PrettyMIDI(tmp_path)
    read_notes = midi.instruments[0].notes
    assert len(read_notes) == len(notes)

    Path(tmp_path).unlink()


def test_transcribe_empty_audio(model_path):
    """Empty/silent audio should return empty list without crashing."""
    import soundfile as sf

    from piano_amt.transcribe import transcribe

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    # Write 1 second of silence
    sf.write(tmp_path, np.zeros(16000, dtype=np.float32), 16000)
    notes = transcribe(tmp_path, model_path)
    assert isinstance(notes, list)

    Path(tmp_path).unlink()


@pytest.mark.slow
def test_evaluate_pair_returns_metrics(model_path, smd_audio, smd_midi):
    """evaluate_pair should return a dict with all expected metric keys."""
    from piano_amt.evaluate import evaluate_pair

    metrics = evaluate_pair(smd_audio, smd_midi, model_path)
    expected_keys = {
        "onset_precision", "onset_recall", "onset_f1",
        "onset_offset_precision", "onset_offset_recall", "onset_offset_f1",
        "n_ref", "n_est",
    }
    assert expected_keys.issubset(metrics.keys())
    for key in expected_keys:
        assert isinstance(metrics[key], (int, float))
