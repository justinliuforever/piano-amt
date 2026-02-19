"""Core transcription pipeline: audio file → Note list → MIDI file.

Merges feature extraction, ONNX inference, note tracking, and MIDI writing
into a single module with a clean public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PIANO_MIN_MIDI = 21  # A0

# Mel spectrogram defaults (match O&V model input spec)
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 384
N_MELS = 229
FMIN = 50.0
FMAX = 8000.0
TOP_DB = 80.0


@dataclass
class Note:
    """A single detected piano note."""

    pitch: int  # MIDI note number (21-108)
    start: float  # seconds
    end: float  # seconds
    velocity: int  # 1-127


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def _load_audio(audio_path: str | Path) -> NDArray[np.float32]:
    """Load audio file, convert to mono float32 at 16 kHz."""
    import librosa
    import soundfile as sf

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


# ---------------------------------------------------------------------------
# Mel spectrogram
# ---------------------------------------------------------------------------

_mel_basis_cache: NDArray[np.float32] | None = None


def _get_mel_basis() -> NDArray[np.float32]:
    """Lazily create and cache the Mel filter bank."""
    global _mel_basis_cache
    if _mel_basis_cache is None:
        import librosa

        _mel_basis_cache = librosa.filters.mel(
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
            htk=True,
            norm=None,
        ).astype(np.float32)
    return _mel_basis_cache


def _compute_mel(audio: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute log-Mel spectrogram matching the O&V model input.

    Returns:
        2D array of shape (n_frames, n_mels) with dB-scaled Mel spectrogram.
    """
    import librosa

    stft = librosa.stft(
        audio, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann", center=True
    )
    power = np.abs(stft) ** 2

    mel_basis = _get_mel_basis()
    mel_spec = mel_basis @ power

    # Convert to dB scale matching torchaudio AmplitudeToDB(stype="power", top_db=80)
    log_mel = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - TOP_DB)

    return log_mel.T.astype(np.float32)


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------


def _run_model(
    mel: NDArray[np.float32], model_path: str | Path
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Run ONNX model inference on a Mel spectrogram.

    Args:
        mel: (n_frames, 229) Mel spectrogram.
        model_path: Path to ONNX model file.

    Returns:
        onset_probs: (88, n_frames) onset probabilities.
        velocity_probs: (88, n_frames) velocity estimates.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    # (n_frames, n_mels) → (1, n_mels, n_frames)
    mel_input = mel.T[np.newaxis, :, :]

    onsets, velocities = session.run(None, {input_name: mel_input})
    # (1, 88, n_frames) → (88, n_frames)
    return onsets[0], velocities[0]


# ---------------------------------------------------------------------------
# Note tracking: onset probs → Note list
# ---------------------------------------------------------------------------


def _gaussian_kernel(ksize: int = 11, sigma: float = 1.0) -> NDArray[np.float32]:
    """Create a 1D Gaussian kernel."""
    x = np.arange(ksize, dtype=np.float32) - (ksize - 1) / 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def _smooth(probs: NDArray[np.float32], ksize: int = 11, sigma: float = 1.0) -> NDArray[np.float32]:
    """Apply Gaussian smoothing along time axis. probs: (88, T)."""
    kernel = _gaussian_kernel(ksize, sigma)
    pad = len(kernel) // 2
    result = np.zeros_like(probs)
    for i in range(probs.shape[0]):
        padded = np.pad(probs[i], pad, mode="edge")
        result[i] = np.convolve(padded, kernel, mode="valid")[: probs.shape[1]]
    return result


def _nms(probs: NDArray[np.float32], window: int = 3) -> NDArray[np.float32]:
    """1D non-maximum suppression along time axis. probs: (88, T)."""
    from scipy.ndimage import maximum_filter1d

    maxima = maximum_filter1d(probs, size=window, axis=1)
    return probs * (probs == maxima) * (probs > 0)


def _track_notes(
    onset_probs: NDArray[np.float32],
    velocity_probs: NDArray[np.float32],
    onset_threshold: float = 0.75,
    min_note_duration_frames: int = 5,
    nms_window: int = 3,
    smooth_sigma: float = 1.0,
    smooth_ksize: int = 11,
) -> list[Note]:
    """Convert frame-level onset/velocity probabilities to a list of Notes.

    Args:
        onset_probs: (88, T) onset probabilities after sigmoid.
        velocity_probs: (88, T) velocity estimates after sigmoid.
        onset_threshold: Minimum probability to trigger a note onset.
        min_note_duration_frames: Minimum frames before a note can end.
        nms_window: Window size for non-maximum suppression.
        smooth_sigma: Gaussian smoothing sigma.
        smooth_ksize: Gaussian smoothing kernel size.

    Returns:
        List of Note objects sorted by start time.
    """
    frame_duration = HOP_LENGTH / SAMPLE_RATE

    # Smooth and NMS
    smoothed = _smooth(onset_probs, smooth_ksize, smooth_sigma)
    nms_probs = _nms(smoothed, nms_window)

    n_keys, n_frames = nms_probs.shape

    # active_notes: key_idx → (onset_frame, velocity)
    active_notes: dict[int, tuple[int, int]] = {}
    notes: list[Note] = []

    for t in range(n_frames):
        time_sec = t * frame_duration

        for key_idx in range(n_keys):
            prob = nms_probs[key_idx, t]
            midi_note = key_idx + PIANO_MIN_MIDI

            if prob >= onset_threshold:
                # Close previous note on same key
                if key_idx in active_notes:
                    onset_frame, vel = active_notes.pop(key_idx)
                    onset_time = onset_frame * frame_duration
                    if time_sec > onset_time:
                        notes.append(Note(midi_note, onset_time, time_sec, vel))

                # Read velocity (average over small window)
                vel_start = max(0, t - 1)
                vel_end = min(n_frames, t + 2)
                raw_vel = float(np.mean(velocity_probs[key_idx, vel_start:vel_end]))
                midi_vel = max(1, min(127, int(round(raw_vel * 127))))

                active_notes[key_idx] = (t, midi_vel)

        # Check for note-offs based on minimum duration + decay
        expired = []
        for key_idx, (onset_frame, vel) in active_notes.items():
            frames_active = t - onset_frame
            if frames_active >= min_note_duration_frames:
                current_prob = onset_probs[key_idx, t]
                if current_prob < onset_threshold * 0.3:
                    midi_note = key_idx + PIANO_MIN_MIDI
                    onset_time = onset_frame * frame_duration
                    if time_sec > onset_time:
                        notes.append(Note(midi_note, onset_time, time_sec, vel))
                    expired.append(key_idx)

        for key_idx in expired:
            del active_notes[key_idx]

    # Flush remaining active notes
    final_time = n_frames * frame_duration
    for key_idx, (onset_frame, vel) in active_notes.items():
        midi_note = key_idx + PIANO_MIN_MIDI
        onset_time = onset_frame * frame_duration
        if final_time > onset_time:
            notes.append(Note(midi_note, onset_time, final_time, vel))

    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transcribe(
    audio_path: str | Path,
    model_path: str | Path,
    onset_threshold: float = 0.75,
) -> list[Note]:
    """Transcribe a WAV file to a list of piano notes.

    Args:
        audio_path: Path to input audio file (WAV, FLAC, etc.).
        model_path: Path to ONNX model file.
        onset_threshold: Onset detection threshold (0-1). Higher = fewer notes.

    Returns:
        List of Note objects sorted by start time.
    """
    audio = _load_audio(audio_path)
    if len(audio) == 0:
        return []

    mel = _compute_mel(audio)
    onset_probs, velocity_probs = _run_model(mel, model_path)
    return _track_notes(onset_probs, velocity_probs, onset_threshold=onset_threshold)


def save_midi(notes: list[Note], output_path: str | Path) -> None:
    """Write a list of Notes to a MIDI file.

    Args:
        notes: List of Note objects.
        output_path: Where to save the .mid file.
    """
    import pretty_midi

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name="Piano")

    for note in notes:
        if note.end > note.start:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end,
                )
            )

    midi.instruments.append(instrument)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))
