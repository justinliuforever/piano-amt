"""Evaluation module: compare predicted transcription against ground truth MIDI.

Supports single-pair evaluation and dataset-level evaluation with detailed
per-piece analysis reports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

from piano_amt.transcribe import Note, transcribe

PIANO_MIN_MIDI = 21
PIANO_MAX_MIDI = 108


# ---------------------------------------------------------------------------
# MIDI / Note extraction helpers
# ---------------------------------------------------------------------------


def _midi_to_notes(midi_path: str | Path) -> list[tuple[float, float, int, int]]:
    """Extract (start, end, pitch, velocity) tuples from a MIDI file."""
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            notes.append((note.start, note.end, note.pitch, note.velocity))
    notes.sort(key=lambda x: (x[0], x[2]))
    return notes


def _to_mir_eval_format(
    notes: list[tuple[float, float, int, int]] | list[Note],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert notes to mir_eval arrays: (intervals, pitches_hz, midi_pitches).

    Accepts either raw tuples (start, end, pitch, vel) or Note dataclass objects.
    """
    if not notes:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
        )

    if isinstance(notes[0], Note):
        intervals = np.array([(n.start, n.end) for n in notes], dtype=np.float64)
        midi_pitches = np.array([n.pitch for n in notes], dtype=np.int64)
    else:
        intervals = np.array([(n[0], n[1]) for n in notes], dtype=np.float64)
        midi_pitches = np.array([n[2] for n in notes], dtype=np.int64)

    pitches_hz = 440.0 * 2 ** ((midi_pitches.astype(np.float64) - 69) / 12)
    return intervals, pitches_hz, midi_pitches


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_pair(
    audio_path: str | Path,
    midi_path: str | Path,
    model_path: str | Path,
    threshold: float = 0.75,
    onset_tolerance: float = 0.05,
) -> dict[str, float]:
    """Evaluate transcription of a single audio file against ground truth MIDI.

    Returns a dictionary with:
    - Note counts: n_ref, n_est
    - Onset-only: onset_precision, onset_recall, onset_f1
    - Onset+offset: onset_offset_precision, onset_offset_recall, onset_offset_f1
    - Velocity: velocity_mae (mean absolute error of matched notes, MIDI units)
    - Timing: onset_deviation_mean_ms, onset_deviation_std_ms
    - Register: onset_f1_low (A0-B2), onset_f1_mid (C3-B5), onset_f1_high (C6-C8)
    - Error counts: false_positives, false_negatives
    """
    import mir_eval

    # Transcribe
    est_notes = transcribe(audio_path, model_path, onset_threshold=threshold)

    # Parse reference
    ref_raw = _midi_to_notes(midi_path)
    ref_intervals, ref_hz, ref_midi = _to_mir_eval_format(ref_raw)
    est_intervals, est_hz, est_midi = _to_mir_eval_format(est_notes)

    results: dict[str, float] = {
        "n_ref": float(len(ref_intervals)),
        "n_est": float(len(est_intervals)),
    }

    empty_metrics = {
        "onset_precision": 0.0, "onset_recall": 0.0, "onset_f1": 0.0,
        "onset_offset_precision": 0.0, "onset_offset_recall": 0.0, "onset_offset_f1": 0.0,
        "velocity_mae": 0.0,
        "onset_deviation_mean_ms": 0.0, "onset_deviation_std_ms": 0.0,
        "onset_f1_low": 0.0, "onset_f1_mid": 0.0, "onset_f1_high": 0.0,
        "false_positives": 0.0, "false_negatives": 0.0,
    }

    if len(ref_intervals) == 0 or len(est_intervals) == 0:
        results.update(empty_metrics)
        return results

    # --- Onset-only ---
    p, r, f, matched = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_hz, est_intervals, est_hz,
        onset_tolerance=onset_tolerance, offset_ratio=None,
    )
    results["onset_precision"] = p
    results["onset_recall"] = r
    results["onset_f1"] = f

    # --- Onset + offset ---
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_hz, est_intervals, est_hz,
        onset_tolerance=onset_tolerance, offset_ratio=0.2, offset_min_tolerance=0.05,
    )
    results["onset_offset_precision"] = p
    results["onset_offset_recall"] = r
    results["onset_offset_f1"] = f

    # --- Error counts ---
    n_matched = int(round(p * len(est_intervals)))  # TP from precision side
    results["false_positives"] = float(len(est_intervals) - n_matched)
    results["false_negatives"] = float(len(ref_intervals) - n_matched)

    # --- Velocity MAE & onset deviation (on matched note pairs) ---
    # Use mir_eval matching to find matched pairs
    matching = mir_eval.transcription.match_notes(
        ref_intervals, ref_hz, est_intervals, est_hz,
        onset_tolerance=onset_tolerance, offset_ratio=None,
    )

    if matching:
        ref_vels = np.array([ref_raw[i][3] for i, _ in matching], dtype=np.float64)
        est_vels = np.array(
            [est_notes[j].velocity for _, j in matching], dtype=np.float64
        )
        results["velocity_mae"] = float(np.mean(np.abs(ref_vels - est_vels)))

        ref_onsets = np.array([ref_intervals[i, 0] for i, _ in matching])
        est_onsets = np.array([est_intervals[j, 0] for _, j in matching])
        deviations_ms = (est_onsets - ref_onsets) * 1000.0
        results["onset_deviation_mean_ms"] = float(np.mean(deviations_ms))
        results["onset_deviation_std_ms"] = float(np.std(deviations_ms))
    else:
        results["velocity_mae"] = 0.0
        results["onset_deviation_mean_ms"] = 0.0
        results["onset_deviation_std_ms"] = 0.0

    # --- Per-register F1 ---
    # Low: A0-B2 (21-47), Mid: C3-B5 (48-83), High: C6-C8 (84-108)
    register_ranges = [
        ("onset_f1_low", 21, 47),
        ("onset_f1_mid", 48, 83),
        ("onset_f1_high", 84, 108),
    ]
    for key, lo, hi in register_ranges:
        ref_mask = (ref_midi >= lo) & (ref_midi <= hi)
        est_mask = (est_midi >= lo) & (est_midi <= hi)
        if ref_mask.sum() == 0 and est_mask.sum() == 0:
            results[key] = 0.0
            continue
        if ref_mask.sum() == 0 or est_mask.sum() == 0:
            results[key] = 0.0
            continue
        _, _, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals[ref_mask], ref_hz[ref_mask],
            est_intervals[est_mask], est_hz[est_mask],
            onset_tolerance=onset_tolerance, offset_ratio=None,
        )
        results[key] = f1

    return results


# ---------------------------------------------------------------------------
# Dataset-level evaluation
# ---------------------------------------------------------------------------


def evaluate_dataset(
    data_dir: str | Path,
    model_path: str | Path,
    threshold: float = 0.75,
) -> pd.DataFrame:
    """Evaluate all audio/MIDI pairs in a dataset directory.

    Expects a metadata.csv file in data_dir with columns:
    audio_path, midi_path (relative to data_dir).

    Returns:
        DataFrame with one row per piece and all metric columns.
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {data_dir}")

    metadata = pd.read_csv(metadata_path)

    rows = []
    skipped = 0
    for _, entry in metadata.iterrows():
        audio_path = data_dir / entry["audio_path"]
        midi_path = data_dir / entry["midi_path"]

        if not audio_path.exists() or not midi_path.exists():
            skipped += 1
            continue

        print(f"  Evaluating: {entry['audio_path']}...", end=" ", flush=True)
        metrics = evaluate_pair(audio_path, midi_path, model_path, threshold)
        print(f"F1={metrics['onset_f1']:.3f}")

        row = {
            "audio_path": entry["audio_path"],
            "midi_path": entry["midi_path"],
        }
        for col in ("split", "duration", "composer", "title", "tags"):
            if col in entry:
                row[col] = entry[col]
        row.update(metrics)
        rows.append(row)

    if skipped:
        print(f"  ({skipped} entries skipped — missing audio or MIDI files)")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def print_report(results_df: pd.DataFrame) -> None:
    """Print a detailed human-readable analysis report."""
    if results_df.empty:
        print("No results to report.")
        return

    print()
    print("=" * 90)
    print("EVALUATION REPORT")
    print("=" * 90)

    # --- 1. Per-piece table ---
    has_title = "title" in results_df.columns and results_df["title"].notna().any()
    label_col = "title" if has_title else "audio_path"

    print(f"\n{'#':>3}  {'Piece':<35} {'Notes':>7} {'Onset F1':>9} "
          f"{'On+Off F1':>9} {'Vel MAE':>8} {'Δt ms':>7}")
    print("-" * 84)
    for i, (_, row) in enumerate(
        results_df.sort_values("onset_f1", ascending=False).iterrows(), 1
    ):
        label = str(row[label_col])[:33]
        n_ref = int(row["n_ref"])
        print(
            f"{i:3d}  {label:<35} {n_ref:7d} {row['onset_f1']:9.4f} "
            f"{row['onset_offset_f1']:9.4f} {row.get('velocity_mae', 0):8.1f} "
            f"{row.get('onset_deviation_mean_ms', 0):+7.1f}"
        )

    # --- 2. Summary statistics ---
    print(f"\n{'Metric':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 62)
    summary_metrics = [
        "onset_f1", "onset_precision", "onset_recall",
        "onset_offset_f1",
        "velocity_mae", "onset_deviation_mean_ms", "onset_deviation_std_ms",
    ]
    for metric in summary_metrics:
        if metric not in results_df.columns:
            continue
        col = results_df[metric]
        print(f"  {metric:<28} {col.mean():8.3f} {col.std():8.3f} "
              f"{col.min():8.3f} {col.max():8.3f}")

    # --- 3. Note counts ---
    n_ref_total = int(results_df["n_ref"].sum())
    n_est_total = int(results_df["n_est"].sum())
    fp_total = int(results_df.get("false_positives", pd.Series([0])).sum())
    fn_total = int(results_df.get("false_negatives", pd.Series([0])).sum())
    print(f"\n  Reference notes:  {n_ref_total:,}")
    print(f"  Estimated notes:  {n_est_total:,}")
    print(f"  False positives (extra):   {fp_total:,}")
    print(f"  False negatives (missed):  {fn_total:,}")

    # --- 4. Per-register F1 ---
    reg_metrics = ["onset_f1_low", "onset_f1_mid", "onset_f1_high"]
    if all(m in results_df.columns for m in reg_metrics):
        print("\n  Register breakdown (mean onset F1):")
        print(f"    Low  (A0-B2):  {results_df['onset_f1_low'].mean():.4f}")
        print(f"    Mid  (C3-B5):  {results_df['onset_f1_mid'].mean():.4f}")
        print(f"    High (C6-C8):  {results_df['onset_f1_high'].mean():.4f}")

    # --- 5. By composer ---
    if "composer" in results_df.columns and results_df["composer"].notna().any():
        print(f"\n  {'Composer':<25} {'N':>4} {'Onset F1':>10} {'Vel MAE':>9}")
        print(f"  {'-'*50}")
        for composer, group in results_df.groupby("composer"):
            vel = group["velocity_mae"].mean() if "velocity_mae" in group.columns else 0
            print(f"  {composer:<25} {len(group):4d} "
                  f"{group['onset_f1'].mean():10.4f} {vel:9.1f}")

    print()
    print("=" * 90)
