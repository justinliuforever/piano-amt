"""CLI entry point for piano-amt.

Two subcommands:
  piano-amt transcribe input.wav -o output.mid
  piano-amt evaluate data/smd/ -o results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "ov_model.onnx"


def _check_model(model_path: Path) -> None:
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}", file=sys.stderr)
        print("Run: python scripts/setup_model.py", file=sys.stderr)
        sys.exit(1)


def cmd_transcribe(args: argparse.Namespace) -> None:
    """Transcribe a WAV file to MIDI."""
    from piano_amt.transcribe import save_midi, transcribe

    model_path = Path(args.model)
    _check_model(model_path)

    notes = transcribe(args.input, model_path, onset_threshold=args.threshold)
    print(f"Detected {len(notes)} notes")

    if args.output:
        save_midi(notes, args.output)
        print(f"MIDI written to {args.output}")
    else:
        for note in notes[:20]:
            print(
                f"  [{note.start:7.3f}s - {note.end:7.3f}s] "
                f"pitch={note.pitch:3d} vel={note.velocity:3d}"
            )
        if len(notes) > 20:
            print(f"  ... and {len(notes) - 20} more notes")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate transcription accuracy on a dataset."""
    from piano_amt.evaluate import evaluate_dataset, print_report

    model_path = Path(args.model)
    _check_model(model_path)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating dataset: {data_dir}")
    results = evaluate_dataset(data_dir, model_path, threshold=args.threshold)
    print_report(results)

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "results.csv"
        results.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="piano-amt",
        description="Piano automatic music transcription (Audio → MIDI)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # transcribe
    p_transcribe = subparsers.add_parser(
        "transcribe", help="Transcribe a WAV file to MIDI"
    )
    p_transcribe.add_argument("input", help="Input audio file (WAV, FLAC, etc.)")
    p_transcribe.add_argument(
        "-o", "--output", help="Output MIDI file (prints to stdout if omitted)"
    )
    p_transcribe.add_argument(
        "--model", default=str(DEFAULT_MODEL_PATH), help="ONNX model path"
    )
    p_transcribe.add_argument(
        "--threshold", type=float, default=0.75, help="Onset threshold (0-1)"
    )

    # evaluate
    p_eval = subparsers.add_parser(
        "evaluate", help="Evaluate transcription on a dataset"
    )
    p_eval.add_argument("data_dir", help="Dataset directory (must contain metadata.csv)")
    p_eval.add_argument(
        "-o", "--output", help="Output directory for results CSV"
    )
    p_eval.add_argument(
        "--model", default=str(DEFAULT_MODEL_PATH), help="ONNX model path"
    )
    p_eval.add_argument(
        "--threshold", type=float, default=0.75, help="Onset threshold (0-1)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "transcribe": cmd_transcribe,
        "evaluate": cmd_evaluate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
