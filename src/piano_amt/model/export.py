"""Export the O&V PyTorch model to ONNX format.

The model uses Tensor.diff() internally which isn't always well-supported
by ONNX export. We wrap the model to compute the diff externally and pass
both channels (spectrogram + temporal derivative) as a pre-computed input.

Two export strategies:
1. ExportableWrapper: wraps the model to handle diff externally
2. Direct export: if the model supports it natively
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class ExportableWrapper(nn.Module):
    """Wraps the O&V model for ONNX export.

    The original model takes input (batch, 229, T) and internally computes
    the temporal derivative. For ONNX compatibility, we pre-compute the
    derivative and directly call the onset/velocity stages.

    Input to this wrapper: mel spectrogram (batch, 229, T)
    Outputs: onset_probs (batch, 88, T), velocity_probs (batch, 88, T)
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with explicit diff computation.

        Replaces aten::diff (unsupported by ONNX) with manual subtraction,
        then calls the model's internal stages directly.

        Args:
            x: Mel spectrogram (batch, 229, T)

        Returns:
            onset_probs: (batch, 88, T) sigmoid onset probabilities
            velocity_probs: (batch, 88, T) sigmoid velocity values
        """
        model = self.model

        # Compute temporal diff manually instead of x.diff(dim=-1)
        xdiff = x[:, :, 1:] - x[:, :, :-1]
        x_trimmed = x[:, :, 1:]
        x2 = torch.stack([x_trimmed, xdiff], dim=1)  # (batch, 2, 229, T-1)

        # Run through model internals, bypassing forward_onsets
        x2 = model.specnorm(x2)
        stem_out = model.stem(x2)
        stage_x = model.onset_stages[0](stem_out)
        x_stages = [stage_x]
        for stg in model.onset_stages[1:]:
            stage_x = stg(stem_out) + x_stages[-1]
            x_stages.append(stage_x)
        for st in x_stages:
            st.squeeze_(1)

        onset_logits = x_stages[-1]
        stem_out = torch.cat([stem_out, onset_logits.unsqueeze(1)], dim=1)
        velocities = model.velocity_stage(stem_out).squeeze(1)

        onset_probs = torch.sigmoid(onset_logits)
        velocity_probs = torch.sigmoid(velocities)

        # Pad to restore original time dimension T
        onset_probs = torch.nn.functional.pad(onset_probs, (1, 0))
        velocity_probs = torch.nn.functional.pad(velocity_probs, (1, 0))

        return onset_probs, velocity_probs


def export_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    n_frames: int = 130,
) -> Path:
    """Export the O&V model to ONNX format.

    Args:
        checkpoint_path: Path to the PyTorch .torch checkpoint.
        output_path: Where to save the .onnx file.
        n_frames: Number of time frames for the example input (dynamic axes used).

    Returns:
        Path to the exported ONNX file.
    """
    from piano_amt.model.architecture import load_model_from_checkpoint

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading PyTorch model...")
    model = load_model_from_checkpoint(checkpoint_path, device="cpu")

    wrapper = ExportableWrapper(model)
    wrapper.eval()

    # Example input: (batch=1, n_mels=229, T=n_frames)
    dummy_input = torch.randn(1, 229, n_frames)

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(output_path),
        input_names=["mel_spectrogram"],
        output_names=["onset_probs", "velocity_probs"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch", 2: "time_frames"},
            "onset_probs": {0: "batch", 2: "time_frames"},
            "velocity_probs": {0: "batch", 2: "time_frames"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify
    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"ONNX model exported successfully: {output_path} ({size_mb:.1f} MB)")
    return output_path
