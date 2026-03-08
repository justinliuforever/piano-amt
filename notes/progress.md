# Piano AMT - Project Notes

## Current Status

**Phase: MVP complete, migrating to Azure VM**

Core pipeline functional: audio → mel spectrogram → ONNX inference → note tracking → MIDI output.
Evaluation system in place with mir_eval metrics. CI pipeline configured.
Azure VM created (`piano-amt-vm`, East US 2), pending environment setup and full MAESTRO WAV download.

## Architecture

```
audio.wav → _load_audio (16kHz mono)
          → _compute_mel (229-band log-Mel, hop=384)
          → _run_model (ONNX Runtime, output: 88-key onset_probs + velocity)
          → _track_notes (smoothing → NMS → state machine)
          → output.mid
```

Model: Onsets & Velocities (iamusica, 2023), pretrained F1=0.9675

## Key Paths

| What | Path |
|------|------|
| Core pipeline | `src/piano_amt/transcribe.py` |
| Evaluation | `src/piano_amt/evaluate.py` |
| CLI | `src/piano_amt/cli.py` |
| Model definition | `src/piano_amt/model/architecture.py` |
| ONNX export | `src/piano_amt/model/export.py` |
| Tests | `tests/test_transcribe.py` |
| Model weights | `models/ov_model.onnx` (gitignored) |
| CI | `.github/workflows/ci.yml` |
| Data docs | `notes/data.md` |
| WAV 合成脚本 | `scripts/synthesize_test_wavs.py` |

## Key Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| SAMPLE_RATE | 16000 | transcribe.py |
| N_MELS | 229 | transcribe.py |
| HOP_LENGTH | 384 | transcribe.py |
| N_FFT | 2048 | transcribe.py |
| FMIN / FMAX | 50 / 8000 Hz | transcribe.py |
| onset_threshold | 0.75 (default) | cli.py |
| MIN_NOTE_FRAMES | 5 | transcribe.py |

## Done

- [x] Project restructure: real-time → offline batch processing
- [x] Core transcription pipeline (audio → MIDI)
- [x] ONNX export wrapper (PyTorch → ONNX), fix `aten::diff` ONNX 导出 bug
- [x] Evaluation system (onset F1, onset+offset F1, velocity MAE, per-register)
- [x] CLI with `transcribe` and `evaluate` subcommands
- [x] CI pipeline (pytest + ruff)
- [x] MAESTRO dataset: MIDI 全量下载 (1276 files), 20 首 test split WAV 合成
- [x] WAV 合成脚本 (`scripts/synthesize_test_wavs.py`)
- [x] Unit/integration tests (5 test cases, `@pytest.mark.slow`)
- [x] Azure VM 创建 (Standard_D4s_v3, 4核/16GB, 512GB SSD, East US 2)

## TODO / Ideas

- [ ] VM 环境初始化 (Python, pip install, clone repo)
- [ ] 下载完整 MAESTRO WAV (~130GB) 到 VM
- [ ] 在 VM 上用真实 WAV 跑 evaluate, 获得真实 F1 基线
- [ ] Onset threshold tuning (当前默认 0.75, 可能需要按数据集调整)
- [ ] Note tracking algorithm improvements (state machine 参数优化)
- [ ] Pedal detection support
- [ ] GPU inference (当前仅 CPU)
- [ ] Streaming/chunked inference for long audio
- [ ] More comprehensive test coverage
- [ ] Benchmark on full MAESTRO dataset

## Decisions & Notes

- **Why ONNX?** Runtime 不需要装 PyTorch, 部署更轻量, CPU 推理够用
- **Why offline?** 从 real-time 转为 offline, 简化架构, 专注转录精度
- **Frame duration:** HOP_LENGTH/SAMPLE_RATE = 384/16000 = 24ms per frame
- **Piano range:** MIDI 21 (A0) ~ 108 (C8), 88 keys
- **Azure VM:** `piano-amt-vm`, D4s_v3, IP `20.109.16.200`, user `justinliu`, 512GB SSD
- **未来 GPU training:** 在 Azure 单独租 GPU VM 或使用 Azure ML
