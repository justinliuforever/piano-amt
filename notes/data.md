# Data Structure & Management

## Directory Layout

```
data/
├── maestro/                        # MAESTRO v3 dataset
│   ├── maestro-v3.0.0.csv          # 原始 MAESTRO CSV
│   ├── metadata.csv                # 统一格式索引 (1276 entries)
│   ├── .midi_downloaded            # 下载完成标记
│   ├── 2004/ ... 2018/             # 按年份组织
│   │   ├── *.midi                  # Ground truth MIDI
│   │   └── *.wav                   # 合成/真实音频 (部分)
│   └── ...
├── smd/                            # SMD 测试数据 (可选)
│   ├── chopin_op28_1.wav
│   └── chopin_op28_1.mid
└── custom/                         # 用户自定义数据 (未来)
    ├── metadata.csv                # 同样的 schema
    └── recordings/
```

## metadata.csv Schema

所有数据集统一使用相同的 CSV 格式：

| Column | Type | Description |
|--------|------|-------------|
| audio_path | str | WAV 文件相对路径 (相对于数据集根目录) |
| midi_path | str | MIDI 文件相对路径 |
| split | str | train / validation / test |
| duration | float | 时长 (秒) |
| composer | str | 作曲家 |
| title | str | 曲目名 |
| tags | str | 标签 (可选) |

## 当前数据状态

| Dataset | MIDI | WAV | 备注 |
|---------|------|-----|------|
| MAESTRO (全量) | 1276 files, ~56MB | 20/1276 test files (合成, 87MB) | 真实 WAV 需手动下载 ~130GB |
| SMD | - | - | 未下载，测试用 |

## 数据获取方式

```bash
# 下载 MAESTRO MIDI + metadata
python scripts/download_maestro.py --midi-only

# 从 MIDI 合成测试用 WAV (不是真钢琴, 但可以跑 pipeline)
python scripts/synthesize_test_wavs.py --n 20 --split test

# 真实 MAESTRO WAV: 需手动下载 (~130GB)
# https://magenta.tensorflow.org/datasets/maestro
```

## 添加自定义数据

1. 创建 `data/custom/` 目录
2. 放入 WAV 和 MIDI 文件
3. 创建 `metadata.csv`，遵循上面的 schema
4. 运行: `piano-amt evaluate --dataset data/custom`

## 关于合成 WAV vs 真实录音

- 合成 WAV 使用 `pretty_midi.synthesize()` 生成 (正弦波合成, 非钢琴音色)
- 模型在真实钢琴音频上训练, 合成音频的 F1 偏低 (~0.62) 是预期行为
- 完整评估需要真实 MAESTRO WAV 或自己录制的钢琴音频
- 合成 WAV 的价值: 验证 pipeline 是否跑通, 回归测试, CI 测试
