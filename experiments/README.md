# experiments — 原始结果

所有数字都由这些文件生成;`docs/` 里的每张表都能追回到这里的某个 JSON。
生成脚本见 `../src/README.md`。

## 转写结果 `runs/`

每行一条切片:`{track, utt, cond, snr, ref, hyp, lang}`(Qwen3 另含 `words` 词级时间戳)。

| 文件 | 模型 | 切片集 |
|---|---|---|
| `ctc.json` / `ctc_ts.json` | whisper_ctc_offline(无 LM) | 461 条主集 / 800 条 50 曲目扫描 |
| `whisper_large_v3.json` / `_ts.json` | Whisper large-v3(隐式 LM) | 同上 |
| `qwen3_asr_1.7b.json` / `_ts.json` | Qwen3-ASR 1.7B(显式 LLM+歌曲训练) | 同上,**含 ForcedAligner 词级时间戳(尚未使用)** |

`manifest.json` / `manifest_ts.json` 是两个切片集的清单(种子 `20260811` 可复现音频)。

## 机制实验

| 文件 | pilot | 内容 |
|---|---|---|
| `pilot_results.json` | 01 | 首个泄漏 pilot(147 次推理,粗指标) |
| `patch_encoder.json` | 05 | **失败**:整层 patching,每层都归零 |
| `logit_lens.json` / `logit_lens_norm.json` | 05 | **失败**:logit lens,中间层概率量级 1e-5 |
| `patch_window.json` | 06 | 2 s 窗 × 层,8 配对(首个有定位能力的结果) |
| `patch_fine.json` | 07 | 200 ms 窗,39 配对 |
| `patch_window_n39.json` | **08** | **Fig 3 主数据**:2 s 窗 × 9 层,39 配对 |
| `patch_ctc.json` | **09** | **对照主数据**:同实验跑在无解码器模型上 |
| `patch_framematched.json` | 10 | 33 帧等向量数对照 |

每个 patching 文件都含 `checks` 字段(S1 自我 patch / S2 padding / S3 全帧),
**读结果前先看这三项**——pilot 05 的教训是操作可能根本没有区分力。

## Table 1

| 文件 | 内容 |
|---|---|
| `table1.json` | M1 起始压低:baseline / 前 2 s / 末 2 s / 全片 × 3 模型 × 48 配对 |
| `m2_detector.json` | M2 扰动检测器(**修正版**:补静音而非截断,SNR 均衡集合) |

## 未入库

- `probe_feats.npz`(104 MB,pilot 05 的探针激活)——太大且实验已作废,需要时用
  `src/probe_extract.py` 重生成。
- 切片 wav(约 380 MB)——种子固定,用 `src/build_*.py` 重生成。
