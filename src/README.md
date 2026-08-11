# src — 脚本与其对应的 pilot

全部在 GPU pod(`fdbd:dc53:13:72c::18`,L20)上运行,工作目录 `/opt/tiger/icassp2027/`。
路径在脚本里是写死的绝对路径,换机器需要改 `ROOT`。

## 数据准备

| 脚本 | 作用 |
|---|---|
| `extract_stems.py` | MUSDB18-HQ → 16 kHz 单声道 `vocals.wav` / `accomp.wav`(torchaudio 重采样) |
| `extract_rhythm.py` | 追加 `rhythm.wav`(drums+bass,**保证无人声**,供 C1 对照) |
| `build_mixtures.py` | 主切片集:5 语音 × 6 曲目 × 3 条件 × 5 SNR = 461 条 + manifest |
| `build_tracksweep.py` | 50 曲目扫描集:5 语音 × 50 曲目 × 3 SNR = 800 条 |

**切片本身没有入库**(几百 MB),但种子固定(`SEED = 20260811`),重跑即可复现同一批音频。

## 模型运行

| 脚本 | 作用 |
|---|---|
| `run_model.py` | 统一入口,后端 `ctc`(Triton)/ `whisper`(transformers),按 manifest 逐条转写 |
| `run_qwen.py` | Qwen3-ASR 后端,走第二台 pod 的 vLLM HTTP(`/v1/transcribe`),8 线程并发 |
| `ctc_torch.py` | **whisper_ctc_offline 的 PyTorch 加载器**——部署版是 TRT 引擎无法插 hook,机制实验全靠它。含对齐 DALI 的 log-mel 与 greedy CTC 解码 |
| `test_ctc_offline.py` | Triton 冒烟测试(preprocessing → asr_ensemble 两段式调用) |

## 各 pilot 对应脚本

| pilot | 结论 | 脚本 |
|---|---|---|
| 01 | 无解码器模型照样泄漏歌词(H1) | `ctc_leak_pilot.py`, `summarize_pilot.py` |
| 02 | CTC vs Whisper,Δ₀ 与 H-LM 反向 | `compare_ladder.py` |
| 03 | 50 曲目:双峰分布、r=0.78、可词汇化驱动 | `analyze_tracksweep.py`, **`check_circularity.py`**(证伪指标循环性的对照) |
| 04 | 三级阶梯单调,能力/倾向分离 | `ladder3.py` |
| **05** | **三次失败** | `probe_extract.py` + `probe_train.py`(天花板)、`logit_lens.py` / `logit_lens_norm.py`(中间层不在输出空间)、`patch_encoder.py`(整层 patch 等价于跑 donor) |
| 06 | 时间窗 patching,首次有定位能力 | `patch_window.py` |
| 07 | 200 ms 细化:承诺是累积的 | `patch_fine.py` |
| 08 | **Fig 3 定稿**:起始 2 s + 逐层固化至 ~L20 | `patch_window_n39.py` |
| 09 | **对照**:承诺是解码器的产物 | `patch_ctc.py` |
| 10 | 堵住帧数混淆,起始优势 6.4× | `patch_framematched.py` |
| Table 1 | 起始压低 + 扰动检测器 | `table1_mitigation.py`, `m2_detector.py` |

**pilot 05 的三个失败脚本刻意保留**,它们记录了三种"测量工具问不出问题"的具体形态,
比结论本身更值得复用——见 `docs/pilot-05-probing-attempt1.md`。

## 环境

| 脚本 | 作用 |
|---|---|
| `setup_env.sh` | 把 jiwer 等装到 `--target` 隔离目录,**不碰 Triton 镜像自带的 torch/DALI** |
| `get_whisper.sh` / `get_whisper2.sh` | 经 hf-mirror 取 whisper-large-v3(pod 直连 HuggingFace 会 429) |
| `probe_qwen.py` | 探测 Qwen3-ASR 的 `/v1/transcribe` 入参格式 |

## 复现时的两个坑

1. **pod 上 `sync.sh exec` 用的是 dash**:大括号展开会变成字面目录名,命令里出现单引号会打断 exec 封装。
   脚本一律 base64 传过去。
2. **`sync.sh` 的 HDFS 暂存路径由目录名决定**,从名字不是 `hyper_boot` 的 worktree 推送会静默传到别处
   (仍然打印 "Synced!")。
