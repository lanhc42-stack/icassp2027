# Pilot 01 — 无解码器模型是否泄漏歌词?

日期 2026-08-11 · 模型 `whisper_ctc_offline`(L20 pod) · 147 次推理 · 原始结果 `work/pilot_results.json`

## 目的

H1 的直接行为检验(§6.7 一)。CTC 是对编码器输出的逐帧线性读出,**没有解码器可供"选择"**。若它照样转出歌词,则编码器确实保留两源,抑制只能是解码器行为。

## 设置

- **目标语音**:LibriSpeech test-clean 3 条,已知参考文本
- **干扰源**:MUSDB18-HQ test 前 3 首,各取人声能量最高的 10 s 窗口
  - `C1_accomp` 伴奏(drums+bass+other,无主唱)
  - `C2_vocals` 纯人声
  - `C3_mixture` 完整混音
- **SNR**:−10 / −5 / 0 / +5 / +10 dB(语音相对干扰源)
- **指标(粗)**:`extra_ratio` = 输出中不在语音参考里的词占比。**这是泄漏的代理量,不是 LLR**——它把 ASR 普通错误也算了进去,所以必须对照下面的假阳性地板读。

## 结果一:H1 成立,而且非常干脆

**纯唱歌输入(无语音)时,模型流畅转出歌词:**

| 曲目 | 输出 |
|---|---|
| Angels In Amplifiers – I'm Alright | `I I am falling. Does anybody hear me call? I'm alright As I pick up the pieces that I've left to me,` |
| Al James – Schoolboy Facination | `She's too glad to see in my car. She looked just like a movie star. She 10 out of ten I by.` |
| AM Contra – Heart Peripheral | `Keep keep me when your got you in my heart.` |

**说话与歌声共存、且歌声占优时,输出被歌词接管:**

```
C2_vocals  SNR −10   'I am falling, Does anybody hear me call?'          ← 全部是歌词
C2_vocals  SNR −5    'I Does anybody hear me call.'
C2_vocals  SNR  0    'After early nightfall, the yellow lamps would…'    ← 全部是语音
```

**一个没有解码器的模型照样泄漏歌词 → 编码器保留了两源信息 → H1 成立。**
这条结论不依赖任何探针,是行为层面的直接证据。

## 结果二:存在拐点,不是平滑过渡(支持 H2)

假阳性地板(纯语音输入):**0.132**

| 条件 | −10 | −5 | 0 | +5 | +10 |
|---|---|---|---|---|---|
| C1_accomp | 0.55 | 0.41 | 0.25 | 0.19 | 0.16 |
| **C2_vocals** | **0.69** | **0.40** | **0.13** | **0.10** | **0.12** |
| C3_mixture | 0.59 | 0.46 | 0.19 | 0.16 | 0.16 |

C2 从 −5 的 0.40 掉到 0 的 0.13(**已等于地板**),之后持平。**拐点落在 −5 与 0 dB 之间,不是渐变。** SNR ≥ 0 时基本不泄漏。

## 结果三:曲目间方差极大——这就是生产上观察到的"不稳定"

同样 SNR、同样条件,**只换一首歌**:

| 曲目 | −10 | −5 | 0 | +5 | +10 |
|---|---|---|---|---|---|
| AM Contra – Heart Peripheral | **0.19** | 0.20 | 0.11 | 0.08 | 0.09 |
| Al James – Schoolboy Facination | **0.95** | 0.38 | 0.14 | 0.10 | 0.13 |
| Angels In Amplifiers – I'm Alright | **0.93** | 0.63 | 0.14 | 0.13 | 0.13 |

−10 dB 处 0.19 vs 0.95,**差 5 倍**。曲目身份的影响远大于 SNR 的边际影响。

**这可能是比相变更有价值的发现**:生产上"同一路流时好时坏"的双峰行为,在这里被还原为"取决于干扰歌曲的某种属性"。那个属性是什么(人声突出度?演唱风格?伴奏密度?)——**值得作为 RQ1 的一个独立子问题**,§7 Fig 1 的方差分析应该按曲目分层,而不是只报均值。

## 结果四:高 SNR 下伴奏比人声更有害(反直觉)

SNR ≥ 0 时 **C1(纯伴奏)0.25 > C2(纯人声)0.13**,顺序与 −10 dB 时相反。

即:语音占优时,一个**竞争人声**反而比**纯器乐**更容易被正确忽略。这与"人声竞争最难"的直觉相反,值得追。

**但先排除一个混淆**:MUSDB 的 `other` 分轨常含和声与背景人声,所以 C1 并非严格"无人声"。**下一步须换用真正纯器乐的对照**,否则这条结论不成立。

## 附带发现:重压下语言判定会翻车

−10 dB 的 C3 出现 `'Я им в те и.'`(俄语)和 `' falling anybody hear越.'`(中文字)。多语种词表模型在音乐压制下会**翻到别的语言**。这与 H4(语言一致性是隐式开关)直接相关,且呼应 `Do LLM Decoders Listen Fairly?` 报告的 Whisper 在静音注入下的选择性幻觉。

另有少量 `<preprocessing-error>`(C1 −10):VAD 判定无语音而直接拒绝。全量实验里要把这类单独计数,不能混进泄漏率。

## 必须记住的四条局限

1. **指标是代理量**。`extra_ratio` 混入了普通 ASR 错误(地板 0.132)。真正的 LLR 需要歌词参考,**而 MUSDB18-HQ 不带歌词标注**——这正是 MIR-1K / NHSS 不可替代的原因。
2. **重采样是粗的**。pilot 用 `np.interp` 从 44.1k 降到 16k,有混叠。正式实验已改用 `torchaudio` 的 resampler(见 `work/extract_stems.py`)。
3. **样本量极小**:3 语音 × 3 曲目。曲目方差那条结论尤其需要扩到全部 50 首才能下定论。
4. **C1 不是干净的无人声对照**(见结果四)。

## 对计划的影响

- **H1 可以从"待检验"改为"已有直接证据"**,Fig 3 的角色随之从"发现"转为"定位与交叉验证"。
- **Fig 1 必须按曲目分层报方差**,不能只报 SNR 曲线均值。
- 结果四给了一个新的、便宜的对照:**真·纯器乐 vs 含人声伴奏**。
- 阶梯零点(§6.1)已有实测锚点:CTC 在 −10 dB 泄漏 0.69,远高于 0.132 的地板。
