# 研究计划:语音—歌声共存下的 ASR 源选择行为

### 隐式语言先验与显式 LLM 解码器的对比研究

**目标会场**:ICASSP 2027(Toronto,2027 年 5 月 16–21 日)
**截稿**:2026 年 9 月 16 日 23:59 UTC-12(AoE)
**征稿页**:https://2027.ieeeicassp.org/call-for-papers/
**篇幅**:4 页正文 + 1 页参考文献
**距今**:约 5 周
**版本**:v3(纳入 Qwen3-ASR 与生产数据方案)

---

## 一、摘要式定位

我们在直播 ASR 生产环境中观察到:当背景音乐含人声时,ASR 的行为是**双峰且不稳定的**——有时把歌词一并转录,有时只转录主播的说话声。

本工作把这一现象形式化为**无条件的源选择歧义**:两个都合法的语音源同时存在、且不给模型任何条件时,它如何选择?

我们在**隐式语言模型解码器(Whisper)与显式 LLM 解码器(Qwen3-ASR)两条架构线**上做受控对比,刻画选择的相变行为、定位决策发生的位置、并给出轻量缓解方法。

**核心问题**:更强的语言先验,是加剧还是缓解源选择歧义?

---

## 二、问题背景

### 2.1 现象描述

直播场景中主播说话与背景音乐(BGM)常并存,而 BGM 往往含人声。此时 ASR 输出呈三种模式:

1. 仅转录主播说话声(期望行为)
2. 说话声与歌词混合转录(污染)
3. 主要转录歌词,主播说话被压制(严重污染)

三者切换缺乏明显规律,同一路流在相近条件下可能给出不同结果。

### 2.2 为什么这是个真问题

**危害是隐蔽的。** 混入的歌词在语法上流畅通顺,下游消费者(意图分类、内容理解、推荐特征、审核)无法从文本本身判断是否被污染。这与"ASR 出错导致文本不通顺"性质完全不同——常规错误可被下游语言模型部分吸收,歌词污染则被当作真实内容处理。

**影响是级联的。** 内容理解链路以 ASR 为共享上游,一次污染同时传播到所有下游任务。

**趋势上会更严重(如果 H-LM 成立)。** 业界正从隐式 LM 解码器转向显式 LLM 解码器。若更强的语言先验加剧了歧义,这个问题会随架构演进而放大,而非缓解。

---

## 三、相关工作

分为六簇。前两簇问题相邻但目标不同,第三、四簇提供可直接借用的方法与证据,第五簇是最新的对照研究范式,第六簇是空白所在。

### 簇一:Whisper 幻觉(输入无合法语音)

| 工作 | 内容 | 链接 |
|---|---|---|
| Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio | 用 AudioSet、MUSAN、UrbanSound8K、FSD50K 构造纯非语音数据集,任何输出即幻觉;系统收集幻觉模式与重复循环 | https://arxiv.org/abs/2501.11378 |
| Whisper Hallucination Detection and Mitigation via Hidden Representation Steering and SAE | 指出训练数据未经人工筛选,静音/噪声/音乐被配上任意文本,模型学到虚假关联;用隐藏表示 steering + SAE 检测与缓解 | https://arxiv.org/abs/2606.07473 |

**与本工作的关系**:输入中**没有**合法语音,不存在"选择"问题。

**可直接引用的论证**:第一篇构造数据集时**主动剔除了所有音乐**,理由是标注无法可靠区分纯器乐与含人声音乐。**我们要研究的样本被前人显式排除在研究范围之外。**

**可借用**:第二篇是缓解方法的现成模板。

### 簇二:歌词转写 ALT(目标就是歌词)

| 工作 | 内容 | 链接 |
|---|---|---|
| LyricWhiz | Whisper + ChatGPT 零样本多语种歌词转写;发现 Whisper 会输出与歌词无关内容(音乐描述、emoji、水印、广告),用 prompt 前缀引导 | https://arxiv.org/abs/2306.17103 |
| Exploiting Music Source Separation for ALT with Whisper | 系统评估音源分离作为前处理的影响 | https://arxiv.org/abs/2506.15514 |
| Enhancing Lyrics Transcription with Consistency Loss(Interspeech 2025) | LoRA + 一致性损失对齐 vocal 与 mixture 编码器表示,不依赖音源分离 | https://arxiv.org/abs/2506.02339 |
| VietLyrics | 越南语 ALT 数据集与模型;用 `<nospeech>` 阈值抑制幻觉 | https://arxiv.org/abs/2510.22295 |

**与本工作的关系**:目标是歌词、伴奏是噪声,与我们相反;且其"噪声"中不含另一个合法说话人。

### 簇三:Whisper 可解释性与层级探针(方法可直接借用)

| 工作 | 关键发现 | 链接 |
|---|---|---|
| **Beyond Transcription: Mechanistic Interpretability in ASR** | **最接近本工作**。Whisper 在**解码器残差流**中把"语音 vs 非语音"编码为线性可分的基本区分,尽管对两类输入都生成自信转写。层级图谱:性别第 25 层达峰 94.6%;清晰/嘈杂第 27 层 90.0%;口音 10–28 层完美分类 | https://arxiv.org/abs/2508.15882 |
| On the Interpretability of Whisper Encodings Using SAE | SAE 应用于 Whisper 编码,讨论能否识别高层语言结构 | https://arxiv.org/abs/2605.12225 |
| The Cascade Equivalence Hypothesis | Whisper 的 CTC 文本可解码性非单调:第 16 层降至 0.16、第 31 层回升至 0.26,存在编码瓶颈;**声学信息全程保留**(能量 R² 0.99→0.96) | https://arxiv.org/abs/2602.17598 |
| Probing Whisper for Dysarthric Speech | Whisper-medium 中层(13–15)最有信息量;微调仅带来轻微改变 | https://arxiv.org/abs/2510.04219 |
| Layer-wise Probing of wav2vec 2.0 and Whisper (CCR in AAE) | 探针方法学;综述早期层声学、中层音位、高层词汇语义的既有结论 | https://arxiv.org/abs/2606.23948 |
| Behind the Scenes: Mechanistic Interpretability of LoRA-adapted Whisper (SER) | 层贡献探针、logit-lens、SVD/CKA 的组合方法 | https://arxiv.org/abs/2509.08454 |

### 簇四:编码器保留背景声信息(H1 核心证据)

| 工作 | 关键发现 | 链接 |
|---|---|---|
| **Whisper-AT**(Interspeech 2023) | Whisper 虽对背景声鲁棒,但其音频表示**并非噪声不变**,而是与非语音声高度相关,**说明 Whisper 是"以噪声类型为条件"识别语音的**;冻结骨干 + 轻量标注头,<1% 额外算力即可输出音频事件 | https://arxiv.org/abs/2307.03183 · https://github.com/YuanGongND/whisper-at |

**"以噪声类型为条件识别语音"意味着模型内部本就存在关于背景类型的表示,并会调制识别行为。** 我们的假设正是:当背景被判定为"含人声的音乐"时,该条件化机制变得不稳定。

### 簇五:隐式 vs 显式 LM 解码器的对照研究(实验范式模板)

| 工作 | 内容 | 链接 |
|---|---|---|
| Do LLM Decoders Listen Fairly? | 用 Whisper small→medium→large-v3(隐式 LM 缩放)、Qwen3 0.6B→1.7B(显式 LLM)、Granite 2B→8B 做受控对比,研究 LM 先验如何塑造 ASR 偏见 | https://arxiv.org/abs/2604.21276 |

**这篇给了我们两样东西**:一个已被验证可发表的实验设计范式,以及一个可对齐的模型选型。我们把同一范式用于源选择歧义。

### 簇六:多源竞争下的选择——空白所在

目标说话人 ASR(TS-ASR)、多说话人 ASR 存在"转录哪个源"的问题,但**目标由说话人 embedding 或注册语音显式指定**。

**没有工作研究:两个合法语音源同时存在、且不给任何条件时,模型自身如何选择,以及为何不稳定。**

### 3.7 定位总结

| 研究方向 | 转写目标 |
|---|---|
| Whisper 幻觉 | 无目标(输入无语音) |
| 歌词转写 ALT | 歌词(伴奏是噪声) |
| 标准 ASR | 说话声(音乐是噪声) |
| 目标说话人 ASR | 由 embedding **显式指定** |
| **本工作** | **两个合法语音源,无任何条件,模型自选** |

**一句话:这不是鲁棒性问题,是无条件的源选择歧义问题。**

### 3.8 与最接近一篇的承接

*Beyond Transcription* 已证明 Whisper 把"语音 vs 非语音"编码为线性可分方向。我们的切入点是它未回答的问题:

> **唱歌既是语音也是音乐。它在这个线性方向上的投影落在哪一侧?该投影的不确定性,是否就是行为不稳定的来源?**

建议直接写进 introduction 主动建立承接,不要等审稿人指出。

---

## 四、研究问题与假设

**RQ1** 源选择是渐变还是相变?由哪些声学 / 语言因素驱动?
**RQ2** 决策发生在编码器还是解码器?
**RQ3** 更强的语言先验(显式 LLM 解码器)加剧还是缓解歧义?
**RQ4** 能否定位到具体的层、方向或注意力头?
**RQ5** 能否用轻量干预稳定该行为?

### 待检验假设

| | 假设 | 先验依据 |
|---|---|---|
| **H1** | 编码器保留两源信息,选择发生在解码器 | Whisper-AT 表明编码器保留并高度相关于背景声;Cascade Equivalence 报告声学信息全程保留 |
| **H2** | 存在由相对能量(SNR)驱动的相变点,而非平滑过渡 | 生产观察到的是双峰而非渐变 |
| **H3** | "歌唱性"本身(而非内容或说话人)是开关的主要输入 | 用 §5.3 朗读对照直接检验 |
| **H4** | 语言一致性是隐式开关(中文主播 + 中文歌 vs + 英文歌行为不同) | Whisper 多任务训练含显式语言 ID;若成立说明选择发生在语言判定之后 |
| **H-LM** | **显式 LLM 解码器的更强语言先验会加剧歌词泄漏** | LLM 解码器更擅长把模糊声学证据补全为流畅文本;且 Qwen3-ASR 被显式训练支持歌曲识别 |
| **H5** | 存在可因果干预的线性方向或组件 | *Beyond Transcription* 已证明 speech/non-speech 方向线性可分 |

**H-LM 是本版新增的核心假设,也是论文最有分量的一条**——它把研究对象从"一个老模型的缺陷"提升为"一个架构趋势的副作用"。

---

## 五、数据构造(关键路径)

### 5.1 设计原则

必须能**独立获知两个源各自的 ground truth**,才能精确度量歌词泄漏。因此以受控人工混合承载主结论,生产数据作动机与验证。

### 5.2 数据来源

| 角色 | 来源 | 说明与链接 |
|---|---|---|
| 说话声(目标) | 生产环境抽取的干净主播语音段 | 需走脱敏流程 |
| 中文歌声 + 伴奏 | **MIR-1K** | 1000 片段,伴奏与歌声**分录左右声道**,含音高轮廓、清音帧、歌词、人声/非人声段标注 https://zenodo.org/records/3532216 |
| 英文歌声 + 伴奏 | **MUSDB18-HQ** | 150 首整轨,提供 mixture/vocals/drums/bass/other 分轨,44.1 kHz 无损;仅限学术用途 https://zenodo.org/records/3338373 · 工具 https://github.com/sigsep/sigsep-mus-db |
| 真实验证集 | 生产环境真实带 BGM 片段 | 见 §9 合规路径 |

### 5.3 MIR-1K 提供的关键对照

**MIR-1K 还提供了由同一位演唱者朗读同样歌词的语音录音。**

这给出一个近乎理想的控制变量设计:**内容相同、说话人相同,唯一差异是"唱"还是"说"**。

| 条件 | 干扰源 | 检验目标 |
|---|---|---|
| C1 | 纯器乐伴奏(无人声) | 基线,应仅转录主播 |
| C2 | 纯歌声(无伴奏) | 纯人声竞争 |
| C3 | 完整歌曲(歌声 + 伴奏) | 生产场景 |
| **C4** | **同一人朗读的同样歌词** | **分离"歌唱性"这一变量** |

**C2 与 C4 的对比是本工作最干净的实验**:若泄漏率显著不同,说明韵律层面的"歌唱性"本身是开关输入,而非内容或说话人差异所致(直接检验 H3)。该对照由 MIR-1K 白送,应优先做。

### 5.4 混合协议

- **SNR 扫描**:主播语音相对干扰源,−10 dB 至 +20 dB,步长 2.5 dB;**范围由生产分布的 5–95 分位校准**(见 §9)
- **语种组合**:中—中、中—英、中—器乐(覆盖 H4)
- **片段长度**:10–30 s,避免长音频分段逻辑引入混淆
- **采样率**:统一 16 kHz 单声道
- 随机种子固定;混合脚本与配置随论文开源

### 5.5 规模建议

优先保证条件覆盖完整性。粗估:4 条件 × 13 个 SNR 点 × 3 语种组合 × 每格 50 片段 ≈ 7800 条。

---

## 六、模型矩阵

**本版核心变化:从"Whisper 为主"改为"隐式 vs 显式 LM 解码器双线对比"。**

| 模型 | 解码器类型 | 规模档 | 参数 | 作用 |
|---|---|---|---|---|
| Whisper small / medium / large-v3 | **隐式 LM**(解码器从头训练) | 3 | — | 隐式先验的缩放曲线 |
| **Qwen3-ASR 0.6B / 1.7B** | **显式 LLM**(Qwen3 预训练解码器) | 2 | AuT 编码器 ~300M + Qwen3-1.7B 解码器(28 层,GQA,SwiGLU,MRoPE) | 显式先验的缩放曲线 |
| 生产 ASR 模型 | — | 1 | — | 真实性佐证 |

### 6.1 Qwen3-ASR 带来的三项方法学优势

**一、Mel 前端与 Whisper 完全一致**
16 kHz、128 mel bins、n_fft=400、hop_length=160、Hann 窗、Slaney mel scale、0–8 kHz——**与 Whisper 逐项相同**。
两个模型吃到的是相同的输入表示,跨模型曲线可直接叠加,无需"输入不可比"的免责声明。**这是很强的方法学优势,应在论文中明确指出。**

**二、官方将"歌曲识别"作为卖点**
Qwen3-ASR 明确支持多语种 speech / music / song 识别,即**被显式训练成"歌也要转"**。技术报告显示其在带 BGM 整曲转写上明显强于 Whisper-large-v3(EntireSongs-zh 13.91,Whisper 因表现过差标记为 N/A)。

这让 RQ3 变得尖锐:**一个被明确训练成能转歌词的模型,在说话与歌声共存时,是更容易泄漏还是选择更准?** 两种结果都是好故事。

**三、附带 ForcedAligner**
Qwen3-ForcedAligner-0.6B 支持 5 分钟内任意单元的时间戳预测,覆盖 11 种语言。可用于**精确定位泄漏发生在片段的哪一时刻**,为 Fig 1 增加时间维度。

链接:https://github.com/QwenLM/Qwen3-ASR · https://huggingface.co/Qwen/Qwen3-ASR-1.7B · 技术报告 https://arxiv.org/abs/2601.21337

### 6.2 版本选择建议

- **`Qwen/Qwen3-ASR-1.7B`(原生)+ 配套 vLLM 推理框架** → 跑 Fig 1 的大批量 SNR 扫描
- **`Qwen/Qwen3-ASR-1.7B-hf`(transformers 版)** → 做层级探针与因果干预,hook 更容易

链接:https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf

---

## 七、实验设计与交付图表

### Figure 1 — 相变曲线(核心图,RQ1 / H2)

扫 SNR × 条件 × 语种,画歌词泄漏率曲线,**五个模型叠在同一张图上**。

观察重点:
- 曲线形状:sharp transition 还是 gradual
- 突变点位置是否随语种 / 条件系统性移动
- 同一配置多次采样的方差:双峰性是否体现在样本间

即使不存在相变,"源选择是连续加权而非离散切换"同样有价值,只需调整叙事。

### Figure 2 — 两条缩放曲线(新增主图,RQ3 / H-LM)

**横轴模型规模,纵轴泄漏率,两条曲线:隐式 LM(Whisper 3 档)vs 显式 LLM(Qwen3-ASR 2 档)。**

四种可能结果及其叙事:

| 结果 | 结论 |
|---|---|
| 两条都随规模下降 | 是能力问题,可被规模解决 |
| 两条都不随规模改善 | **不是能力问题,是目标定义的歧义**(最强结论) |
| 显式 LLM 更高 | H-LM 成立,语言先验加剧歧义,趋势上会恶化 |
| 显式 LLM 更低 | 显式先验有助消歧,指向训练配方 |

**这张图是本版新增的最大价值所在。** 它把论文从"发现一个 bug"提升为"刻画一个架构趋势的副作用"。

### Figure 3 — 层级探针(枢纽图,RQ2 / H1)

逐层训练线性探针,三个任务:
1. **歌词存在性**:该层表示能否判别输入是否含歌声
2. **歌词内容可解码性**:能否解码出歌词词级信息
3. **speech / non-speech 方向投影**:复现 *Beyond Transcription* 的方向,考察唱歌样本落在何处——**与该文承接的关键实验**

覆盖 Whisper 编码器 + 解码器全部层,以及 Qwen3-ASR 的 AuT 编码器 + Qwen3 解码器 28 层。
按歌曲 / 说话人划分训练与测试,避免泄漏。

判读:各层歌词信息都高 → H1 成立,选择在解码器;某层后骤降 → 编码器主动丢弃。

**这张图决定缓解方法应做在编码侧还是解码侧。**
**跨架构对比额外回答一个问题:显式 LLM 解码器是否把选择推迟到了更靠后的层?**

### Figure 4 — 因果定位(加分项,RQ4 / H5)

同一 SNR 与条件下寻找配对样本(一泄漏、一不泄漏),将泄漏样本某层残差流激活替换为不泄漏样本的对应激活(activation patching),观察输出是否翻转。先逐层定位,再在定位层内逐注意力头细分。

**时间不足可整体裁剪,不影响投稿。建议不早于第 3 周投入。**

### Table 1 — 缓解方法(必须有,RQ5)

按实现成本排序:

1. **基于探针的泄漏检测器**——用 Fig 3 的探针在线检测,标记可疑片段交下游处理。最稳妥兜底。
2. **上下文注入**——**Qwen3-ASR 独有**:显式提示"只转录说话人"能否消除歧义?Whisper 做不了这个实验,是跨架构对比的额外收获。
3. **Steering vector**——从泄漏 / 不泄漏样本对提取方向差,推理时干预。参考 https://arxiv.org/abs/2606.07473
4. **解码约束**——若 H1 成立(选择在解码器),成本最低。

**纯分析论文在 ICASSP 易被批"无方法贡献",此项不可省。**

---

## 八、指标

| 指标 | 定义 |
|---|---|
| **歌词泄漏率(LLR)** | 转写中包含歌词参考 n-gram 的片段占比(主指标) |
| **泄漏词插入率** | 相对说话内容参考的插入错误中可归因于歌词的比例 |
| **说话内容 WER** | 确保缓解方法不损伤主任务 |
| **双峰系数** | 同配置内泄漏率分布的双峰性,量化"不稳定"本身 |
| 探针准确率 / AUC | 逐层报告 |
| 泄漏时间定位误差 | 用 ForcedAligner,可选 |

---

## 九、生产数据的使用与合规路径

**定位:生产数据提供动机与验证,不承载主结论。**

三个理由:缺乏两源 ground truth;审稿人无法复现;审批周期不可控。

### 9.1 四个不可替代的用途

**一、一个学术组写不出的数字**
> "在 N 小时真实直播流量中,X% 的含 BGM 片段出现歌词泄漏。"

放进 introduction,把"我们观察到一个现象"变成"这是一个有规模的生产问题"。**只需统计量,不需公开音频,是最高性价比的用途。**

**二、校准合成实验的参数范围**
先从生产数据测出人声 / BGM 能量比的实际分布,据此设定 SNR 扫描范围。论文中可写"扫描范围覆盖生产分布的 5–95 分位",而非拍脑袋定值。**审稿人会注意到这个细节。**

**三、纵向一致性:直接量化"不稳定"**
同一主播、同一 BGM、不同时间点的行为是否一致?合成数据只能测样本间方差,**生产数据能测同一配置在时间上的重复不一致性**——这才是我们最初观察到的现象本体。

**四、极端案例:主播跟唱**
主播跟着 BGM 一起唱时,两个"源"是同一个人、内容相同、都在唱。这是源选择的极限测试,**合成数据无论如何构造不出来**。作为 discussion 中的 case study,并可反向验证 H3。

### 9.2 合规路径(按难度排序)

| 内容 | 难度 | 建议 |
|---|---|---|
| 统计量(泄漏率、能量比分布) | 低 | **争取,主要目标** |
| 匿名化转写片段样例 | 中 | 争取 |
| 音频样本 | 高 | 不强求 |
| 数据集公开发布 | 极高 | 不考虑 |

论文中平台匿名化为 "a large-scale production livestreaming platform",工业界论文的标准做法。

### 9.3 风险隔离

- 合成数据的申请与构建**照常进行,不等待审批**
- 生产数据分析作为**并行支线**,拿到即加入,拿不到即砍掉
- **每一个主结论都必须只依赖合成数据**
- 若第 4 周仍未获批,直接放弃该部分,不为其推迟投稿

---

## 十、五周排期

| 周 | 内容 | 交付 |
|---|---|---|
| **1** | 数据集许可与脱敏启动;混合脚本与 C1–C4 构建;**Whisper 与 Qwen3-ASR 双模型 hook 跑通**;生产统计支线启动 | 数据集就绪 |
| **2** | 相变曲线(五模型)+ 缩放曲线 | Fig 1、Fig 2 |
| **3** | 层级探针(双架构)+ C2/C4 朗读对照 | Fig 3 |
| **4** | 因果干预(可裁剪)+ 缓解方法 + 撰写 | Fig 4、Table 1、初稿 |
| **5** | 打磨 + 内部评审 + 文献复检 | 终稿 |

### 最小可发版本

**Fig 1 + Fig 2 + Fig 3 + C2/C4 对照 + Table 1 即可成篇。** Figure 4 为加分项。

注意本版比 v2 多了 Fig 2,但它只需推理不需新方法,边际成本低、价值高。

---

## 十一、风险与应对

| 风险 | 概率 | 应对 |
|---|---|---|
| 数据构造超期(**最可能**) | 高 | 本周启动;MIR-1K 与 MUSDB18 均需 Zenodo 人工审批,**今天提交** |
| Qwen3-ASR 探针需重新摸索 | 中 | 架构已知(AuT + Qwen3 28 层);优先用 `-hf` 版本;若探针受阻,Qwen3-ASR 仅进 Fig 1/2/Table 1,不影响成篇 |
| 相变不存在,仅平滑过渡 | 中 | 仍可发:改为刻画加权函数 |
| 探针结果与 H1 不符 | 中 | 更好——说明编码器主动丢弃,结论更强 |
| H-LM 不成立(显式 LLM 反而更好) | 中 | 同样是好结论,叙事改为"显式先验有助消歧",并指向训练配方 |
| 干预无效 | 中 | 退回探针检测器作为实用产出 |
| 生产数据未获批 | 中 | 已做风险隔离(§9.3),主结论不受影响 |
| 被抢发 | 中 | 该方向迭代快;投稿前复检 Interspeech 2026 接收名单与 arXiv 近三个月 |

---

## 十二、为什么值得做

**一、无论结果如何都能成篇。** 相变存在 → 定位离散开关;不存在 → 连续加权模型。探针支持 H1 → 解码器选择;不支持 → 编码器丢弃。H-LM 成立 → 趋势警示;不成立 → 训练配方指引。所有分支都有清晰叙事。

**二、地基已由前人打好。** 层级功能分工、探针方法学、声学信息保留、speech/non-speech 线性方向、隐式 vs 显式 LM 对照范式——均不必重做。

**三、跨代模型对比挡住了最致命的质疑。** 纳入 Qwen3-ASR 后,"这只是老模型的历史包袱"这一质疑不再成立;而 mel 前端完全一致使对比无需免责声明。

**四、数据条件优越。** MIR-1K 的"同一人朗读同样歌词"对照可干净分离"歌唱性"变量,零额外标注成本。

**五、有直接业务价值。** 歌词污染传播到所有下游任务,且因转写流畅而不可察觉。

**六、可能白捡一个能力。** 若发现模型内部隐式做"唱歌 vs 说话"判别,该表示可直接用于音乐检测——恰是我们已有的业务需求。参照 Whisper-AT,冻结骨干加轻量头即可,额外算力 <1%。

---

## 十三、本周待办

- [ ] 向 Zenodo 申请 MIR-1K 与 MUSDB18-HQ 访问权限(需人工审批,**今天提交**)
- [ ] 确认两数据集许可条款覆盖我们的使用方式
- [ ] 从生产环境抽取干净主播语音段,启动脱敏流程
- [ ] **并行启动生产统计支线的合规申请**(泄漏率、能量比分布)
- [ ] 编写可控混合脚本(SNR 扫描 + 语种组合 + C1–C4)
- [ ] 跑通 Whisper 与 Qwen3-ASR(`-hf` 版)的逐层 hook,确认可取编码器与解码器各层激活
- [ ] 复现 *Beyond Transcription* 的 speech/non-speech 探针,作为方法学基线
- [ ] 复检最新文献(重点:Interspeech 2026 接收名单、arXiv 近三个月)

**数据是关键路径,卡住则后续全部顺延。**

---

## 附:参考文献链接汇总

**幻觉**
- Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio — https://arxiv.org/abs/2501.11378
- Whisper Hallucination Detection and Mitigation via Hidden Representation Steering and SAE — https://arxiv.org/abs/2606.07473

**歌词转写**
- LyricWhiz — https://arxiv.org/abs/2306.17103
- Exploiting Music Source Separation for ALT with Whisper — https://arxiv.org/abs/2506.15514
- Enhancing Lyrics Transcription on Music Mixtures with Consistency Loss — https://arxiv.org/abs/2506.02339
- VietLyrics — https://arxiv.org/abs/2510.22295

**可解释性**
- Beyond Transcription: Mechanistic Interpretability in ASR — https://arxiv.org/abs/2508.15882
- On the Interpretability of Whisper Encodings Using Sparse Autoencoders — https://arxiv.org/abs/2605.12225
- The Cascade Equivalence Hypothesis — https://arxiv.org/abs/2602.17598
- Probing Whisper for Dysarthric Speech — https://arxiv.org/abs/2510.04219
- Layer-wise Probing of wav2vec 2.0 and Whisper for CCR in AAE — https://arxiv.org/abs/2606.23948
- Behind the Scenes: Mechanistic Interpretability of LoRA-adapted Whisper for SER — https://arxiv.org/abs/2509.08454

**编码器与背景声**
- Whisper-AT — https://arxiv.org/abs/2307.03183 · https://github.com/YuanGongND/whisper-at

**隐式 vs 显式 LM 解码器**
- Do LLM Decoders Listen Fairly? — https://arxiv.org/abs/2604.21276

**模型**
- Qwen3-ASR Technical Report — https://arxiv.org/abs/2601.21337
- Qwen3-ASR GitHub — https://github.com/QwenLM/Qwen3-ASR
- Qwen3-ASR-1.7B — https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- Qwen3-ASR-1.7B-hf(探针用) — https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf

**数据集**
- MIR-1K — https://zenodo.org/records/3532216
- MUSDB18-HQ — https://zenodo.org/records/3338373
- musdb 工具 — https://github.com/sigsep/sigsep-mus-db

**会议**
- ICASSP 2027 Call for Papers — https://2027.ieeeicassp.org/call-for-papers/
