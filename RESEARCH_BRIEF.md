# LCTSCap Research Brief

更新时间：2026-03-18 16:47 CST

## 1. Research Topic

**题目方向**：Hierarchical and Verifiable Long-Context Time Series Captioning

**一句话概括**：给定长时间序列，不再只做分类或短窗口识别，而是生成一段**层次化、可验证、可追溯**的自然语言描述，使模型能够像“分析者”一样解释较长时间范围内的行为模式和事件结构。

**目标论文定位**：
- 面向 CCF-A 级别会议的时序理解/多模态生成方向工作。
- 不把贡献停留在“又一个 caption 模型”，而是强调：
  1. 长上下文 time series captioning；
  2. 层次化中间表示；
  3. caption 的事实性与可验证性；
  4. 时序与文本的检索式对齐。

## 2. 写论文要解决的核心问题

### 2.1 为什么这不是普通 caption 任务

现有 time series caption 工作普遍存在四个不足：
- **上下文短**：常见设置只覆盖几个到几十个点，难以表达长程模式。
- **描述平面化**：只能输出一句全局描述，缺少“事件表-分段摘要-全局总结”的结构。
- **事实性弱**：文本看起来合理，但未必真的被原始时序支持。
- **模态对齐不足**：时间序列表示与文本表示没有形成可检索、可复用的共享空间。

### 2.2 本项目真正要回答的问题

如果论文要成立，必须回答下面这几个核心问题：

1. **如何把长时间序列变成适合生成文本的表示？**
   不能把整条原始信号直接喂给语言模型，必须先做语义 token 化和多级压缩。

2. **如何让模型不仅会“说”，还会“分层说”？**
   需要从事件级、片段级、全局级三个层面同时建模，而不是只做 end-to-end 文本生成。

3. **如何判断生成内容是不是被原始时序支持？**
   论文必须给出 grounding / factuality 评测，而不是只报 BLEU/ROUGE。

4. **如何证明文本和时序之间形成了有用的对齐空间？**
   这部分由 retrieval aligner 承担，用 R@K、双向检索等指标说明对齐能力。

## 3. 方法概述

### 3.1 数据表示

项目当前把时间序列定义成：
- 每个 token = 一个 10 秒传感器窗口；
- 一个样本由 `128 / 256 / 512` 个语义 token 组成；
- 对应真实时间大约是 `21.3 / 42.7 / 85.3` 分钟。

当前数据集设计：
- **主数据集**：CAPTURE-24
- **外部验证集**：HARTH

这一定义的价值是：把“长时间序列”从 raw points 的长度，转换成更接近模型可处理的**长语义上下文**。

### 3.2 模型设计

整体方法由四个模块组成：

1. **Local Encoder**
   - 对每个 `[channel, 500]` 的 10 秒窗口做编码；
   - 输出窗口级表示。

2. **Channel Fusion + Hierarchical Planner**
   - 先做多通道融合；
   - 再做 token 级与 segment 级建模；
   - 输出 `H_token` 和 `H_seg` 两层时序表示。

3. **Event Head + Retrieval Aligner**
   - `Event Head` 负责事件类型与时间跨度的建模；
   - `Retrieval Aligner` 用对比学习把时序和文本映射到同一空间。

4. **Caption Decoder**
   - 以 segment 表示和事件表示为条件，生成 caption；
   - Phase 4 再考虑接到冻结 LLM 上。

### 3.3 核心卖点

如果论文要成立，最关键的 method selling points 是：
- **长上下文建模**：不是短序列 caption。
- **层次化输出**：事件表 + 分段摘要 + 全局描述。
- **检索式对齐**：不是纯生成，而是带有跨模态对齐能力。
- **可验证性**：caption 中的活动、时序顺序、持续时间、跨度能回查到事件表。

## 4. 整个方法流程

### 4.1 离线数据流

1. 原始数据预处理
   - 下采样到 50Hz；
   - 切成 10 秒窗口；
   - 保存为张量和 manifest。

2. 长上下文样本构建
   - 以 `128 / 256 / 512` token 的上下文长度滑窗构造样本；
   - 保持 participant-level split，避免数据泄漏。

3. 自动标注构建
   - 合并连续同标签窗口形成 `Event`；
   - 根据事件生成 segment summary；
   - 根据统计和事件序列生成 short / long caption；
   - 后续再加入 paraphrase。

### 4.2 在线训练流

1. **Phase 0**
   - 不训练模型；
   - 跑模板基线，建立数据与评测闭环。

2. **Phase 1**
   - 训练编码器、层次规划器、事件头、检索对齐器；
   - 不开 decoder。

3. **Phase 2**
   - 冻结 local encoder 和 aligner；
   - 加入 caption decoder；
   - 在 `ctx=128/256` 上训练生成能力。

4. **Phase 3**
   - 引入 LLM paraphrase；
   - 扩展到 `ctx=512`。

5. **Phase 4**
   - 视实验结果决定是否接入 LLM bridge。

## 5. 当前工程进度

### 5.1 已完成

**Phase 0 已完成**
- 工程骨架、配置系统、脚本入口、数据 schema 已建好；
- CAPTURE-24 和 HARTH 的预处理与上下文样本生成已跑通；
- 自动模板标注已生成；
- 模板基线已经评估。

**Phase 1 已完成**
- Encoder + Planner + Event Head + Retrieval Aligner 已训练；
- 最优 checkpoint 已保存在：
  - `<data_root>/runs/phase1/checkpoints/best.pt`
- 关键结果：
  - best val loss = `30.54`
  - align loss 从 `1.02` 降到 `0.83`
  - event loss 从 `31.18` 降到 `28.62`

### 5.2 今天补齐的基础框架

为了真正进入 Phase 2，今天补了以下关键能力：
- Phase 2/3 配置继承与 `_config` 生效；
- `init_from` checkpoint 初始化；
- 模块级冻结/解冻；
- 离线 tokenizer 和 text encoder fallback；
- 评测脚本不再直接复用 gold 作为 prediction；
- paraphrase pipeline 可执行；
- 训练脚本改为把临时 manifest 写到 `/tmp`；
- 加载 checkpoint 时忽略 shape 不兼容的层，允许 Phase 2 decoder 重新初始化。

### 5.3 Phase 2 当前状态

截至本次记录，Phase 2 已经成功启动，并且越过了最容易失败的几个阶段：
- 成功加载 GPU 2；
- 成功加载 `ctx=128` 与 `ctx=256` 两套数据；
- 成功从 Phase 1 checkpoint 初始化兼容权重；
- 成功进入 `epoch 0` 训练。

最近一次监控到的日志关键信息如下：
- `ctx=128` 已完整跑完；
- `epoch 0 / ctx=128` 平均：
  - `total=19.7899`
  - `caption=1.4540`
  - `align=2.6279`
  - `event=28.4464`
  - `coverage=13.9936`
- `ctx=256` 已开始训练，早期观测：
  - `step 0/2803: loss=30.5938`
  - `step 140/2803: loss=29.1837`
  - `step 280/2803: loss=27.5217`

从目前趋势看：
- decoder 已经学起来，caption loss 明显下降；
- 长上下文下 event head 仍然是最难部分；
- 目前还没有出现新的 NaN / OOM / device assert。

随后，这轮训练在 `epoch 3` 后中断。当前已确认：
- 这不是正常 early stopping；
- 未发现 `2026-03-18 14:46 CST` 附近的 OOM kill 或显式 Python traceback；
- 更像是训练进程被外部终止；
- `GPU 2` 在 `2026-03-18 11:52:49 CST` 曾出现一次 `NVRM Xid 43`，说明存在潜在 GPU 不稳定风险。

目前已从
`<data_root>/runs/phase2/checkpoints/latest.pt`
成功恢复续训，日志显示：
- `Resumed from checkpoint ... (epoch 3)`
- `=== Starting training from epoch 4 ===`
- `epoch 4 / ctx=128` 早期观测：
  - `step 0/1421: loss=24.3665`
  - `step 71/1421: loss=21.8160`

这意味着当前项目状态应理解为：
- Phase 2 正在进行，而不是“准备开始”；
- 工程主风险已经从“代码框架能否跑通”转向“训练稳定性与验证指标能否持续改善”。

## 6. 如果论文要成立，接下来必须补的实验与论证

### 6.1 论文实验主线

必须至少做出下面这些对比：
- Template baseline
- Flat encoder + seq2seq
- Full model without hierarchy
- Full model without aligner
- Full model without event head
- Full model

### 6.2 论文必须回答的关键论证

1. **长上下文真的带来更难的问题吗？**
   - 需要比较 `ctx=128 / 256 / 512`。

2. **层次化结构是否有收益？**
   - 需要做去层次结构的消融。

3. **检索式对齐是否真的有帮助？**
   - 需要做 `no_align` 消融。

4. **caption 是否可验证？**
   - 需要把 unsupported claim、order consistency、span IoU 报出来。

5. **方法是否能跨数据集泛化？**
   - HARTH 的外部测试必须保留。

## 7. 当前主要风险

### 风险 1：event head 在长上下文下仍然较难

从当前 Phase 2 日志看，caption decoder 学得较快，但 event loss 仍高，说明：
- 事件标签建模比文本建模难；
- 长上下文下跨度和边界学习压力更大。

### 风险 2：Phase 3 还未正式接入 mixed caption source

虽然 paraphrase pipeline 已修通，但 mixed caption source 的正式训练闭环还未成为主线实验的一部分。

### 风险 3：评测体系仍需更论文化

当前基础评测已经能跑，但如果要投稿，还需要：
- 更稳定的 verification report；
- 更系统的 ablation；
- 更清晰的人类评测或案例分析。

## 8. 下一步工作建议

优先级建议如下：

1. **先盯完 Phase 2**
   - 拿到完整 epoch 和验证结果；
   - 判断 decoder 训练是否稳定。

2. **做最小可发表实验主表**
   - Template
   - Flat
   - Ours-no-hierarchy
   - Ours-full

3. **接 Phase 3**
   - 加入 paraphrase；
   - 扩展 `ctx=512`；
   - 观察是否真正提升文本自然度和跨长度鲁棒性。

4. **补论文材料**
   - 方法图；
   - 案例分析；
   - 消融表；
   - 评测定义图。

## 9. 给导师汇报时可以强调的点

汇报时最建议强调三句话：

1. **这个项目不是简单的 caption，而是在做“长上下文时序的层次化解释生成”。**
2. **它的创新点不只在生成文本，而在“可验证、可追溯、可检索对齐”。**
3. **目前已经完成数据、模板基线和 Phase 1，Phase 2 已经真实开始训练，后续的核心任务是把生成性能和事件 grounding 一起做强。**
