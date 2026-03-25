# 技术规格

<!-- AUTO_SYNC_STATUS:START -->
## 自动同步状态

- 同步时间：`2026-03-25 00:48:44`
- 当前主线：`hierarchical long-context time-series captioning with grounding-aware evaluation`
- 当前下一步：当前没有待启动实验；应整理结果并准备下一轮 review。
- 详细入口：`docs/DOC_INDEX.md`
- 最新状态页：`refine-logs/LATEST_STATUS.md`
<!-- AUTO_SYNC_STATUS:END -->

当前怎么实现以本文件为准。文档入口看 `docs/DOC_INDEX.md`。

## 1. 任务定义

输入是长时间序列上下文。当前主设置是 `128 / 256` 个语义 token。每个 token 对应一个 `10` 秒窗口。输出是 `1-2` 句 factual caption，并用 grounding-aware 指标评测。

当前主论文只承诺：
- 全局 caption
- grounding-aware 评测

当前不承诺：
- 强 sentence-level verifiable captioning
- LLM polished narration
- `ctx=512` 作为主结果
- auxiliary benchmark 主表替代 caption 主结果

## 2. 当前主链路

```text
LocalEncoder
  -> ChannelFusion
  -> HierarchicalPlanner
  -> EventProposalHead
  -> RetrievalAligner
  -> CaptionDecoder
```

说明：
- 当前 decoder 主要 cross-attend `H_seg`。
- `EventProposalHead` 现在还是辅助监督分支。
- event proposal 还没有稳定进入 decoder 主生成链路。
- `LLMBridge` 不是当前主链路能力。

## 3. 数据规格

### CAPTURE-24
- 151 名参与者
- 100Hz 下采样到 50Hz
- 3 通道 wrist accelerometer

### HARTH
- 22 名受试者
- 50Hz
- 6 通道（背部 + 大腿）

### 长上下文样本
- token 长度：`128 / 256 / 512`
- 对应真实时长：约 `21.3 / 42.7 / 85.3` 分钟
- split：participant / subject 级别切分，禁止泄漏

## 4. 训练阶段

### Phase 1
- 训练：encoder + planner + event head + aligner
- 不训 decoder

### Phase 2 主线
- `phase2_bosfix`：当前 full model 基线
- `phase2_flat`：`no_hierarchy + no_align + no_event`
- `phase2_noalign`：保留 hierarchy，但去掉 aligner 作用

### 暂缓阶段
- `Phase 3` paraphrase
- `ctx=512`
- `Phase 4 / LLMBridge`
- human evaluation
- auxiliary benchmark 主表化

## 5. 实验合同与 claim map

### 当前主 claim
- C1：hierarchy 让长上下文 captioning 真正可做。
- C2：aligner 可能有帮助，但不是主故事。
- A1：`phase2_bosfix` 的改进不是 prompt hack，也不只是 collapse 假象。

### 当前最低证据要求
- C1：full 在 `CAPTURE-24` 上优于 flat，并且后续在 HARTH 上至少保持同方向。
- C2：full 稳定优于 `noalign`；否则 aligner 降级为训练细节或支撑性模块。
- A1：`phase2_bosfix` 在多样性和 factuality 上明显优于旧 `phase2`。

### 当前阶段 gate
- M0：证明 full model 可用。已通过。
- M1：证明 hierarchy 胜过 flat。已通过。
- M2：判断 aligner 的真实地位。当前就在这里。
- M3：HARTH 外部检查。只在 M2 结论稳定后进入。
- M4：appendix 或补充支撑。只有主线稳定后再看。

## 6. 当前评测口径

### 主评测
- `activity_f1`
- `dominant_accuracy`
- `unsupported_claim_rate`
- `verification_precision`
- `transition_accuracy`
- `event_span_iou`

### lexical 辅助指标
- `rouge_l`
- 其他文本重叠指标

### 当前证据侧扩展
- 可选输出 `predicted_events`
- 可选 `evidence_text_mode`
- 诊断级指标：
  - `event_evidence_span_iou`
  - `event_evidence_precision`
  - `event_evidence_unsupported_rate`

说明：
- 主判断优先看 factuality 和 grounding-aware 指标。
- lexical 指标只做辅助说明，不能单独决定方法去留。
- `event_evidence_*` 与 `evidence_text_mode` 目前只算诊断增强。
- 它们证明 event signal 不是完全没有，但不等于强可验证 caption 已经实现。

## 7. R007 比较口径

`R007` 只回答一个问题：aligner 是否还值得留在贡献列表里。

### 比较对象
- Full：`configs/train/phase2_bosfix.yaml` + `phase2_bosfix` best checkpoint
- NoAlign：`configs/train/phase2_noalign.yaml` + `phase2_noalign` best checkpoint

### 生成约束
- 两边必须同口径生成，只换模型，不换解码策略。
- 固定：
  - `--split test`
  - `--restrict_to_caption_vocab`
  - `--caption_vocab_split train`
  - `--repetition_penalty 1.1`
  - `--no_repeat_ngram_size 3`
- 默认不要开：
  - `--emit_event_evidence`
  - `evidence_text_mode != none`

### 判定规则
- full 明显优于 `noalign`：aligner 保留为支撑性贡献。
- full 和 `noalign` 很接近：aligner 降级，不再写成主贡献。
- `noalign` 更好：把 aligner 从主贡献里拿掉。

## 8. 当前技术边界

- 当前正确说法是 `grounding-aware evaluation`，不是强 `verifiable captioning`。
- 原因很直接：主结果里的 `event_span_iou` 仍偏低，decoder 还没有稳定显式消费 event evidence。
- constrained decoding 现在只是止血层。它能压住脏文本，不能根修复句法和语序。
- evidence verbalization 现在只是诊断层。只有当它持续给出非零信号，才值得继续做训练侧绑定。
- 当前 auxiliary alignment 结果只支撑 alignment generalization，不能代替 caption 主结果。
- 当前主线在 `128 / 256` caption 上还没完全整理好，所以不能提前回到 `Phase 4`。

## 9. 当前已验证约束

- 默认 GPU：`GPU 2`
- 当前主线不允许提前扩展到：
  - `Phase 3`
  - `ctx=512`
  - `Phase 4 / LLMBridge`
  - 大规模 auxiliary benchmark 主表
  - human evaluation

## 10. 维护规范

- 任何改动训练流程、评测口径、phase 配置、数据 schema，都必须同步更新本文件。
- 本文件记录的是当前执行合同，不是历史聊天记录。
