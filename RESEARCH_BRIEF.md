# 研究简报

<!-- AUTO_SYNC_STATUS:START -->
## 自动同步状态

- 同步时间：`2026-03-25 00:48:44`
- 当前主线：`hierarchical long-context time-series captioning with grounding-aware evaluation`
- 当前下一步：当前没有待启动实验；应整理结果并准备下一轮 review。
- 详细入口：`docs/DOC_INDEX.md`
- 最新状态页：`refine-logs/LATEST_STATUS.md`
<!-- AUTO_SYNC_STATUS:END -->

导师汇报或论文写作时看这页。日常入口看 `docs/DOC_INDEX.md`。

## 当前唯一说法

> hierarchical long-context time-series captioning with grounding-aware evaluation

这句话现在可以讲。其他方向现在都以后再做。

## 当前已经站住的结论

- 长上下文 captioning 现在不是完全做不成。证据是 `phase2_bosfix` 已把训练、生成、评测都跑起来了。
- hierarchy 现在是主结果里最稳的一部分。证据是 `R005 full vs flat` 在 `CAPTURE-24 test` 上达到 `activity_f1 0.6025 vs 0.2637`、`dominant_accuracy 0.4832 vs 0.1932`、`rouge_l 0.3451 vs 0.2331`。
- grounding-aware evaluation 现在是有价值的。证据是它把主语义指标和 `event_span_iou 0.0000` 的差距直接暴露出来了。
- aligner 现在不能再写成核心贡献。证据是 `R007 full vs noalign` 里，`noalign` 在 `activity_f1 0.6240 vs 0.5371`、`verification_precision 0.5778 vs 0.5622` 更高，`full` 只在 `dominant_accuracy 0.5296 vs 0.4602`、`rouge_l 0.1955 vs 0.1525` 更高，而且两边 `event_span_iou` 都是 `0.0000`。

## 当前还不能讲大的地方

- 现在不能说强 `verifiable captioning` 已成立。原因是主结果里的 `event_span_iou` 还太低。
- 现在不能把 aligner 写成核心贡献。原因是 `R007` 已经不支持这个结论。
- 现在不能把跨数据集结论写稳。原因是 `HARTH` 还只是后续外部检查。
- 现在不能把 `Phase 3`、`ctx=512`、`Phase 4 / LLMBridge`、大量 auxiliary 主表、`human evaluation` 写成当前阶段。原因是主结果还没整理好。

## 当前最关键的问题

- 当前最关键的问题是 decoder 文字还不稳。受限解码能压住脏符号和重复，但修不好句法和语序。
- 当前第二个问题是 event 证据还没稳定进入主输出。诊断级 evidence verbalization 已给出非零信号，但它还不是主结果。
- 当前第三个问题是把论文说法改到和结果一致。`R007` 之后，主结论要以 hierarchy 为主，aligner 最多只保留为支撑模块。

## 导师汇报时最稳的说法

- 这条线现在最稳的结论是：hierarchy 让长上下文时序 captioning 更可做，grounding-aware evaluation 把系统边界测出来了。
- 这条线现在最该强调的证据是 `R005` 的主语义增益，不是 lexical 小修小补。
- 这条线现在最该坦白的问题是 `event_span_iou` 仍是 `0.0000`，强可验证结论还没成立。
- 这条线现在最重要的下一步不是继续扩方法，而是先把 `R007` 之后的说法改对；如果继续做外部检查，优先考虑不带 aligner 的更简版本。

## R007 之后怎么写 aligner

- 如果后续主结果还是以当前数字为准，就只保留 hierarchy 作为核心结果。aligner 只写成训练细节或支撑模块。
- 如果以后继续做外部检查，优先带不带 aligner 的更简版本去看 `HARTH`。这样更符合当前结果，也更省成本。
- 如果以后真要把 aligner 重新抬回核心贡献，必须先拿到新的主结果证据，而不是继续沿用当前 `R007` 的数字。

## 当前明确以后再做的内容

- `Phase 3` paraphrase
- `ctx=512`
- `Phase 4 / LLMBridge`
- auxiliary benchmark 主表化
- human evaluation
