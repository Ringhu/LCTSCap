# 项目进度

<!-- AUTO_SYNC_STATUS:START -->
## 自动同步状态

- 同步时间：`2026-03-25 00:48:44`
- 当前主线：`hierarchical long-context time-series captioning with grounding-aware evaluation`
- 当前下一步：当前没有待启动实验；应整理结果并准备下一轮 review。
- 详细入口：`docs/DOC_INDEX.md`
- 最新状态页：`refine-logs/LATEST_STATUS.md`
<!-- AUTO_SYNC_STATUS:END -->

当前长期情况只看这页。其他文档不再重复维护阶段判断。

> hierarchical long-context time-series captioning with grounding-aware evaluation

## 当前判断

- 当前唯一主线是 `hierarchical long-context time-series captioning with grounding-aware evaluation`。`Phase 3`、`ctx=512`、`Phase 4 / LLMBridge`、大量 auxiliary 主表、human evaluation 现在都不进主链路。
- 当前真实阶段是 late `Phase 2 / M2` 后的结果整理。`R004 phase2_flat`、`R005 full vs flat`、`R006 phase2_noalign`、`R007 full vs noalign` 已完成。
- `R007` 已经给出结论。`noalign` 在 `activity_f1 0.6240 vs 0.5371`、`verification_precision 0.5778 vs 0.5622` 更高，`full` 在 `dominant_accuracy 0.5296 vs 0.4602`、`rouge_l 0.1955 vs 0.1525` 更高；两边 `event_span_iou` 都是 `0.0000`，不支持把 aligner 保留为核心贡献。

## 已验证事实

### Phase 0

- `CAPTURE-24` 和 `HARTH` 的预处理、模板标注、模板基线评测已经跑完。Phase 0 只作为上界参考，不作为当前主结果。

### Phase 1

- `phase1` 已完成。最优 checkpoint 在 `/cluster1/user1/lctscap_data/runs/phase1/checkpoints/best.pt`。
- Phase 1 说明 `LocalEncoder -> ChannelFusion -> HierarchicalPlanner -> EventProposalHead -> RetrievalAligner` 这段训练链路能稳定工作。它还不说明最终 caption 质量已经够用。

### Phase 2 主线

- 原始 `phase2` 已确认出现 mode collapse。`phase2_bosfix` 已把长上下文 caption 的训练、生成、评测都跑起来。
- `R004 phase2_flat` 已完成。它是 hierarchy 的主比较基线，`best val_loss=0.5970`。
- `R005 full vs flat` 已完成。`Full` 在 `CAPTURE-24 test` 上高于 `flat`：`activity_f1 0.6025 vs 0.2637`、`dominant_accuracy 0.4832 vs 0.1932`、`rouge_l 0.3451 vs 0.2331`。
- `R005 full vs flat` 也把边界暴露出来了。两边的 `event_span_iou` 都还是 `0.0000`，而且 decoder 文字仍有脏格式和语序问题。
- `R006 phase2_noalign` 训练已完成。最终训练日志记录的是 `Best val_loss=24.7823 best_score=0.1475`。
- `R007 full vs noalign` 已完成。结果是 aligner 没有给出足够稳定的主结果增益，因此后续主说法应以 hierarchy 为主，不再把 aligner 写成核心贡献。

## 诊断结果

- 受限解码已经有用。`--restrict_to_caption_vocab`、`--repetition_penalty`、`--no_repeat_ngram_size` 能明显压住 `$ / !!! / 引号串 / 乱码重复` 一类污染。
- 受限解码还不够。它能清掉脏格式，但修不好句法和语序。
- evidence verbalization 说明 event signal 不是零。在 `Capture-24 val` 的 16 条子集上，textual `event_span_iou` 从 `0.0` 提到 `0.0520`。
- evidence verbalization 还不是主结果。它只说明 event 信息开始进入文本评测链路，不说明 caption grounding 已经成立。

## 当前已经成立的说法

- 长上下文 captioning 在当前设置下不是完全做不成。`phase2_bosfix` 已把训练、生成、评测都跑起来了。
- hierarchy 在 CAPTURE-24 上是正的。`R005` 已经说明 full 明显强于 flat。
- grounding-aware evaluation 有价值。它把主语义指标和 `event_span_iou` 的差距直接暴露出来了。

## 当前还不能讲大的地方

- 现在不能说强 verifiable captioning 已经成立。主结果里的 `event_span_iou` 还太低。
- 现在不能说 aligner 还是核心贡献。`R007` 已经说明这点站不住。
- 现在不能说跨数据集主结论已经站住。`HARTH` 只能在 CAPTURE-24 主结果说法整理好之后再看。
- 现在不能把 `Phase 3`、`ctx=512`、`Phase 4 / LLMBridge` 写成当前阶段。

## 当前主要风险

- `event_span_iou` 主结果太低。grounding-aware 现在只能说明评测协议有用，不能说明文本 grounding 已经稳了。
- decoder 文字质量还不够稳。语序和句法问题会继续压低 factuality 与 grounding-aware 指标。
- `R007` 之后主说法必须变窄。如果继续把 aligner 当核心贡献，结论会和结果不一致。

## 下一步

- 第一步先按 `R007` 结果改说法。主结论保留 hierarchy、long-context captioning 和 grounding-aware evaluation，不再把 aligner 写成核心贡献。
- 第二步如果继续做外部检查，优先考虑不带 aligner 的更简版本，再看 `HARTH`。
- 第三步继续压 decoder 文字问题和 event 进入文本的路径。到这一步之前，不回到 `Phase 3 / ctx=512 / Phase 4 / 大量 auxiliary 主表`。
