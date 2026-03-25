# 实验结果汇总

**日期**：2026-03-25

## M0：`phase2_bosfix` 跑通

### R001：BOS-only on CAPTURE-24 val
- `unique = 2587 / 12670`
- `top1_share = 0.1627`
- `activity_f1 = 0.6111`
- `dominant_accuracy = 0.5096`
- `unsupported_claim_rate = 0.1808`
- `event_span_iou = 0.0000`

### R002：Prompt diagnostic
- prompt 版本明显更差，不作为主结果。

### R003：旧 `phase2` 参考
- collapse 明显更重。

## M1：`full vs flat` on CAPTURE-24 test

### Full (`phase2_bosfix`)
- `activity_f1 = 0.6025`
- `dominant_accuracy = 0.4832`
- `unsupported_claim_rate = 0.1981`
- `verification_precision = 0.8019`
- `rouge_l = 0.3451`
- `event_span_iou = 0.0000`

### Flat (`phase2_flat`)
- `activity_f1 = 0.2637`
- `dominant_accuracy = 0.1932`
- `unsupported_claim_rate = 0.1357`
- `verification_precision = 0.8643`
- `rouge_l = 0.2331`
- `event_span_iou = 0.0000`

### 当前解释
- 在主语义指标上，full 明显优于 flat。
- 但两边都仍存在 decoder 文本污染，因此不能只看某一个 precision 类指标下结论。
- 当前保留下来的主结论是：hierarchy 方向为正。

## M2：`full vs noalign` on CAPTURE-24 test

### Noalign (`phase2_noalign`)
- `activity_f1 = 0.6240`
- `dominant_accuracy = 0.4602`
- `unsupported_claim_rate = 0.4222`
- `verification_precision = 0.5778`
- `rouge_l = 0.1525`
- `order_consistency = -0.1552`
- `event_span_iou = 0.0000`

### Full (`phase2_bosfix`)
- `activity_f1 = 0.5371`
- `dominant_accuracy = 0.5296`
- `unsupported_claim_rate = 0.4378`
- `verification_precision = 0.5622`
- `rouge_l = 0.1955`
- `order_consistency = 0.0972`
- `event_span_iou = 0.0000`

### 当前解释
- `noalign` 在 `activity_f1`、`activity_precision`、`verification_precision` 更高。
- `full` 在 `dominant_accuracy`、`order_consistency`、`rouge_l` 更高。
- 两边 `event_span_iou` 都还是 `0.0000`。这个比较不支持把 aligner 保留为核心贡献。

## 诊断级修复结果

### 推理侧脏文本治理
- constrained decoding 明显降低了 `$ / !!! / 引号串 / 乱码重复`。
- 但语序和句法仍未根修复。

### Evidence verbalization
- 在 16 条 `Capture-24 val` 子集上：
  - `event_evidence_span_iou = 0.0557`
  - textual `event_span_iou = 0.0520`
- 解释：event signal 不是完全没有；更大的问题是 event evidence 以前没有进入文本评测链路。
- 注意：这仍只是诊断级证据，不是主论文结果。
