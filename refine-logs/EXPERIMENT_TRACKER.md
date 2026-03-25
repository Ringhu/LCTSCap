# Experiment Tracker

这是实验台账。它只维护 run 状态、优先级和备注，不重复主线结论。日常入口看 `docs/DOC_INDEX.md`。

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | Close best-model loop | `phase2_bosfix` best, BOS-only decode | CAPTURE-24 val | Diversity stats, `activity_f1`, `dominant_accuracy`, `unsupported_claim_rate` | MUST | DONE | `unique=2587`, `top1_share=0.1627`, `activity_f1=0.6111`, `dominant_accuracy=0.5096`, `unsupported_claim_rate=0.1808`, `event_span_iou=0.0000`; collapse is clearly reduced but grounding is still absent |
| R002 | M0 | Prompt diagnostic | `phase2_bosfix` best, `"The participant"` prompt | CAPTURE-24 val | Same as R001 | NICE | DONE | `unique=534`, `top1_share=0.8931`, `activity_f1=0.5127`, `dominant_accuracy=0.3456`, `unsupported_claim_rate=0.2998`; prompt-text decode is worse and should not be used as the main result |
| R003 | M0 | Reference point | Old `phase2` best | CAPTURE-24 val | Existing results from 2026-03-19 | MUST | DONE | `activity_f1=0.4814`, `dominant_accuracy=0.3280`, `unsupported_claim_rate=0.2613`, severe collapse |
| R004 | M1 | Anchor baseline training | `phase2_flat` (`no_hierarchy`, `no_align`, `no_event`) | CAPTURE-24 train/val | Train loss, val factuality | MUST | DONE | 已于 `2026-03-21 00:12 CST` early stop 完成；最终 `best val_loss=0.5970`，最近完成 `epoch 21 val_loss=0.6130` |
| R005 | M1 | Main anchor comparison | Full vs `phase2_flat` | CAPTURE-24 test | Primary paper metrics | MUST | DONE | Full 在主语义指标上明显优于 flat：`activity_f1 0.6025 vs 0.2637`，`dominant_accuracy 0.4832 vs 0.1932`，`rouge_l 0.3451 vs 0.2331`；但两边都仍有脏格式污染，`event_span_iou` 仍为 `0.0` |
| R006 | M2 | Alignment ablation training | `phase2_noalign` | CAPTURE-24 train/val | Train loss, val factuality | MUST | DONE | 已于 `2026-03-22 03:00:38` 完成训练；最终日志为 `=== Training complete. Best val_loss=24.7823 best_score=0.1475 ===` |
| R007 | M2 | Alignment ablation eval | Full vs `phase2_noalign` | CAPTURE-24 test | Primary paper metrics | MUST | DONE | `full vs noalign` 已完成；`noalign` 在 `activity_f1 0.6240 vs 0.5371` 更高，`full` 在 `dominant_accuracy 0.5296 vs 0.4602`、`rouge_l 0.1955 vs 0.1525` 更高；两边 `event_span_iou` 都是 `0.0000`，不支持把 aligner 保留为核心贡献 |
| R008 | M3 | External full-model check | Full hierarchical recipe | HARTH train/test | Primary paper metrics | MUST | TODO | Run only after CAPTURE-24 anchor result is positive |
| R009 | M3 | External flat baseline | `phase2_flat` equivalent on HARTH | HARTH train/test | Primary paper metrics | MUST | BLOCKED | Blocked on R008 setup |
| R010 | M4 | Appendix-only support | Auxiliary alignment benchmarks | SHL / Sleep-EDF / UCR | Retrieval metrics | NICE | DONE | UCR and Sleep-EDF alignment-only evaluation completed on 2026-03-20; these support cross-domain alignment only and do not close the main caption thesis gap |
