# Refine Logs 实时状态

这是自动生成的最新状态页。它只用于快速看最近状态，不承担长期事实主文档职责。长期事实看 `PROGRESS.md`，文档入口看 `docs/DOC_INDEX.md`。

- 生成时间：`2026-03-25 00:48:46`
- 触发来源：`post-doc-sync:post-r007-doc-sync`

## 当前主线

- `hierarchical long-context time-series captioning with grounding-aware evaluation`

## Refine 文档快照

- 最新 refine 更新时间：`2026-03-25 00:43:42`
- `EXPERIMENT_RESULTS.md`：`2026-03-25 00:43:42`
- `EXPERIMENT_TRACKER.md`：`2026-03-25 00:37:26`
- `REVIEW_SUMMARY.md`：`2026-03-25 00:43:42`

## 实验状态

- `R001` `DONE`：Close best-model loop | `phase2_bosfix` best, BOS-only decode | `unique=2587`, `top1_share=0.1627`, `activity_f1=0.6111`, `dominant_accuracy=0.5096`, `unsupported_claim_rate=0.1808`, `event_span_iou=0.0000`; collapse is clearly reduced but grounding is still absent
- `R002` `DONE`：Prompt diagnostic | `phase2_bosfix` best, `"The participant"` prompt | `unique=534`, `top1_share=0.8931`, `activity_f1=0.5127`, `dominant_accuracy=0.3456`, `unsupported_claim_rate=0.2998`; prompt-text decode is worse and should not be used as the main result
- `R003` `DONE`：Reference point | Old `phase2` best | `activity_f1=0.4814`, `dominant_accuracy=0.3280`, `unsupported_claim_rate=0.2613`, severe collapse
- `R004` `DONE`：Anchor baseline training | `phase2_flat` (`no_hierarchy`, `no_align`, `no_event`) | 已于 `2026-03-21 00:12 CST` early stop 完成；最终 `best val_loss=0.5970`，最近完成 `epoch 21 val_loss=0.6130`
- `R005` `DONE`：Main anchor comparison | Full vs `phase2_flat` | Full 在主语义指标上明显优于 flat：`activity_f1 0.6025 vs 0.2637`，`dominant_accuracy 0.4832 vs 0.1932`，`rouge_l 0.3451 vs 0.2331`；但两边都仍有脏格式污染，`event_span_iou` 仍为 `0.0`
- `R006` `DONE`：Alignment ablation training | `phase2_noalign` | 已于 `2026-03-22 03:00:38` 完成训练；最终日志为 `=== Training complete. Best val_loss=24.7823 best_score=0.1475 ===`
- `R007` `DONE`：Alignment ablation eval | Full vs `phase2_noalign` | `full vs noalign` 已完成；`noalign` 在 `activity_f1 0.6240 vs 0.5371` 更高，`full` 在 `dominant_accuracy 0.5296 vs 0.4602`、`rouge_l 0.1955 vs 0.1525` 更高；两边 `event_span_iou` 都是 `0.0000`，不支持把 aligner 保留为核心贡献
- `R008` `TODO`：External full-model check | Full hierarchical recipe | Run only after CAPTURE-24 anchor result is positive
- `R009` `BLOCKED`：External flat baseline | `phase2_flat` equivalent on HARTH | Blocked on R008 setup
- `R010` `DONE`：Appendix-only support | Auxiliary alignment benchmarks | UCR and Sleep-EDF alignment-only evaluation completed on 2026-03-20; these support cross-domain alignment only and do not close the main caption thesis gap

## 已验证产物

- `phase2_bosfix` best checkpoint：`epoch=5`，`loss=26.58820429221023`，更新时间 `2026-03-19 17:50:30`
- `phase2_flat` 日志：`complete`，更新时间 `2026-03-21 00:12:12`
- `phase2_flat` 最后一行：`2026-03-21 00:12:12,219 [INFO] train: === Training complete. Best val_loss=0.5970 best_score=-0.5970 ===`
- `phase2_noalign` 日志：`complete`，更新时间 `2026-03-22 03:00:38`
- `phase2_noalign` 最后一行：`2026-03-22 03:00:38,244 [INFO] train: === Training complete. Best val_loss=24.7823 best_score=0.1475 ===`
- `Sleep-EDF` 原始下载：`306` 个 EDF，`7.07 GB`，最新 `SC4822GC-Hypnogram.edf`，`robots.txt=True`
- `SHL` staging 路径：`/cluster1/user1/lctscap_data/auxiliary/raw/shl`

## 文档同步检查

- `README.md`：`current`
- `PROGRESS.md`：`current`
- `SPEC.md`：`current`
- `RESEARCH_BRIEF.md`：`current`
- `docs/DOC_INDEX.md`：`current`

## 当前下一步

- 当前没有待启动实验；应整理结果并准备下一轮 review。
