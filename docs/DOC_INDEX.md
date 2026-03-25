# 文档索引

<!-- AUTO_SYNC_STATUS:START -->
## 自动同步状态

- 同步时间：`2026-03-25 00:48:44`
- 触发来源：`post-r007-doc-sync`
- 当前主线：`hierarchical long-context time-series captioning with grounding-aware evaluation`

### 核心实验状态

- `R004` `DONE`：Anchor baseline training；已于 `2026-03-21 00:12 CST` early stop 完成；最终 `best val_loss=0.5970`，最近完成 `epoch 21 val_loss=0.6130`
- `R005` `DONE`：Main anchor comparison；Full 在主语义指标上明显优于 flat：`activity_f1 0.6025 vs 0.2637`，`dominant_accuracy 0.4832 vs 0.1932`，`rouge_l 0.3451 vs 0.2331`；但两边都仍有脏格式污染，`event_span_iou` 仍为 `0.0`
- `R006` `DONE`：Alignment ablation training；已于 `2026-03-22 03:00:38` 完成训练；最终日志为 `=== Training complete. Best val_loss=24.7823 best_score=0.1475 ===`
- `R007` `DONE`：Alignment ablation eval；`full vs noalign` 已完成；`noalign` 在 `activity_f1 0.6240 vs 0.5371` 更高，`full` 在 `dominant_accuracy 0.5296 vs 0.4602`、`rouge_l 0.1955 vs 0.1525` 更高；两边 `event_span_iou` 都是 `0.0000`，不支持把 aligner 保留为核心贡献

### 当前主要风险

- `event_span_iou` 主结果仍偏低，强可验证结论还没成立。
- decoder 文本污染已部分缓解，但语序和句法错误仍存在。

### 下一步

- 当前没有待启动实验；应整理结果并准备下一轮 review。
<!-- AUTO_SYNC_STATUS:END -->

当前入口只看这页和 4 个主文档。长期情况看 `../PROGRESS.md`，实现边界看 `../SPEC.md`，命令看 `../README.md`，导师汇报和论文说法看 `../RESEARCH_BRIEF.md`。

## 先看什么

- 新接手仓库先看 `../PROGRESS.md`。这页单独说明当前主线、阶段、风险和下一步。
- 需要理解模型、phase 和评测边界时看 `../SPEC.md`。合同级说明只放这里。
- 需要装环境、复制命令、查脚本入口时看 `../README.md`。可执行命令只放这里。
- 需要准备导师汇报或论文摘要时看 `../RESEARCH_BRIEF.md`。这页只保留当前能讲的说法。

## 什么时候看什么

- 需要知道 run 编号、排队顺序、完成情况时看 `../refine-logs/EXPERIMENT_TRACKER.md`。它只管实验台账。
- 需要查主结果数字时看 `../refine-logs/EXPERIMENT_RESULTS.md`。它只管结果和指标。
- 需要看长判断、决策原因、口径边界时看 `../refine-logs/REVIEW_SUMMARY.md`。这页不是日常入口。
- 需要快速回看当天摘要时看 `../refine-logs/LATEST_STATUS.md`。长期情况仍以 `../PROGRESS.md` 为准。

## 低频文档

- 新人补背景直接看 `../PROGRESS.md`、`../README.md` 和这页。仓库不再保留单独的新手状态页。
- 旧的 proposal、gap、refinement、playbook 汇总页不再作为入口。需要追溯历史判断时，统一看 `../refine-logs/REVIEW_SUMMARY.md`。
- `../AGENTS.md` 和 `../CLAUDE.md` 只在协作约定或项目指令变动时看。它们不是人工日常入口。

## 自动生成页

- `../refine-logs/LATEST_STATUS.md` 只看当天摘要。长期事实不要写回这页。
- 根文档里的 `AUTO_SYNC_STATUS` 区块只看同步摘要。长期判断仍以正文为准。

## 临时计划文档

- `plans/` 只放短期执行页。事情做完后，只把仍长期成立的规则迁回主文档。
- 当前 `R007` 的临时执行页只负责短期命令、输出路径和当天安排。这类信息过期很快，不再写回长期状态页。
