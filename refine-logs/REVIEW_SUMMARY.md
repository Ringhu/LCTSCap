# 长审查与决策记录

这页只保留长判断和决策依据。日常入口先看 `../docs/DOC_INDEX.md`，长期情况先看 `../PROGRESS.md`。

**日期**：2026-03-25

## 当前结论

- 当前唯一主线还是 `hierarchical long-context time-series captioning with grounding-aware evaluation`。这条线已经够满，再加新说法只会稀释主结果。
- 当前真实阶段是 late `Phase 2 / M2` 后的结果整理。`R004 phase2_flat`、`R005 full vs flat`、`R006 phase2_noalign`、`R007 full vs noalign` 已完成。
- `R007` 已经回答了 aligner 有没有必要留在核心贡献里。当前答案是不支持。

## 为什么现在只保留这一条主线

- 当前仓库里真正站住的事实有四条。`phase2_bosfix` 说明长上下文 captioning 不再完全塌缩，`R005 full vs flat` 说明 hierarchy 在主语义指标上是正的，`R007 full vs noalign` 说明 aligner 没有给出足够稳定的核心增益，grounding-aware evaluation 说明系统边界能被量化地看出来。
- 当前仓库里还没有更硬的事实去支撑更大的说法。`event_span_iou` 主结果仍是 `0.0000`，所以强 verifiable captioning 还不能讲大。
- 当前主线必须保持最小。只保留 hierarchy、long-context captioning、grounding-aware evaluation，才能把已经跑出来的结果说清楚。

## 已经站住的部分

- `phase2_bosfix` 已经把训练、生成、评测都跑起来了。它说明项目已经离开“只能跑模板或早期表征”的阶段。
- `R005 full vs flat` 已经给出主数字。`activity_f1 0.6025 vs 0.2637`、`dominant_accuracy 0.4832 vs 0.1932`、`rouge_l 0.3451 vs 0.2331` 说明 hierarchy 方向现在是正的。
- `R007 full vs noalign` 已经给出 aligner 判断。`noalign` 在 `activity_f1 0.6240 vs 0.5371`、`verification_precision 0.5778 vs 0.5622` 更高，`full` 在 `dominant_accuracy 0.5296 vs 0.4602`、`rouge_l 0.1955 vs 0.1525` 更高；两边 `event_span_iou` 都是 `0.0000`，所以 aligner 不能继续当核心贡献。
- 推理侧清洗已经有价值。受限词表、重复惩罚、`no_repeat_ngram_size` 能明显减少脏符号和重复模式。
- 诊断级 evidence verbalization 已经给出弱信号。在 16 条 `Capture-24 val` 子集上，textual `event_span_iou` 从 `0.0` 提到 `0.0520`。

## 还不能继续放大的说法

- 现在不能把项目写成强 verifiable captioning。主结果里的 `event_span_iou` 还太低，而且 event 信息还没有稳稳进入 caption 主输出。
- 现在不能把 aligner 写成核心贡献。`R007` 已经不支持这个结论。
- 现在不能把 cross-domain auxiliary alignment 写成主表证据。那组实验最多只说明匹配能力，不说明 caption 主结果已经够硬。
- 现在不能把 `Phase 3`、`ctx=512`、`Phase 4 / LLMBridge`、human evaluation 写成当前阶段。它们现在都放后面，等主结果先整理好。
- 现在不能把 `event_evidence_*` 的诊断结果混成主论文结果。它们目前只适合解释问题出在哪，不适合替代主比较。

## R007 回答了什么

- `R007` 只比较 full 和 `noalign`。比较时没有顺手换解码策略、诊断开关或证据文本模式。
- `R007` 的结果不是“full 全面更好”，也不是“noalign 全面更好”。它给出的事实是：aligner 没有带来足够稳定、足够干净的主结果优势。
- `R007` 之后最稳的处理方式是把 aligner 降成支撑模块，主结论回到 hierarchy、long-context captioning 和 grounding-aware evaluation。

## R007 之后怎么改说法

- 主结论只保留 hierarchy 让长上下文 captioning 更可做，以及 grounding-aware evaluation 把当前边界测出来了。
- aligner 只保留为训练细节或支撑模块。至少在当前结果下，不能继续把它放进核心贡献列表。
- 如果后面还要做外部检查，优先带不带 aligner 的更简版本去看 `HARTH`。这样更符合 `R007` 的结果，也更省成本。

## HARTH 为什么还放在后面

- HARTH 现在只能做外部检查。它不能替代 CAPTURE-24 上已经做出来的主比较。
- 现在先把 `R007` 之后的说法改对，再决定是否用更简版本去做外部检查，成本更低。

## 如果主线继续受阻，还有哪些后路

- Evidence-first captioning 是第一条后路。它适合 hierarchy 不是唯一主因，但 event 信息开始变成非零的时候。
- Retrieval-grounded captioning 是第二条后路。它适合 caption 文本仍不稳，但检索侧更可靠的时候。
- Benchmark + evaluation paper 是第三条后路。它适合方法增益不够硬，但 long-context 数据和 grounding-aware 评测本身仍有价值的时候。
- LLM realizer 是最后一条后路。只有 event 结构已经站住、只剩文字表达差时才考虑它。

## 为什么这次文档这样分工

- `../docs/DOC_INDEX.md` 只做入口最省时间。读者先知道去哪里看，不需要先读一篇长情况说明。
- `../PROGRESS.md` 只做长期情况最稳。当前阶段、风险、下一步如果分散在旧的新手页、旧的一页摘要或临时计划页里，很快就会互相打架。
- 这页只保留长判断最合适。为什么主线只能保留这一条、哪些说法不能再放大、`R007` 之后怎么改口径，都不该塞回入口页或长期情况页。
- `../docs/plans/*.md` 只留短期计划最省维护。命令、输出路径、当天安排过期很快，长期规则迁回主文档就够了。

## 旧文档现在怎么用

- 旧的 proposal、refinement、gap、playbook 汇总页都退出主入口了。历史判断统一回收到这页和 `../PROGRESS.md`。
- 当前 `R007` 的临时执行页只保留短期执行信息。里面仍长期成立的规则，已经迁到这页和 `../PROGRESS.md`。
