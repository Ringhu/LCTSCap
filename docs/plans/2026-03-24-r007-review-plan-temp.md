# 2026-03-24 R007 审查与下一步实验计划（临时）

## 审查结论

当前不该扩方法，下一步就是 `R007 full vs noalign`。原因很直接：`R005` 已经证明 hierarchy 是正的，但 `R006` 之后，aligner 到底算有效贡献还是只算次要细节，还没用主指标定下来。

当前阶段判断是 **late Phase 2 / M2**。

## 证据

### 1. 主线没有变
- 当前唯一研究问题仍是 `hierarchical long-context time-series captioning with grounding-aware evaluation`。
- 模型主链路仍是 `LocalEncoder -> ChannelFusion -> HierarchicalPlanner -> EventProposalHead -> RetrievalAligner -> CaptionDecoder`。
- 当前没有证据说明项目已经切到 `Phase 3`、`ctx=512`、`Phase 4 / LLMBridge`、human evaluation 或 auxiliary benchmark 主表。

### 2. hierarchy 已经站住了
- `R005 full vs flat` 已完成。
- `Full` 在 CAPTURE-24 test 上明显优于 `flat`：
  - `activity_f1 = 0.6025 vs 0.2637`
  - `dominant_accuracy = 0.4832 vs 0.1932`
  - `rouge_l = 0.3451 vs 0.2331`
- 这说明 hierarchy 方向当前是正的，不需要再回头证明一次。

### 3. grounding 还没站住
- `event_span_iou` 在当前主结果里仍是 `0.0000`。
- evidence verbalization 只在小子集上给出过弱信号：
  - `event_evidence_span_iou = 0.0557`
  - textual `event_span_iou = 0.0520`
- 这只能说明 event signal 不是完全没有，不能说明 caption grounding 已经成立。

### 4. 当前真正没定下来的，是 aligner 的地位
- `R006 phase2_noalign` 训练已完成。
- `R007` 在 tracker 里已经是 `READY`。
- 所以下一步不是再训新分支，而是把 `full` 和 `phase2_noalign` 用同一套生成与评测口径做主指标比较。

### 5. HARTH 现在还不能先跑
- HARTH 的作用是外部验证，不是替代当前 CAPTURE-24 上的主结论。
- 如果 `R007` 之后发现 full 和 noalign 差不多，论文里就不该继续把 aligner 当主贡献，此时先跑 HARTH 的收益不高。

## 下一步实验计划

## Plan A：先完成 `R007 full vs noalign`

这是唯一应该先做的实验。

### 实验目标
用 CAPTURE-24 test 上的主指标回答：
1. full 是否稳定优于 `phase2_noalign`
2. aligner 是否值得继续保留在论文核心贡献里

### 比较对象
- Full：`configs/train/phase2_bosfix.yaml` + `/cluster1/user1/lctscap_data/runs/phase2_bosfix/checkpoints/best.pt`
- NoAlign：`configs/train/phase2_noalign.yaml` + `/cluster1/user1/lctscap_data/runs/phase2_noalign/checkpoints/best.pt`

### 生成口径
两边必须完全同口径，只改模型，不改解码策略。

建议固定以下 decode 设置：
- `--split test`
- `--restrict_to_caption_vocab`
- `--caption_vocab_split train`
- `--repetition_penalty 1.1`
- `--no_repeat_ngram_size 3`
- 先不要开 `--emit_event_evidence`
- `--evidence_text_mode` 保持默认 `none`

原因：
- 这套设置已经被仓库用作推理侧文本清洗。
- `event_evidence_*` 目前还是诊断能力，不该混进这次主比较。

### 主指标
优先看这 5 个：
- `activity_f1`
- `dominant_accuracy`
- `unsupported_claim_rate`
- `verification_precision`
- `transition_accuracy`

次级再看：
- `rouge_l`
- `event_span_iou`

解释规则：
- 主判断以 factuality / grounding-aware 指标为主。
- lexical 指标只能辅助解释，不能单独决定结论。

## 决策规则

### 情况 1：full 明显优于 noalign
保留 aligner，但只把它写成 **supporting contribution** 更稳。

建议阈值：
- `activity_f1` 和 `dominant_accuracy` 至少有一项稳定更高；
- `unsupported_claim_rate` 不明显变差；
- `verification_precision` 不明显变差。

这时下一步才是 `R008 HARTH full-model check`。

### 情况 2：full 和 noalign 很接近
把 aligner 降为训练细节或 supporting detail，不再当核心贡献讲。

这时论文核心只保留：
- hierarchy 让长上下文 captioning 可做
- grounding-aware evaluation 暴露了当前系统的边界

这时也可以进 HARTH，但目标要改成验证 hierarchy 主结论，而不是验证 aligner。

### 情况 3：noalign 反而更好
直接把 aligner 从主贡献里拿掉。

这时下一步应该是：
- 更新实验计划与文档口径
- 只保留 hierarchy 主结论
- HARTH 只验证不带 aligner 的更简版本

## 具体执行命令

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/generate_predictions.py \
  --config configs/train/phase2_bosfix.yaml \
  --checkpoint /cluster1/user1/lctscap_data/runs/phase2_bosfix/checkpoints/best.pt \
  --data_root /cluster1/user1/lctscap_data \
  --split test \
  --output_path outputs/r007/full_test_predictions.jsonl \
  --restrict_to_caption_vocab \
  --caption_vocab_split train \
  --repetition_penalty 1.1 \
  --no_repeat_ngram_size 3

CUDA_VISIBLE_DEVICES=2 python scripts/generate_predictions.py \
  --config configs/train/phase2_noalign.yaml \
  --checkpoint /cluster1/user1/lctscap_data/runs/phase2_noalign/checkpoints/best.pt \
  --data_root /cluster1/user1/lctscap_data \
  --split test \
  --output_path outputs/r007/noalign_test_predictions.jsonl \
  --restrict_to_caption_vocab \
  --caption_vocab_split train \
  --repetition_penalty 1.1 \
  --no_repeat_ngram_size 3

python scripts/evaluate.py \
  --predictions_path outputs/r007/full_test_predictions.jsonl \
  --gold_path /cluster1/user1/lctscap_data/processed/capture24/annotations \
  --output_dir outputs/r007/eval_full \
  --skip_bertscore \
  --compare_with outputs/r007/noalign_test_predictions.jsonl
```

## 预期产物
- `outputs/r007/full_test_predictions.jsonl`
- `outputs/r007/noalign_test_predictions.jsonl`
- `outputs/r007/eval_full/results.csv`
- `outputs/r007/eval_full/results.md`
- `outputs/r007/eval_full/comparison.md`

## 本轮不做的事
- 不回到 `phase2_flat` 再补训练
- 不开 `Phase 3`
- 不开 `ctx=512`
- 不开 `Phase 4 / LLMBridge`
- 不把 `event_evidence_*` 诊断结果混进主比较
- 不先跑 HARTH 再决定 aligner 去留

## 一句话结论

下一步实验计划就是先做 `R007 full vs noalign`。这一步做完，才能决定 aligner 还值不值得留在论文里；在这之前，扩方法和提前跑 HARTH 都不是最短路径。
