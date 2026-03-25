# LCTSCap

<!-- AUTO_SYNC_STATUS:START -->
## 自动同步状态

- 同步时间：`2026-03-25 00:48:44`
- 当前主线：`hierarchical long-context time-series captioning with grounding-aware evaluation`
- 当前下一步：当前没有待启动实验；应整理结果并准备下一轮 review。
- 详细入口：`docs/DOC_INDEX.md`
- 最新状态页：`refine-logs/LATEST_STATUS.md`
<!-- AUTO_SYNC_STATUS:END -->

当前进度看 `PROGRESS.md`。文档入口看 `docs/DOC_INDEX.md`。

## 项目简介

LCTSCap 做的是长时间序列短描述生成。当前唯一主线是：

> hierarchical long-context time-series captioning with grounding-aware evaluation

当前主设置输入 `128 / 256` 个语义 token。每个 token 对应 `10` 秒窗口。输出是 `1-2` 句 factual caption。

当前不提前扩展到：`Phase 3`、`ctx=512`、`Phase 4 / LLMBridge`、大规模 auxiliary benchmark 主表、人工评测。

## 新接手先看什么

- 先看 `PROGRESS.md`。这里写当前主线、阶段、风险和下一步。
- 再看 `SPEC.md`。这里写模型合同、阶段边界和评测口径。
- 需要导师汇报或论文口径时看 `RESEARCH_BRIEF.md`。
- 需要 run 台账时看 `refine-logs/EXPERIMENT_TRACKER.md`。

## 目录结构

```text
src/lctscap/                 主代码
  ├── data/                  数据处理与标注
  ├── models/                模型模块
  ├── eval/                  评测与报告
scripts/                     训练、推理、评测、同步脚本
configs/                     数据/模型/训练配置
tests/                       单元测试
refine-logs/                 研究整理记录与实验台账
docs/plans/                  当前仍在执行的临时计划页
```

## 环境与安装

```bash
conda create -n lctscap python=3.12 -y
conda activate lctscap
pip install -e .
pip install -e ".[dev]"
pip install -e ".[moment]"   # 可选
```

## 常用命令

```bash
# 数据预处理
python scripts/preprocess.py --dataset capture24 --config configs/data/capture24.yaml
python scripts/generate_annotations.py --manifest_dir /path/to/lctscap_data/processed/capture24

# 训练
CUDA_VISIBLE_DEVICES=2 python scripts/train.py --config configs/train/phase1.yaml
CUDA_VISIBLE_DEVICES=2 python scripts/train.py --config configs/train/phase2_bosfix.yaml
CUDA_VISIBLE_DEVICES=2 python scripts/train.py --config configs/train/phase2_flat.yaml
CUDA_VISIBLE_DEVICES=2 python scripts/train.py --config configs/train/phase2_noalign.yaml

# 基础推理与评测
python scripts/generate_predictions.py --config configs/train/phase2_bosfix.yaml --checkpoint /path/to/best.pt --data_root /path/to/lctscap_data --split val
python scripts/evaluate.py --predictions_path /path/to/predictions.jsonl --gold_path /path/to/annotations --skip_bertscore
```

## R007 正式命令模板

`R007` 只比较 full 和 `noalign`。两边必须同口径生成，只换模型，不换解码策略。

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

## 当前推理侧清洗开关

```bash
# 受限解码：压住脏符号和重复模式
python scripts/generate_predictions.py \
  --config configs/train/phase2_bosfix.yaml \
  --checkpoint /path/to/best.pt \
  --data_root /path/to/lctscap_data \
  --split val \
  --restrict_to_caption_vocab \
  --caption_vocab_split train \
  --repetition_penalty 1.1 \
  --no_repeat_ngram_size 3

# 证据文本诊断：把 predicted_events 写进可解析文本
python scripts/generate_predictions.py \
  --config configs/train/phase2_bosfix.yaml \
  --checkpoint /path/to/best.pt \
  --data_root /path/to/lctscap_data \
  --split val \
  --emit_event_evidence \
  --evidence_text_mode append
```

说明：
- 受限解码现在只算文本清洗层。它能止住脏格式，不能替代训练侧修复。
- `--emit_event_evidence` 和 `evidence_text_mode` 现在只算诊断功能。主比较默认不要混进去。
- 当前论文承诺的是 grounding-aware evaluation，不是强 verifiable captioning。
