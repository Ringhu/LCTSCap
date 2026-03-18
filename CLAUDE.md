# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Environment setup
conda create -n lctscap python=3.12 -y && conda activate lctscap
pip install -e ".[dev]"          # core + pytest, ruff
pip install -e ".[moment]"       # optional MOMENT encoder

# Lint
ruff check src/

# Tests
pytest tests/                    # all tests
pytest tests/test_smoke.py -v    # smoke tests verbose
pytest tests/test_smoke.py::test_full_model_forward -v  # single test

# Full pipeline (in order)
python scripts/preprocess.py --dataset capture24 --config configs/data/capture24.yaml
python scripts/generate_annotations.py --manifest_dir /path/to/lctscap_data/processed/capture24
python scripts/run_baseline.py --type template --config configs/train/phase0.yaml
python scripts/train.py --config configs/train/phase1.yaml
python scripts/train.py --config configs/train/phase2.yaml --resume
python scripts/evaluate.py --predictions_path <pred.jsonl> --gold_path <gold_dir> --skip_bertscore
```

## Architecture

**Task**: Generate hierarchical, verifiable captions from long time-series (128/256/512 semantic tokens, each = 10s sensor window).

### Model Pipeline (`src/lctscap/models/full_model.py`)

```
[B, T, C, L] raw sensor input
  → LocalEncoder  (per-channel Conv1d+Transformer, mean pool → [B*T*C, d])
  → ChannelFusion (attention-weighted sum over C → [B, T, d])
  → HierarchicalPlanner (token Transformer → H_token; segment pooling → H_seg)
  → EventProposalHead (type logits + span regression per token, top-K extraction)
  → RetrievalAligner (CLIP-style: H_seg → z_ts ↔ z_text ← TextEncoder)
  → CaptionDecoder (cross-attention over H_seg, autoregressive generation)
```

Ablation flags (`no_hierarchy`, `no_event`, `no_align`) skip the corresponding module. Phase 4 swaps CaptionDecoder for `LLMBridge` (PerceiverResampler → LLM soft prompt).

### Data Flow

1. **Window extraction** (`data/capture24.py`, `data/harth.py`): raw CSVs → downsample → 10s windows `[C, 500]` → `.pt` files + `manifest.json`
2. **Splits** (`data/splits.py`): deterministic participant-level split, no subject leakage
3. **Long-context samples** (`data/long_context.py`): sliding window over per-participant contiguous runs → `ContextSample` objects
4. **Annotation** (`data/annotation.py`): events extraction → segment summaries → template captions → evidence bullets
5. **Paraphrase** (`data/paraphrase.py`): LLM-based rewrite with hallucination/coverage verification
6. **Dataset/Collator** (`data/dataset.py`, `data/collator.py`): loads manifest JSON, stacks raw tensors or precomputed embeddings; collator pads + tokenizes captions

### Configuration

Nested dataclasses in `config.py`: `LCTSCapConfig` → `{DataConfig, ModelConfig, TrainConfig, EvalConfig}` + ablation flags. `load_config(yaml_path)` merges YAML overrides onto defaults.

**Note**: `config.py` and `models/full_model.py` each define their own `ModelConfig` dataclass. `scripts/train.py` maps from the config-level one to the model-level one in `build_model()`.

### Training Curriculum

| Phase | Config | Modules trained | Context lengths |
|-------|--------|-----------------|-----------------|
| 0 | phase0.yaml | None (template baseline eval) | 128/256/512 |
| 1 | phase1.yaml | encoder + planner + event + aligner | 128 |
| 2 | phase2.yaml | + decoder (encoder frozen) | 128, 256 |
| 3 | phase3.yaml | all (lower lr) + paraphrases | 128, 256, 512 |
| 4 | phase4.yaml | LLM bridge only, all else frozen | 256 |

Loss: `L_cap + 0.5*L_align + 0.3*L_event + 0.1*L_coverage`

### Evaluation Metrics (`src/lctscap/eval/`)

Four metric groups: Classic (BLEU/ROUGE/METEOR/BERTScore), Factuality (activity F1, dominant accuracy, transition accuracy, duration-bin accuracy), Grounding (span IoU, unsupported claim rate, order consistency via Kendall's tau), Retrieval (R@k, MRR, MedR bidirectional).

Composite early-stop score: `0.45*grounding + 0.35*semantic + 0.20*lexical`.

## GPU Usage

**IMPORTANT**: Always use GPU 2 (`CUDA_VISIBLE_DEVICES=2`) for all training and inference. Never use other GPUs, 除非用户要求。

## Data Paths

- **Code**: `<repo_root>/`
- **Data**: `<data_root>/` — `raw/`, `processed/{dataset}/manifest.json`, `contexts/`, `annotations/`, `runs/`

## Known Caveats

- `LCTSCapCollator.__init__` requires a `tokenizer` argument but `scripts/train.py` instantiates it without one — pass a HuggingFace tokenizer (e.g. `AutoTokenizer.from_pretrained("gpt2")`)
- `momentfm` is an optional dependency listed in `[moment]` extras but not yet integrated in model code
- `LLMBridge` produces soft prompts only; actual LLM forward + LoRA must be wired externally
