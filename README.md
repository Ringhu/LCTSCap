# LCTSCap: Hierarchical and Verifiable Long-Context Time Series Captioning

## Overview

LCTSCap is a research system for generating hierarchical, verifiable natural language descriptions of long time series data. Unlike existing time series captioning work that handles short sequences (a few to dozens of time steps), LCTSCap operates on **long-context semantic token sequences** (128/256/512 tokens, where each token represents a 10-second window of sensor data).

## Key Innovation

- **Long-context captioning**: Each "token" is a 10-second window embedding, so 128/256/512 tokens = ~21/43/85 minutes of real-world sensor data
- **Hierarchical annotation**: 3-level structure (global summary → segment summaries → event table with evidence)
- **Verifiable generation**: Every caption claim can be traced back to evidence in the event table
- **Retrieval-augmented alignment**: CLIP-like dual encoder for TS↔text retrieval and caption verification

## Installation

```bash
# Create conda environment
conda create -n lctscap python=3.12 -y
conda activate lctscap

# Install the package
pip install -e .

# Or with MOMENT encoder support
pip install -e ".[moment]"

# Or with dev dependencies
pip install -e ".[dev]"
```

## Project Structure

```
LCTSCap/
├── configs/                  # YAML configuration files
│   ├── data/                 # Dataset configs (capture24, harth)
│   ├── model/                # Model architecture config
│   ├── train/                # Training phase configs (phase0-4)
│   └── eval/                 # Evaluation config
├── src/lctscap/              # Main package
│   ├── config.py             # Dataclass configuration
│   ├── data/                 # Data loading, preprocessing, annotation
│   ├── models/               # All model modules
│   ├── eval/                 # Evaluation metrics and reporting
│   ├── baselines/            # Template and retrieval baselines
│   └── utils/                # I/O, logging, visualization
├── scripts/                  # CLI scripts for all pipeline stages
├── tests/                    # Unit tests
└── notebooks/                # Analysis notebooks
```

## Data

- **Primary dataset**: [CAPTURE-24](https://github.com/OxWearables/capture24) — 151 participants, ~24h/person, 100Hz 3-axis wrist accelerometer
- **Validation dataset**: [HARTH](https://archive.ics.uci.edu/dataset/779/harth) — 22 subjects, 50Hz, 6-axis (back+thigh)
- Data stored at: `/path/to/lctscap_data/`

## Quick Start

```bash
# 1. Download data
python scripts/download_data.py --dataset capture24 --output_dir /path/to/lctscap_data/raw

# 2. Preprocess
python scripts/preprocess.py --dataset capture24 --config configs/data/capture24.yaml

# 3. Generate annotations
python scripts/generate_annotations.py --manifest_dir /path/to/lctscap_data/processed/capture24

# 4. Run template baseline
python scripts/run_baseline.py --type template --config configs/train/phase0.yaml

# 5. Train main model (curriculum)
python scripts/train.py --config configs/train/phase1.yaml
python scripts/train.py --config configs/train/phase2.yaml
python scripts/train.py --config configs/train/phase3.yaml

# 6. Evaluate
python scripts/evaluate.py --predictions_path /path/to/predictions.jsonl --gold_path /path/to/gold.jsonl
```

## Training Curriculum

| Phase | What | Context | Data | Key Metric |
|-------|------|---------|------|------------|
| 0 | Template baseline (no training) | 128/256/512 | Template | Factuality |
| 1 | Local encoder + Planner + Aligner | 128 | Template | R@5, Event F1 |
| 2 | + Caption decoder | 128, 256 | Template | Composite |
| 3 | + LLM paraphrase + ctx=512 | 128,256,512 | Template+Paraphrase | Composite |
| 4 | + LLM bridge (optional) | 256 | All | Composite |

## Evaluation Metrics

- **Classic**: BLEU, ROUGE-L, METEOR, BERTScore
- **Factuality**: Activity mention F1, dominant activity accuracy, transition accuracy, duration-bin accuracy
- **Grounding**: Event span IoU, unsupported claim rate, order consistency
- **Retrieval**: R@1/R@5/R@10 (bidirectional)
