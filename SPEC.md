# LCTSCap Technical Specification

## 1. Problem Definition

**Task**: Given a long-context time series represented as T ∈ {128, 256, 512} semantic tokens (each token = 10-second sensor window embedding), generate a hierarchical, verifiable natural language caption.

**Output structure**:
- Global caption (1-2 sentences)
- Segment summaries (one per 32-token segment)
- Event table with evidence spans

## 2. Data Pipeline

### 2.1 Source Datasets

| Dataset | Subjects | Duration | Sample Rate | Channels | Activities |
|---------|----------|----------|-------------|----------|------------|
| CAPTURE-24 | 151 | ~24h/person | 100Hz→50Hz | 3 (wrist xyz) | ~15 coarse types |
| HARTH | 22 | ~90-120min | 50Hz | 6 (back+thigh xyz) | 12 types |

### 2.2 Window Extraction
- Window size: 10 seconds @ 50Hz = 500 samples
- CAPTURE-24 window shape: [3, 500]
- HARTH window shape: [6, 500]

### 2.3 Long-Context Samples
- Context lengths: {128, 256, 512} windows
- Corresponding real time: ~21.3 / 42.7 / 85.3 minutes
- Stride: {32, 64} windows
- Only contiguous windows from single participant

### 2.4 Subject-Level Splits
- CAPTURE-24: 70% train / 10% val / 20% test (by participant)
- HARTH: 60% train / 40% test (by subject)
- **Critical**: No subject overlap between splits

## 3. Annotation Pipeline

### 3.1 Three-Level Hierarchy
1. **Event Table** (automatic): merge consecutive same-label windows
   - Fields: type, start_token, end_token, duration_sec, transition_to
2. **Segment Summaries** (template): one sentence per 32-token segment
3. **Global Caption** (template + LLM paraphrase): short (1 sentence) + long (2-4 sentences)

### 3.2 Annotation Budget
- Synthetic train: 12k-18k samples (auto template + LLM paraphrase)
- Human pilot: 100 (schema refinement)
- Human dev: 200 (hyperparameter tuning)
- Human test: 500 (main evaluation)
- Hard factual: 100 (grounding stress test)

## 4. Model Architecture

### Module A: Local Encoder
```
Input: [B*T*C, 1, 500]
Conv1d(1, d_model, kernel=25, stride=25) → 20 patches
+ learnable positional encoding
+ Transformer encoder (4 layers, 512 dim, 8 heads)
→ mean pool → [B*T*C, 512]
```

### Channel Fusion
```
Input: [B*T, C, 512]
Attention-weighted sum over channels
→ [B, T, 512]
```

### Module B: Hierarchical Planner
```
Token Transformer: 4 layers → H_token [B, T, 512]
Segment Pooling: mean+max per 32 tokens → H_seg [B, S, 512]
Event Head: type classification + span prediction → top-K events
```

### Module C: Retrieval Aligner
```
TS projection: mean(H_seg) → MLP → z_ts [B, 256]
Text encoder: sentence-transformers → MLP → z_text [B, 256]
InfoNCE: symmetric contrastive loss
```

### Module D: Caption Decoder
```
Transformer decoder (6 layers, 512 dim, 8 heads)
Cross-attention over: H_seg + top-K event tokens
Autoregressive generation
```

## 5. Training

### Loss Function
```
L = L_cap + 0.5*L_align + 0.5*L_event + 0.2*L_coverage
```

### Curriculum
- Phase 0: Template baseline (no training)
- Phase 1: Encoder + Planner + Aligner (ctx=128)
- Phase 2: + Decoder (ctx=128,256)
- Phase 3: + Paraphrases + ctx=512
- Phase 4: + LLM bridge (optional)

### Hyperparameters
- lr_new: 3e-4, lr_adapter: 1e-5
- AdamW, weight_decay=0.01, cosine schedule, 5% warmup
- Batch: ctx128=32, ctx256=16, ctx512=8
- Early stop: 0.45*grounding + 0.35*semantic + 0.20*lexical

## 6. Evaluation

### Metric Groups
| Group | Metrics | Purpose |
|-------|---------|---------|
| Classic | BLEU, ROUGE-L, METEOR, BERTScore | Reviewer familiarity |
| Factuality | Activity F1, dominant acc, transition acc, duration-bin acc | Content correctness |
| Grounding | Event span IoU, unsupported claim rate, order consistency | Evidence traceability |
| Retrieval | R@1/5/10 bidirectional | Alignment quality |

### Main Comparisons
1. Template baseline
2. MOMENT + seq2seq (flat, no hierarchy)
3. MOMENT + aligner + caption (no hierarchy)
4. Full model (hierarchy + aligner + verifier)

### Ablation Study
- Full model
- −Hierarchy (no_hierarchy)
- −Retrieval (no_align)
- −Event head (no_event)
- −Coverage (no_coverage)

## 7. Storage Layout
```
<repo_root>/                                       # code
<data_root>/                                       # data + runs
    ├── raw/{capture24,harth}/
    ├── processed/{capture24,harth}/{windows,manifests,annotations}/
    ├── splits/
    └── runs/phase{0,1,2,3,4}/
```
