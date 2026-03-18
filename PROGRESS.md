# LCTSCap 项目进度总结

## 项目概述

**LCTSCap** (Long-Context Time Series Captioning) 是一个面向 CCF-A 会议的研究项目，目标是从长时间序列（128/256/512 个语义 token，每个 token = 10 秒传感器窗口）生成**层次化、可验证**的自然语言描述。

### 核心创新点
1. **长上下文建模**: 支持 128/256/512 窗口（~21/43/85 分钟）的时序数据
2. **层次化生成**: 事件表 → 段落摘要 → 全局描述 三级结构
3. **可验证性**: 每句描述都有事件证据支撑，可追溯到原始时序段
4. **课程式训练**: 5 阶段渐进式训练策略

### 代码与数据位置
- **代码**: `<repo_root>/`
- **数据**: `<data_root>/`
- **GPU**: 默认使用 GPU 2 (`CUDA_VISIBLE_DEVICES=2`)

---

## 已完成工作

### Phase 0: 项目搭建与模板基线 [已完成]

#### 0.1 工程骨架搭建
- 创建完整的 Python 包结构 (`src/lctscap/`)
- 38 个源文件: data/, models/, eval/, baselines/, utils/
- 配置系统 (`config.py`): 嵌套 dataclass + YAML 加载
- pyproject.toml, environment.yml, SPEC.md, CLAUDE.md

#### 0.2 数据预处理

**CAPTURE-24 数据集:**
- 151 名参与者, 每人 ~24 小时, 100Hz → 50Hz 下采样
- 数据格式: CSV.gz (`time, x, y, z, annotation`), 标注嵌入在 CSV 中
- `annotation-label-dictionary.csv` 将原始标注映射到 WillettsSpecific2018 标签
- 粗粒度活动: sleeping, sitting, standing, walking, cycling, vehicle, household, running, other, unknown
- **输出**: 1,398,027 个窗口 `[3, 500]`, 190,704 个上下文样本
- 处理时间: ~50 分钟

**HARTH 数据集:**
- 22 名受试者, ~90-120 分钟, 50Hz, 6 通道 (背部+大腿 xyz)
- 12 种活动类型
- **输出**: 12,911 个窗口 `[6, 500]`, 1,016 个上下文样本

**数据划分:**
| 数据集 | 训练集 | 验证集 | 测试集 | 划分方式 |
|---------|--------|--------|--------|----------|
| CAPTURE-24 | 70% | 10% | 20% | 按参与者 |
| HARTH | 60% | - | 40% | 按受试者 |

#### 0.3 标注生成
- 自动事件提取: 合并连续同标签窗口 → Event 对象
- 段落摘要: 每 32 个 token 一句模板描述
- 全局描述: short (1句) + long (2-4句) 模板生成
- 证据子弹: 结构化证据条目

#### 0.4 模板基线评估

| 指标 | CAPTURE-24 | HARTH |
|------|------------|-------|
| Activity Mention F1 | 0.814 | 0.397 |
| Dominant Activity Acc | 0.881 | 0.853 |
| Event Span IoU | 1.000 | 1.000 |
| Order Consistency | 1.000 | 1.000 |
| BLEU/ROUGE/METEOR | 1.000 | 1.000 |

> 模板基线 = gold reference, 所以文本指标为 1.0; Activity F1 受限于关键词提取能力

---

### Phase 1: 编码器 + 规划器 + 对齐器训练 [已完成]

#### 1.1 训练配置
- **训练模块**: LocalEncoder + ChannelFusion + HierarchicalPlanner + EventProposalHead + RetrievalAligner
- **跳过模块**: CaptionDecoder (cap_loss=0)
- **上下文长度**: 128 (单一)
- **数据**: 45,499 训练 / 6,380 验证 样本
- **超参**: batch_size=4, grad_accum=8 (有效 batch=32), lr=3e-4
- **Loss**: align_loss (InfoNCE) × 1.0 + event_loss (CE + smooth_L1) × 1.0
- **模型参数**: 68,080,408 (68M)

#### 1.2 训练过程中修复的 Bug
| 文件 | 问题 | 修复 |
|------|------|------|
| `collator.py` | tokenizer 参数必须传入 | 改为可选, Phase 1 无 decoder 不需要 tokenizer |
| `collator.py` | 事件 GT 格式不匹配 | 新增 `_events_to_per_token()` 将 per-event 转为 per-token 格式 |
| `config.py` | Phase YAML 单一 `context_len` vs 列表 `context_lens` | 自动转换 `context_len` → `context_lens` 列表 |
| `config.py` | 嵌套 training YAML (optimizer.lr_new) 无法解析 | 展平嵌套 config 字段 |
| `config.py` | loss_weights 缺失 | 添加到 TrainConfig, 支持从 YAML 读取 |
| `text_encoder.py` | SentenceTransformer inference_mode 张量无法 backward | `.clone().detach()` |
| `aligner.py` | Text encoder (CPU) vs projection (GPU) 设备不匹配 | 显式 `.to(device)` |
| `train.py` | 验证阶段对 encoder-only phase 返回 0 | 添加 align+event loss 计算 |
| `train.py` | 无进度日志, epoch 过长无输出 | 添加每 ~5% 步的进度日志 |

#### 1.3 训练结果

| Epoch | Train Loss | Align Loss | Event Loss | Val Loss | 备注 |
|-------|-----------|------------|------------|----------|------|
| 0 | 32.20 | 1.02 | 31.18 | 31.34 | |
| 3 | 29.96 | 0.90 | 29.06 | 31.01 | |
| 7 | 29.64 | 0.85 | 28.79 | 30.82 | |
| **10** | **29.44** | **0.83** | **28.62** | **30.54** | **Best** |
| 15 | 29.25 | 0.80 | 28.45 | 30.63 | Early Stop |

- **Best val_loss**: 30.54 (Epoch 10)
- **Align loss 变化**: 1.02 → 0.83 (TS-text 对齐持续改善)
- **Event loss 变化**: 31.18 → 28.62 (事件检测改善)
- **训练时间**: ~10 小时 (16 epochs × 37 min/epoch)
- **Checkpoint**: `<data_root>/runs/phase1/checkpoints/best.pt` (546MB)

---

## 待完成工作

### Phase 2: 加入 Caption Decoder [进行中]

**目标**: 冻结 Phase 1 的 encoder, 训练 caption decoder 生成文本

| 配置项 | 值 |
|--------|-----|
| 上下文长度 | 128 + 256 |
| 训练模块 | channel_fusion, planner, event_head, decoder (encoder 冻结) |
| 冻结模块 | local_encoder, aligner (来自 Phase 1) |
| 初始化 | Phase 1 best.pt |
| Loss 权重 | cap=1.0, align=0.5, event=0.5, coverage=0.2 |
| Batch Size | ctx128=32, ctx256=16 |
| Early Stop | composite: 45% grounding + 35% semantic + 20% lexical |
| 最大 Epochs | 50, patience=7 |

**框架已补齐**:
1. `train.py`: 已支持 `init_from` 加载 Phase 1 checkpoint
2. `train.py`: 已支持按模块冻结/解冻 (`local_encoder`, `aligner` 等)
3. `train.py`: 已支持 decoder tokenizer 与离线 fallback
4. `train.py`: 已支持 composite early stop 代理分数
5. Phase 2 decoder forward 与 caption loss 已跑通

**2026-03-18 最新训练状态**:
- 本轮训练曾在 `epoch 3` 后中断，最后一次 `latest.pt` 更新时间为 `2026-03-18 14:46 CST`
- 排查结果：
  - 不是正常 early stopping
  - 未发现 `14:46` 附近的 OOM kill 或显式 Python traceback
  - 更像是训练进程被外部终止
  - `GPU 2` 在 `2026-03-18 11:52:49 CST` 出现过一次 `NVRM Xid 43`，属于潜在硬件/驱动风险
- 已通过 `--resume` 从 `<data_root>/runs/phase2/checkpoints/latest.pt` 恢复训练
- 恢复后日志确认：
  - `Resumed from checkpoint ... (epoch 3)`
  - `=== Starting training from epoch 4 ===`
  - `epoch 4 / ctx=128` 已继续推进，早期观测：
    - `step 0/1421: loss=24.3665`
    - `step 71/1421: loss=21.8160`

**当前判断**:
- Phase 2 已不再是“未开始”，而是正式进行中
- decoder 学习正常，后续重点继续观察：
  1. `ctx=256` 下的 event loss 是否继续下降
  2. validation composite 指标是否优于 `epoch 0`
  3. GPU 2 是否再次出现异常中断

---

### Phase 3: LLM 改写 + 长上下文 [Phase 2 之后]

**目标**: 引入 LLM 改写的多样化描述, 扩展到 ctx=512

| 配置项 | 值 |
|--------|-----|
| 上下文长度 | 128 + 256 + 512 |
| 训练模块 | channel_fusion, planner, event_head, aligner (解冻), decoder |
| 冻结模块 | local_encoder |
| 初始化 | Phase 2 best.pt |
| Loss 权重 | cap=1.0, align=0.5, event=0.5, coverage=0.2 |
| Learning Rate | lr_new=1e-4 (更低), lr_adapter=5e-6 |
| Batch Size | ctx128=32, ctx256=16, ctx512=8 |
| 最大 Epochs | 30, patience=5 |

**需要的代码改动**:
1. `scripts/generate_paraphrases.py`: 调用 LLM API 生成改写版本
2. `data/paraphrase.py`: 幻觉检测 + 覆盖率验证
3. DataLoader: 支持 mixed caption source (template + paraphrase)
4. ctx=512 的显存优化 (可能需要 gradient checkpointing)

---

### Phase 4: LLM Bridge (可选) [Phase 3 之后]

**目标**: 用 Perceiver Resampler + 冻结 LLM (Qwen2.5-1.5B) + LoRA 替代小 decoder

| 配置项 | 值 |
|--------|-----|
| 上下文长度 | 256 |
| 训练模块 | LLMBridge (Perceiver Resampler + LoRA) |
| 冻结模块 | 所有 Phase 1-3 模块 |
| 初始化 | Phase 3 best.pt |
| Loss 权重 | cap=1.0 only |
| LLM | Qwen/Qwen2.5-1.5B, LoRA r=16 alpha=32 |
| Perceiver | 32 latents |
| 最大 Epochs | 20, patience=5 |

**需要的代码改动**:
1. `models/llm_bridge.py`: 实现 PerceiverResampler + LLM soft prompt
2. 集成 LoRA (peft 库)
3. LLM forward + generation pipeline
4. 显存管理 (1.5B LLM + encoder 约需 ~20GB)

---

### Phase 5: 评估与消融实验 [Phase 3/4 之后]

**目标**: 完整评估 + 消融实验 + 对比实验, 形成论文表格

#### 主实验对比
| 方法 | 描述 |
|------|------|
| Template Baseline | Phase 0 模板生成 (上界参考) |
| Flat Encoder + Seq2Seq | 无层次结构的基线 |
| Ours (no hierarchy) | 消融: 去掉 HierarchicalPlanner |
| Ours (no align) | 消融: 去掉 RetrievalAligner |
| Ours (no event) | 消融: 去掉 EventProposalHead |
| Ours (no coverage) | 消融: 去掉 coverage loss |
| **Ours (Full)** | **完整模型** |
| Ours + LLM Bridge | Phase 4 (可选) |

#### 评估指标
| 组别 | 指标 | 说明 |
|------|------|------|
| Classic | BLEU, ROUGE-L, METEOR, BERTScore | 文本相似度 |
| Factuality | Activity F1, Dominant Acc, Transition Acc, Duration-bin Acc | 内容正确性 |
| Grounding | Event Span IoU, Unsupported Claim Rate, Order Consistency | 证据可追溯性 |
| Retrieval | R@1/5/10, MRR, MedR (双向) | 对齐质量 |

**需要的代码改动**:
1. 消融实验脚本 (`scripts/run_experiment.py`)
2. 对比模型实现 (flat baseline)
3. 完整评估 pipeline (所有指标)
4. 结果可视化 + LaTeX 表格生成

---

### Phase 6: 论文撰写 [最终阶段]

**目标**: CCF-A 会议论文

#### 论文结构
1. Introduction: 长上下文时序描述的挑战
2. Related Work: 时序描述 + 长序列建模 + 可验证生成
3. Method: 层次化编码器 + 事件感知规划器 + 检索对齐 + 可验证解码
4. Experiments: 主实验 + 消融 + 案例分析
5. Conclusion

#### 关键数据
- 两个数据集 (CAPTURE-24, HARTH) 的完整结果
- 三种上下文长度 (128, 256, 512) 的对比
- 四组消融实验
- 可视化: 注意力热力图, 事件检测示例, 生成样本对比

---

## 项目文件结构

```
LCTSCap/
├── configs/
│   ├── data/          capture24.yaml, harth.yaml
│   ├── model/         default.yaml
│   └── train/         phase0.yaml ~ phase4.yaml
├── scripts/
│   ├── preprocess.py          数据预处理
│   ├── generate_annotations.py 标注生成
│   ├── run_baseline.py        模板基线
│   ├── train.py               训练主脚本
│   ├── evaluate.py            评估脚本
│   └── run_experiment.py      实验管理
├── src/lctscap/
│   ├── config.py              配置系统
│   ├── data/
│   │   ├── schema.py          Pydantic 数据模型
│   │   ├── capture24.py       CAPTURE-24 预处理
│   │   ├── harth.py           HARTH 预处理
│   │   ├── long_context.py    长上下文样本构建
│   │   ├── annotation.py      标注生成
│   │   ├── paraphrase.py      LLM 改写
│   │   ├── dataset.py         PyTorch Dataset
│   │   ├── collator.py        DataLoader collator
│   │   └── splits.py          数据划分
│   ├── models/
│   │   ├── local_encoder.py   Conv1d + Transformer 编码器
│   │   ├── channel_fusion.py  注意力通道融合
│   │   ├── planner.py         层次化规划器
│   │   ├── event_head.py      事件提议头
│   │   ├── aligner.py         CLIP 式对齐器
│   │   ├── text_encoder.py    Sentence-Transformer 文本编码器
│   │   ├── decoder.py         Transformer 解码器
│   │   ├── llm_bridge.py      LLM Bridge (Phase 4)
│   │   ├── losses.py          所有损失函数
│   │   └── full_model.py      完整模型组装
│   ├── eval/
│   │   ├── classic_metrics.py BLEU/ROUGE/METEOR/BERTScore
│   │   ├── factuality.py      事实性指标
│   │   ├── grounding.py       可追溯性指标
│   │   ├── retrieval.py       检索指标
│   │   └── report.py          结果报告
│   └── baselines/
│       └── template_captioner.py  模板基线
└── tests/
    ├── test_smoke.py          冒烟测试
    ├── test_schema.py         数据模型测试
    ├── test_splits.py         数据划分测试
    ├── test_metrics.py        指标测试
    └── test_losses.py         损失函数测试
```

## 数据存储

```
<data_root>/
├── raw/
│   ├── capture24/             原始 CSV.gz (6.5GB)
│   └── harth/                 原始 CSV
├── processed/
│   ├── capture24/
│   │   ├── windows/           1,398,027 个 .pt 文件
│   │   ├── manifest.json      窗口清单
│   │   ├── splits.json        参与者划分
│   │   ├── contexts/          上下文样本 JSONL
│   │   ├── annotations/       标注后的样本 (190,704)
│   │   └── statistics.json    统计信息
│   └── harth/                 同上 (12,911 窗口, 1,016 上下文)
└── runs/
    ├── phase0_template/       模板基线结果
    │   ├── predictions/       预测输出
    │   └── eval/              评估结果
    └── phase1/
        └── checkpoints/
            ├── best.pt        最优模型 (546MB, Epoch 10, val_loss=30.54)
            └── latest.pt      最后检查点
```

---

*最后更新: 2026-03-17*
*Phase 1 训练完成, 准备进入 Phase 2*
