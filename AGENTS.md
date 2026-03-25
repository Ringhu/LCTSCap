# Repository Guidelines

## 文档定位

本文件是仓库的补充协作约定。Claude Code 的项目级主指令以 `CLAUDE.md` 为准。

日常入口先看 `docs/DOC_INDEX.md`。

高频文档只保留 4 个：
- `PROGRESS.md`
- `SPEC.md`
- `README.md`
- `docs/DOC_INDEX.md`

`RESEARCH_BRIEF.md` 只在导师汇报或论文写作时使用。`refine-logs/` 默认不是日常入口。

## 第一性原理

请使用第一性原理思考。不要假设用户已经把问题定义完整。若动机、目标或约束不清晰，先收缩问题，再开始改动。

## 方案规范

当需要提出修改或重构方案时，必须满足：
- 不允许给出兼容性补丁式方案；
- 不允许过度设计；
- 保持最短路径实现；
- 不允许擅自扩展到用户未要求的方向；
- 需要能通过全链路逻辑验证。

## 项目结构与核心架构

- 代码主目录：`src/lctscap/`
- 数据处理：`src/lctscap/data/`
- 模型模块：`src/lctscap/models/`
- 评测与验证：`src/lctscap/eval/`
- 基线：`src/lctscap/baselines/`
- 脚本入口：`scripts/`
- 配置：`configs/{data,model,train,eval}/`
- 测试：`tests/`

主链路：
`LocalEncoder -> ChannelFusion -> HierarchicalPlanner -> EventProposalHead -> RetrievalAligner -> CaptionDecoder`

`Phase 4` 只是可选占位，不是当前主线。

## 2026-03-21 当前全局研究状态

- 当前唯一主线：`hierarchical long-context time-series captioning with grounding-aware evaluation`
- 当前实验顺序必须保持：
  1. `R001-R003`
  2. `R004 phase2_flat`
  3. `R005 full vs flat`
  4. `R006 phase2_noalign`
  5. `R007 full vs noalign`
  6. `HARTH`
- 当前已验证事实：
  - `R004` 已完成，`best val_loss=0.5970`
  - `R005` 已完成，full 在主语义指标上明显优于 flat
  - `R006` 训练已完成，`R007` 评测已完成；结果是不支持把 aligner 保留为核心贡献
- 当前两个主技术问题：
  - `event_span_iou` 主结果仍偏低，说明 event evidence 还没有真正进入语言输出链路
  - decoder 文本仍有污染与语序问题；受限解码只能当清洗层，不能当根修复
- 明确禁止提前扩展的方向：
  - `Phase 3`
  - `ctx=512`
  - `Phase 4 / LLMBridge`
  - 大量 auxiliary benchmark 主表
  - human evaluation
- Phase 4 / paraphrase 的默认 LLM 占位统一使用 `Qwen3` 系列；`qwen3.5-plus` 视为托管服务模型，不直接当作本地 open-weight 默认。

## 环境、训练与常用命令

```bash
conda create -n lctscap python=3.12 -y
conda activate lctscap
pip install -e ".[dev]"
pip install -e ".[moment]"
ruff check src tests scripts
pytest -q tests
python scripts/preprocess.py --dataset capture24 --config configs/data/capture24.yaml
python scripts/generate_annotations.py --manifest_dir /path/to/lctscap_data/processed/capture24
CUDA_VISIBLE_DEVICES=2 python scripts/train.py --config configs/train/phase1.yaml
CUDA_VISIBLE_DEVICES=2 python scripts/train.py --config configs/train/phase2_bosfix.yaml
python scripts/evaluate.py --predictions_path <pred.jsonl> --gold_path <gold_dir> --skip_bertscore
```

默认始终使用 `GPU 2`，除非用户明确要求改卡。代码根目录记为 `<repo_root>`，数据与 checkpoint 根目录记为 `<data_root>`。

## 编码、命名与测试规范

- Python 使用 4 空格缩进，`ruff` 行宽 100。
- 函数、变量、文件、配置键使用 `snake_case`；类名使用 `PascalCase`。
- 修改训练流程、评测逻辑、数据 schema、phase 配置时，必须补充或更新 `tests/`。
- 常用定向测试：
  - `pytest -q tests/test_phase_framework.py`
  - `pytest -q tests/test_smoke.py`
- 维护中的项目文档必须使用中文。

## 管理中的文档集合

维护中的项目文档只包括：
- 根文档：`AGENTS.md`、`README.md`、`PROGRESS.md`、`SPEC.md`、`RESEARCH_BRIEF.md`、`CLAUDE.md`
- 研究收敛文档：`refine-logs/*.md`
- 设计文档：`docs/plans/*.md`
- 文档索引：`docs/DOC_INDEX.md`

以下内容不是项目主文档，不参与翻译、合并、自动同步：
- `.vendor/` 第三方依赖许可证
- `.skill-build/` 技能构建产物
- `.pytest_cache/` 等工具缓存文档

## 文档写入规范

### 文档角色
- `docs/DOC_INDEX.md`：唯一总入口
- `README.md`：项目简介、环境、命令、运行入口
- `PROGRESS.md`：只写已验证事实与阶段进展
- `SPEC.md`：技术规格、phase 合同、评测协议
- `RESEARCH_BRIEF.md`：导师汇报口径、论文写作口径
- `refine-logs/LATEST_STATUS.md`：自动生成的最新状态页，不是长期事实主文档
- `refine-logs/EXPERIMENT_TRACKER.md`：实验台账，只维护 run 状态
- `refine-logs/REVIEW_SUMMARY.md`：长审查文档，不作为日常入口

### 写入要求
- 只记录已验证事实，不写猜测。
- 文档内容统一使用中文。
- 一个信息只放一个主归属文档；其他文档只做引用或摘要，不复制长段重复内容。
- 新文档只有在现有文档无法承担维护职责时才允许创建。
- 不要从 `refine-logs/` 开始读全目录。
- 不要为了新实验再开一篇说明文档。
- 过时闲聊、草稿、对话转储不保留在仓库中。

### 同步规则
出现下面任一情况时，必须同步相关 Markdown：
- 完成一个 phase 或恢复/中断训练：更新 `PROGRESS.md`
- 改动架构、loss、评测口径、配置字段：更新 `SPEC.md`
- 改动安装、脚本参数、运行方式：更新 `README.md`
- 改变研究问题、主实验设计、风险判断、导师汇报口径：更新 `RESEARCH_BRIEF.md`
- 任何 `research-*`、`experiment-*`、`run-experiment` 类工作结束后：必须同步 `refine-logs/EXPERIMENT_TRACKER.md` 与 `refine-logs/LATEST_STATUS.md`

## 自动同步 hook

仓库当前使用两个脚本形成对话后同步链路：
- `scripts/sync_refine_logs.py`：生成 `refine-logs/LATEST_STATUS.md`
- `scripts/sync_project_docs.py`：刷新 `README.md`、`PROGRESS.md`、`SPEC.md`、`RESEARCH_BRIEF.md`、`docs/DOC_INDEX.md` 的自动状态块

watcher 脚本：
- `scripts/watch_refine_logs_hook.py`

要求：
- 每次对话结束后，如果 session log 或 `refine-logs/` 有变化，watcher 必须先触发 `sync_refine_logs.py`，再触发 `sync_project_docs.py`。
- 这条规则同样适用于使用 `research-*` 与 `experiment-*` 技能后的工作回合。
- 主文档中的自动同步区域必须保留下面的标记，禁止人工删除：
  - `<!-- AUTO_SYNC_STATUS:START -->`
  - `<!-- AUTO_SYNC_STATUS:END -->`

## 文档使用顺序

1. 先看 `docs/DOC_INDEX.md` 找入口。
2. 再看 `PROGRESS.md` 判断当前做到哪里。
3. 再看 `SPEC.md` 明确应该怎么做。
4. 用 `README.md` 确认实际命令与入口。
5. 只有在导师汇报或论文写作时再看 `RESEARCH_BRIEF.md`。

如果代码、配置与文档不一致，先以已验证的代码行为为准定位事实，再立即修正文档，不要让不一致继续存在。

## 已知注意事项

- `momentfm` 目前只是可选依赖，尚未真正接入主模型链路。
- `LLMBridge` 仍是 Phase 4 占位实现，真实 LLM forward / LoRA 需后续补全。
- 训练或推理过程中若发现 GPU 2 异常退出，先检查 checkpoint、日志和系统级 GPU 错误，再决定是否 `--resume`。
