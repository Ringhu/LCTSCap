# Repository Guidelines

## 文档定位

本文件是仓库的主协作约定，面向 Codex、Claude Code 和人工贡献者。`CLAUDE.md` 中已有的工程约束已并入这里；后续若两者冲突，以 `AGENTS.md` 为准。


## 第一性原理
请使用第一性原理思考。你不能总是假设我非常清楚自己想要什么和该怎么得到。请保持审慎，从原始需求和问题出发，如果动机和目标不清晰，停下来和我讨论。

## 方案规范
当需要你给出修改或重构方案时必须符合以下规范:
不允许给出兼容性或补丁性的方案不允许过度设计，保持最短路径实现且不能违反第一条要求不允许自行给出我提供的需求以外的方案，例如一些兜底和降级方案，
这可能导致业务逻辑偏移问题
必须确保方案的逻辑正确，必须经过全链路的逻辑验证

## 项目结构与核心架构

- 代码主目录：`src/lctscap/`
- 数据处理：`src/lctscap/data/`
- 模型模块：`src/lctscap/models/`
- 评测与验证：`src/lctscap/eval/`
- 基线：`src/lctscap/baselines/`
- 脚本入口：`scripts/`
- 配置：`configs/{data,model,train,eval}/`
- 测试：`tests/`

任务目标是对长时间序列生成“层次化、可验证”的文本描述。主链路为：
`LocalEncoder -> ChannelFusion -> HierarchicalPlanner -> EventProposalHead -> RetrievalAligner -> CaptionDecoder`。
Phase 4 可选切换到 `LLMBridge`。

## 环境、训练与常用命令

```bash
conda create -n lctscap python=3.12 -y
conda activate lctscap
pip install -e ".[dev]"
pip install -e ".[moment]"   # 可选
ruff check src tests scripts
pytest -q tests
python scripts/preprocess.py --dataset capture24 --config configs/data/capture24.yaml
python scripts/generate_annotations.py --manifest_dir /path/to/lctscap_data/processed/capture24
CUDA_VISIBLE_DEVICES=2 python scripts/train.py --config configs/train/phase1.yaml
CUDA_VISIBLE_DEVICES=2 python scripts/train.py --config configs/train/phase2.yaml --resume
python scripts/evaluate.py --predictions_path <pred.jsonl> --gold_path <gold.jsonl>
```

默认始终使用 `GPU 2`，除非用户明确要求改卡。代码根目录记为 `<repo_root>`，数据与 checkpoint 根目录记为 `<data_root>`。

## 编码、命名与测试规范

- Python 使用 4 空格缩进，`ruff` 行宽 100。
- 函数、变量、文件、配置键使用 `snake_case`；类名使用 `PascalCase`。
- 修改训练流程、评测逻辑、数据 schema、phase 配置时，必须补充或更新 `tests/`。
- 常用定向测试：
  `pytest -q tests/test_phase_framework.py`
  `pytest -q tests/test_smoke.py`

## 文档如何辅助开发

- `README.md`：对外入口和可运行命令。新增脚本、改变训练/评测用法、调整依赖时要同步更新。
- `SPEC.md`：技术规格与实验合同。修改模型结构、loss、phase 定义、数据格式、评测协议时先对照这里，变更后必须回写。
- `PROGRESS.md`：当前真实进度与里程碑。开始工作前先看，避免重复劳动；完成阶段性结果、修复关键 bug、产出训练结论后立即更新。
- `RESEARCH_BRIEF.md`：论文叙事与导师沟通版本。做方法取舍、实验设计、汇报材料时优先参考，避免实现偏离研究主线。

推荐使用顺序：
1. 先看 `PROGRESS.md` 判断当前做到哪里。
2. 再看 `SPEC.md` 明确“应该怎么做”。
3. 用 `README.md` 确认实际命令与入口。
4. 用 `RESEARCH_BRIEF.md` 检查实现是否服务于论文故事。

如果代码、配置与文档不一致，先以“已验证的代码行为”为准定位事实，再立即修正文档，不要让漂移继续存在。

## 进度回写规则

出现下面任一情况时，必须同步更新相关 Markdown：

- 完成一个 phase 或恢复/中断训练：更新 `PROGRESS.md`
- 改动架构、loss、评测口径、配置字段：更新 `SPEC.md`
- 改动安装、脚本参数、运行方式：更新 `README.md`
- 改变研究问题、主实验设计、风险判断、导师汇报口径：更新 `RESEARCH_BRIEF.md`

更新要求：
- 写清日期、变更内容、结果指标、未解决风险
- 只记录已验证事实，不写猜测
- 训练进度至少记录：起止时间、epoch、best checkpoint、关键 loss/metric、是否中断及原因

## 已知注意事项

- `momentfm` 目前只是可选依赖，尚未真正接入主模型链路。
- `LLMBridge` 仍是 Phase 4 占位实现，真实 LLM forward / LoRA 需后续补全。
- 训练或推理过程中若发现 GPU 2 异常退出，先检查 checkpoint、日志和系统级 GPU 错误，再决定是否 `--resume`。
