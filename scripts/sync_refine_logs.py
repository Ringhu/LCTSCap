#!/usr/bin/env python3
"""根据 refine-logs 和已验证产物生成中文实时状态页。"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

STEP_RE = re.compile(r"\[(?P<step>\d+)/(?P<total>\d+)\] loss=(?P<loss>[0-9.]+)")
ROW_RE = re.compile(r"^\|\s*(R\d+)\s*\|")


def load_text(path: Path) -> str:
    return path.read_text(errors="ignore") if path.exists() else ""


def format_ts(ts: float | None) -> str:
    if ts is None:
        return "missing"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def parse_thesis(path: Path) -> str:
    for line in load_text(path).splitlines():
        if line.startswith("> "):
            return line[2:].strip()
    return "hierarchical long-context time-series captioning with grounding-aware evaluation"


def parse_tracker(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in load_text(path).splitlines():
        if not ROW_RE.match(line):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 9:
            continue
        rows.append(
            {
                "run_id": parts[0],
                "milestone": parts[1],
                "purpose": parts[2],
                "variant": parts[3],
                "split": parts[4],
                "metrics": parts[5],
                "priority": parts[6],
                "status": parts[7],
                "notes": parts[8],
            }
        )
    return rows


def checkpoint_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    payload = torch.load(path, map_location="cpu")
    return {
        "exists": True,
        "epoch": payload.get("epoch"),
        "loss": payload.get("loss"),
        "mtime": path.stat().st_mtime,
    }


def run_log_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    lines = [line.strip() for line in load_text(path).splitlines() if line.strip()]
    if not lines:
        return {"exists": True, "state": "empty"}
    state = "running"
    for line in reversed(lines):
        if "Training complete" in line:
            state = "complete"
            break
    payload: dict[str, Any] = {
        "exists": True,
        "state": state,
        "last_line": lines[-1],
        "mtime": path.stat().st_mtime,
    }
    match = STEP_RE.search(lines[-1])
    if match:
        payload["step"] = int(match.group("step"))
        payload["total"] = int(match.group("total"))
        payload["loss"] = float(match.group("loss"))
    return payload


def sleep_edf_status(root: Path) -> dict[str, Any]:
    target = root / "sleep-cassette"
    if not target.exists():
        return {"exists": False}
    edf_files = sorted(target.glob("*.edf"))
    total_bytes = sum(p.stat().st_size for p in edf_files)
    return {
        "exists": True,
        "edf_files": len(edf_files),
        "total_gb": round(total_bytes / (1024**3), 2),
        "latest_file": edf_files[-1].name if edf_files else "",
        "robots_present": (root / "robots.txt").exists(),
    }


def refine_files(refine_dir: Path) -> list[Path]:
    return sorted(p for p in refine_dir.glob("*.md") if p.name != "LATEST_STATUS.md")


def latest_refine_time(refine_dir: Path) -> float | None:
    files = refine_files(refine_dir)
    return max((p.stat().st_mtime for p in files), default=None)


def doc_sync_status(path: Path, reference_mtime: float | None) -> str:
    if not path.exists():
        return "missing"
    if reference_mtime is None:
        return "current"
    return "stale" if path.stat().st_mtime < reference_mtime else "current"


def next_actions(tracker_rows: list[dict[str, str]]) -> list[str]:
    running = [r for r in tracker_rows if r["status"] == "RUNNING"]
    ready = [r for r in tracker_rows if r["status"] == "READY"]
    actions = []
    if running:
        actions.append(f"优先继续监控 `{running[0]['run_id']}`：{running[0]['purpose']}")
    if ready:
        actions.append(f"下一步待启动 `{ready[0]['run_id']}`：{ready[0]['purpose']}")
    if not actions:
        actions.append("当前没有待启动实验；应整理结果并准备下一轮 review。")
    return actions


def build_markdown(
    *,
    trigger: str,
    repo_root: Path,
    refine_dir: Path,
    tracker_rows: list[dict[str, str]],
    thesis: str,
    bosfix_best: dict[str, Any],
    phase2_flat: dict[str, Any],
    phase2_noalign: dict[str, Any],
    sleep_edf: dict[str, Any],
    refine_mtime: float | None,
) -> str:
    lines: list[str] = []
    lines.append("# Refine Logs 实时状态")
    lines.append("")
    lines.append("这是自动生成的最新状态页。它只用于快速看最近状态，不承担长期事实主文档职责。长期事实看 `PROGRESS.md`，文档入口看 `docs/DOC_INDEX.md`。")
    lines.append("")
    lines.append(f"- 生成时间：`{format_ts(datetime.now().timestamp())}`")
    lines.append(f"- 触发来源：`{trigger}`")
    lines.append("")
    lines.append("## 当前主线")
    lines.append("")
    lines.append(f"- `{thesis}`")
    lines.append("")
    lines.append("## Refine 文档快照")
    lines.append("")
    lines.append(f"- 最新 refine 更新时间：`{format_ts(refine_mtime)}`")
    for path in refine_files(refine_dir):
        lines.append(f"- `{path.name}`：`{format_ts(path.stat().st_mtime)}`")
    lines.append("")
    lines.append("## 实验状态")
    lines.append("")
    for row in tracker_rows:
        lines.append(
            f"- `{row['run_id']}` `{row['status']}`：{row['purpose']} | {row['variant']} | {row['notes']}"
        )
    lines.append("")
    lines.append("## 已验证产物")
    lines.append("")
    if bosfix_best.get("exists"):
        lines.append(
            f"- `phase2_bosfix` best checkpoint：`epoch={bosfix_best['epoch']}`，`loss={bosfix_best['loss']}`，更新时间 `{format_ts(bosfix_best['mtime'])}`"
        )
    if phase2_flat.get("exists"):
        lines.append(
            f"- `phase2_flat` 日志：`{phase2_flat['state']}`，更新时间 `{format_ts(phase2_flat['mtime'])}`"
        )
        lines.append(f"- `phase2_flat` 最后一行：`{phase2_flat['last_line']}`")
    if phase2_noalign.get("exists"):
        summary = f"`{phase2_noalign['state']}`"
        if 'step' in phase2_noalign:
            summary += f"，最新 step `{phase2_noalign['step']}/{phase2_noalign['total']}`，`loss={phase2_noalign['loss']}`"
        lines.append(f"- `phase2_noalign` 日志：{summary}，更新时间 `{format_ts(phase2_noalign['mtime'])}`")
        lines.append(f"- `phase2_noalign` 最后一行：`{phase2_noalign['last_line']}`")
    if sleep_edf.get("exists"):
        lines.append(
            f"- `Sleep-EDF` 原始下载：`{sleep_edf['edf_files']}` 个 EDF，`{sleep_edf['total_gb']} GB`，最新 `{sleep_edf['latest_file']}`，`robots.txt={sleep_edf['robots_present']}`"
        )
    lines.append("- `SHL` staging 路径：`/cluster1/user1/lctscap_data/auxiliary/raw/shl`")
    lines.append("")
    lines.append("## 文档同步检查")
    lines.append("")
    for name in ("README.md", "PROGRESS.md", "SPEC.md", "RESEARCH_BRIEF.md", "docs/DOC_INDEX.md"):
        lines.append(f"- `{name}`：`{doc_sync_status(repo_root / name, refine_mtime)}`")
    lines.append("")
    lines.append("## 当前下一步")
    lines.append("")
    for action in next_actions(tracker_rows):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="生成中文实时状态页。")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--refine-dir", default="refine-logs")
    parser.add_argument("--output", default="refine-logs/LATEST_STATUS.md")
    parser.add_argument("--trigger", default="manual")
    parser.add_argument("--phase2-bosfix-best", default="/cluster1/user1/lctscap_data/runs/phase2_bosfix/checkpoints/best.pt")
    parser.add_argument("--phase2-flat-log", default="/cluster1/user1/lctscap_data/runs/phase2_flat/logs/train_20260320_phase2_flat.log")
    parser.add_argument("--phase2-noalign-log", default="/cluster1/user1/lctscap_data/runs/phase2_noalign/logs/train_20260321_phase2_noalign.log")
    parser.add_argument("--sleep-edf-root", default="/cluster1/user1/lctscap_data/auxiliary/raw/sleep_edf")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    refine_dir = (repo_root / args.refine_dir).resolve()
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracker_rows = parse_tracker(refine_dir / "EXPERIMENT_TRACKER.md")
    thesis = parse_thesis(repo_root / "PROGRESS.md")
    refine_mtime = latest_refine_time(refine_dir)
    bosfix_best = checkpoint_info(Path(args.phase2_bosfix_best))
    phase2_flat = run_log_status(Path(args.phase2_flat_log))
    phase2_noalign = run_log_status(Path(args.phase2_noalign_log))
    sleep_edf = sleep_edf_status(Path(args.sleep_edf_root))

    markdown = build_markdown(
        trigger=args.trigger,
        repo_root=repo_root,
        refine_dir=refine_dir,
        tracker_rows=tracker_rows,
        thesis=thesis,
        bosfix_best=bosfix_best,
        phase2_flat=phase2_flat,
        phase2_noalign=phase2_noalign,
        sleep_edf=sleep_edf,
        refine_mtime=refine_mtime,
    )
    output_path.write_text(markdown)
    output_path.with_suffix('.json').write_text(
        json.dumps(
            {
                'generated_at': datetime.now().isoformat(),
                'trigger': args.trigger,
                'thesis': thesis,
                'tracker': tracker_rows,
                'phase2_bosfix_best': bosfix_best,
                'phase2_flat': phase2_flat,
                'phase2_noalign': phase2_noalign,
                'sleep_edf': sleep_edf,
                'refine_mtime': refine_mtime,
            },
            ensure_ascii=False,
            indent=2,
        )
        + '\n'
    )


if __name__ == '__main__':
    main()
