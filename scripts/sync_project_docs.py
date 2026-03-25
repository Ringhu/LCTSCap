#!/usr/bin/env python3
"""Sync the latest project state into the managed docs."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

AUTO_START = "<!-- AUTO_SYNC_STATUS:START -->"
AUTO_END = "<!-- AUTO_SYNC_STATUS:END -->"
ROW_RE = re.compile(r"^\|\s*(R\d+)\s*\|")


def load_text(path: Path) -> str:
    return path.read_text(errors="ignore") if path.exists() else ""


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


def parse_thesis(path: Path) -> str:
    for line in load_text(path).splitlines():
        if line.startswith("> "):
            return line[2:].strip()
    return "hierarchical long-context time-series captioning with grounding-aware evaluation"


def summarize_tracker(rows: list[dict[str, str]]) -> tuple[list[str], list[str], str]:
    running = [row for row in rows if row["status"] == "RUNNING"]
    ready = [row for row in rows if row["status"] == "READY"]

    bullets: list[str] = []
    for row in rows:
        if row["run_id"] in {"R004", "R005", "R006", "R007"}:
            bullets.append(f"- `{row['run_id']}` `{row['status']}`：{row['purpose']}；{row['notes']}")

    risks = [
        "- `event_span_iou` 主结果仍偏低，强可验证结论还没成立。",
        "- decoder 文本污染已部分缓解，但语序和句法错误仍存在。",
    ]

    if running:
        next_action = f"优先继续监控 `{running[0]['run_id']}`；完成后再推进其后续评测。"
    elif ready:
        next_action = f"下一步应启动 `{ready[0]['run_id']}`。"
    else:
        next_action = "当前没有待启动实验；应整理结果并准备下一轮 review。"

    return bullets, risks, next_action


def build_detailed_auto_block(thesis: str, rows: list[dict[str, str]], trigger: str, synced_at: str) -> str:
    bullets, risks, next_action = summarize_tracker(rows)
    lines = [
        AUTO_START,
        "## 自动同步状态",
        "",
        f"- 同步时间：`{synced_at}`",
        f"- 触发来源：`{trigger}`",
        f"- 当前主线：`{thesis}`",
        "",
        "### 核心实验状态",
        "",
        *bullets,
        "",
        "### 当前主要风险",
        "",
        *risks,
        "",
        "### 下一步",
        "",
        f"- {next_action}",
        AUTO_END,
    ]
    return "\n".join(lines)


def build_compact_auto_block(thesis: str, rows: list[dict[str, str]], synced_at: str) -> str:
    _, _, next_action = summarize_tracker(rows)
    lines = [
        AUTO_START,
        "## 自动同步状态",
        "",
        f"- 同步时间：`{synced_at}`",
        f"- 当前主线：`{thesis}`",
        f"- 当前下一步：{next_action}",
        "- 详细入口：`docs/DOC_INDEX.md`",
        "- 最新状态页：`refine-logs/LATEST_STATUS.md`",
        AUTO_END,
    ]
    return "\n".join(lines)


def replace_or_insert_block(text: str, block: str) -> str:
    if AUTO_START in text and AUTO_END in text:
        start = text.index(AUTO_START)
        end = text.index(AUTO_END) + len(AUTO_END)
        return text[:start] + block + text[end:]

    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        insert_at = 1
        while insert_at < len(lines) and lines[insert_at].strip() == "":
            insert_at += 1
        new_lines = lines[:insert_at] + [""] + block.splitlines() + [""] + lines[insert_at:]
        return "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")
    return block + "\n\n" + text


def sync_doc(path: Path, block: str) -> None:
    original = load_text(path)
    updated = replace_or_insert_block(original, block)
    path.write_text(updated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync managed docs with latest project status.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--trigger", default="manual")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    tracker_path = repo_root / "refine-logs" / "EXPERIMENT_TRACKER.md"
    progress_path = repo_root / "PROGRESS.md"

    rows = parse_tracker(tracker_path)
    thesis = parse_thesis(progress_path)
    synced_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    compact_block = build_compact_auto_block(thesis, rows, synced_at)
    detailed_block = build_detailed_auto_block(thesis, rows, args.trigger, synced_at)

    for rel in ["README.md", "PROGRESS.md", "SPEC.md", "RESEARCH_BRIEF.md"]:
        path = repo_root / rel
        if path.exists():
            sync_doc(path, compact_block)

    doc_index_path = repo_root / "docs" / "DOC_INDEX.md"
    if doc_index_path.exists():
        sync_doc(doc_index_path, detailed_block)


if __name__ == "__main__":
    main()
