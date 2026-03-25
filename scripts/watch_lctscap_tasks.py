#!/usr/bin/env python3
"""Watch long-running LCTSCap tasks and append noteworthy updates to a log."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


EPOCH_DONE_RE = re.compile(
    r"Epoch (?P<epoch>\d+) done .* train_loss=(?P<train>[0-9.]+) val_loss=(?P<val>[0-9.]+) composite=(?P<composite>[0-9.]+)"
)
BEST_RE = re.compile(
    r"New best model saved \(composite=(?P<composite>[0-9.]+), val_loss=(?P<val>[0-9.]+)\)"
)
STEP_RE = re.compile(
    r"\[(?P<step>\d+)/(?P<total>\d+)\] loss=(?P<loss>[0-9.]+)"
)
CTX_EPOCH_RE = re.compile(
    r"Epoch (?P<epoch>\d+) ctx=(?P<ctx>\d+): loss=(?P<loss>[0-9.]+)"
)


def load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def append_log(path: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a") as f:
        f.write(f"[{timestamp}] {message}\n")


def scan_training(log_path: Path, ckpt_dir: Path, state: dict[str, Any], out_log: Path) -> None:
    if not log_path.exists():
        return

    text = log_path.read_text(errors="ignore")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return

    last_epoch_done = None
    last_best = None
    last_step = None
    last_ctx_epoch = None
    for line in lines:
        if EPOCH_DONE_RE.search(line):
            last_epoch_done = line
        if BEST_RE.search(line):
            last_best = line
        if STEP_RE.search(line):
            last_step = line
        if CTX_EPOCH_RE.search(line):
            last_ctx_epoch = line

    if last_epoch_done and state.get("last_epoch_done") != last_epoch_done:
        m = EPOCH_DONE_RE.search(last_epoch_done)
        append_log(
            out_log,
            f"training epoch {m.group('epoch')} complete: train_loss={m.group('train')}, val_loss={m.group('val')}, composite={m.group('composite')}",
        )
        state["last_epoch_done"] = last_epoch_done

    if last_best and state.get("last_best") != last_best:
        m = BEST_RE.search(last_best)
        append_log(
            out_log,
            f"new best checkpoint: composite={m.group('composite')}, val_loss={m.group('val')}",
        )
        state["last_best"] = last_best

    if last_ctx_epoch and state.get("last_ctx_epoch") != last_ctx_epoch:
        m = CTX_EPOCH_RE.search(last_ctx_epoch)
        append_log(
            out_log,
            f"finished train pass for epoch {m.group('epoch')} ctx={m.group('ctx')}: loss={m.group('loss')}",
        )
        state["last_ctx_epoch"] = last_ctx_epoch

    if last_step and state.get("last_step") != last_step:
        m = STEP_RE.search(last_step)
        state["last_step"] = last_step
        state["step_summary"] = {
            "step": int(m.group("step")),
            "total": int(m.group("total")),
            "loss": float(m.group("loss")),
        }

    for name in ("best.pt", "latest.pt"):
        ckpt_path = ckpt_dir / name
        if not ckpt_path.exists():
            continue
        mtime_key = f"{name}_mtime"
        current_mtime = ckpt_path.stat().st_mtime
        if state.get(mtime_key) == current_mtime:
            continue
        state[mtime_key] = current_mtime
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            append_log(
                out_log,
                f"checkpoint {name} updated: epoch={ckpt.get('epoch')}, loss={ckpt.get('loss')}",
            )
        except Exception as exc:
            append_log(out_log, f"checkpoint {name} updated but could not be read: {exc}")


def scan_sleep_edf(root: Path, state: dict[str, Any], out_log: Path) -> None:
    target = root / "sleep-cassette"
    if not target.exists():
        return

    files = sorted([p for p in target.iterdir() if p.is_file()])
    total_bytes = sum(p.stat().st_size for p in files)
    total_files = len(files)
    latest_file = files[-1].name if files else ""

    if state.get("sleep_files") != total_files or state.get("sleep_latest") != latest_file:
        append_log(
            out_log,
            f"sleep-edf download progress: files={total_files}, size_mb={total_bytes / (1024 * 1024):.1f}, latest={latest_file}",
        )
        state["sleep_files"] = total_files
        state["sleep_bytes"] = total_bytes
        state["sleep_latest"] = latest_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--sleep-edf-root", required=True)
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--output-log", required=True)
    parser.add_argument("--interval-sec", type=int, default=120)
    args = parser.parse_args()

    train_log = Path(args.train_log)
    ckpt_dir = Path(args.checkpoint_dir)
    sleep_root = Path(args.sleep_edf_root)
    state_path = Path(args.state_path)
    out_log = Path(args.output_log)

    out_log.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    state = load_state(state_path)
    append_log(out_log, "watcher started")

    while True:
        scan_training(train_log, ckpt_dir, state, out_log)
        scan_sleep_edf(sleep_root, state, out_log)
        save_state(state_path, state)
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
