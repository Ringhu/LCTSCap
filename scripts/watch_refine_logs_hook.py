#!/usr/bin/env python3
"""Approximate a post-dialog hook by syncing refine-logs after session activity settles."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def latest_session_signature(sessions_root: Path) -> dict[str, Any]:
    candidates = list(sessions_root.rglob("rollout-*.jsonl"))
    if not candidates:
        return {}
    latest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    stat = latest.stat()
    return {"path": str(latest), "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def refine_signature(refine_dir: Path) -> dict[str, Any]:
    signature = []
    for path in sorted(refine_dir.glob("*.md")):
        if path.name == "LATEST_STATUS.md":
            continue
        stat = path.stat()
        signature.append((path.name, stat.st_mtime_ns, stat.st_size))
    return {"files": signature}


def same_signature(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def run_command(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def run_sync_chain(repo_root: Path, output_path: Path, reason: str) -> tuple[bool, list[str]]:
    messages: list[str] = []

    docs_result = run_command(
        repo_root,
        [
            sys.executable,
            str(repo_root / "scripts" / "sync_project_docs.py"),
            "--repo-root",
            str(repo_root),
            "--trigger",
            f"pre-status-sync:{reason}",
        ],
    )
    if docs_result.returncode != 0:
        messages.append(f"sync_project_docs failed: {docs_result.stderr.strip()}")
        return False, messages
    messages.append("sync_project_docs complete")

    refine_result = run_command(
        repo_root,
        [
            sys.executable,
            str(repo_root / "scripts" / "sync_refine_logs.py"),
            "--repo-root",
            str(repo_root),
            "--output",
            str(output_path),
            "--trigger",
            f"post-doc-sync:{reason}",
        ],
    )
    if refine_result.returncode != 0:
        messages.append(f"sync_refine_logs failed: {refine_result.stderr.strip()}")
        return False, messages
    messages.append("sync_refine_logs complete")
    return True, messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch refine-logs and Codex session logs.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--sessions-root", default=str(Path.home() / ".codex" / "sessions"))
    parser.add_argument("--refine-dir", default="refine-logs")
    parser.add_argument("--output-path", default="refine-logs/LATEST_STATUS.md")
    parser.add_argument("--state-path", default="logs/refine_logs_hook_state.json")
    parser.add_argument("--log-path", default="logs/refine_logs_hook.log")
    parser.add_argument("--interval-sec", type=int, default=15)
    parser.add_argument("--debounce-sec", type=int, default=20)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sessions_root = Path(args.sessions_root).expanduser().resolve()
    refine_dir = (repo_root / args.refine_dir).resolve()
    output_path = (repo_root / args.output_path).resolve()
    state_path = (repo_root / args.state_path).resolve()
    log_path = (repo_root / args.log_path).resolve()

    state = load_state(state_path)
    append_log(log_path, "refine-logs hook watcher started")

    dirty_since = 0.0
    pending_reason = ""

    while True:
        session_sig = latest_session_signature(sessions_root)
        refine_sig = refine_signature(refine_dir)

        if not same_signature(session_sig, state.get("last_seen_session", {})):
            state["last_seen_session"] = session_sig
            dirty_since = time.monotonic()
            pending_reason = "session-log change"
            append_log(log_path, f"detected session activity: {session_sig.get('path', 'missing')}")

        if not same_signature(refine_sig, state.get("last_seen_refine", {})):
            state["last_seen_refine"] = refine_sig
            dirty_since = time.monotonic()
            pending_reason = "refine-logs change"
            append_log(log_path, "detected refine-logs change")

        already_synced = same_signature(session_sig, state.get("last_synced_session", {})) and same_signature(
            refine_sig, state.get("last_synced_refine", {})
        )
        if dirty_since and not already_synced and time.monotonic() - dirty_since >= args.debounce_sec:
            ok, messages = run_sync_chain(repo_root, output_path, pending_reason or "debounced")
            if ok:
                state["last_synced_session"] = session_sig
                state["last_synced_refine"] = refine_sig
                append_log(log_path, f"sync complete: reason={pending_reason or 'debounced'} steps={' ; '.join(messages)}")
            else:
                append_log(log_path, f"sync failed: reason={pending_reason or 'debounced'} details={' ; '.join(messages)}")
            dirty_since = 0.0
            pending_reason = ""

        save_state(state_path, state)
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
