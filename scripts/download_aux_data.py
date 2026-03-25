#!/usr/bin/env python
"""Download helper for auxiliary cross-domain benchmarks."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


SLEEP_EDF_BASE = "https://physionet.org/files/sleep-edfx/1.0.0/"


def download_sleep_edf(output_dir: str, subset: str = "sleep-cassette") -> None:
    dest = Path(output_dir) / "sleep_edf"
    dest.mkdir(parents=True, exist_ok=True)
    url = f"{SLEEP_EDF_BASE}{subset}/"
    cmd = [
        "wget",
        "-r",
        "-np",
        "-nH",
        "--cut-dirs=3",
        "-R",
        "index.html*",
        "-P",
        str(dest),
        url,
    ]
    print("[INFO] Downloading Sleep-EDF subset:", subset)
    print("[INFO] Destination:", dest)
    subprocess.run(cmd, check=True)


def explain_shl(output_dir: str) -> None:
    dest = Path(output_dir) / "shl"
    dest.mkdir(parents=True, exist_ok=True)
    print(
        f"""
[INFO] SHL dataset requires manual retrieval at the moment.

Official notes:
- The SHL official server has technical issues.
- The preview version is available via IEEE DataPort / archived SHL website.

Recommended staging directory:
  {dest}

When the files are available locally, we can add an automated preprocessor for them.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download auxiliary benchmark datasets.")
    parser.add_argument("--dataset", type=str, choices=["sleep_edf", "shl"], required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--subset", type=str, default="sleep-cassette")
    args = parser.parse_args()

    if args.dataset == "sleep_edf":
        download_sleep_edf(args.output_dir, subset=args.subset)
    elif args.dataset == "shl":
        explain_shl(args.output_dir)


if __name__ == "__main__":
    main()
