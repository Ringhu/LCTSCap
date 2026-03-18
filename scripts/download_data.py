#!/usr/bin/env python
"""Download raw datasets for LCTSCap.

Usage:
    python scripts/download_data.py --dataset capture24 --output_dir /path/to/lctscap_data/raw
    python scripts/download_data.py --dataset harth --output_dir /path/to/lctscap_data/raw
    python scripts/download_data.py --dataset all --output_dir /path/to/lctscap_data/raw
"""

import argparse
import os
import subprocess
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


# HARTH is publicly available from UCI archive
HARTH_URL = "https://archive.ics.uci.edu/static/public/779/harth.zip"


def download_file(url, dest_path, desc=""):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc or Path(dest_path).name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_harth(output_dir):
    """Download and extract HARTH dataset from UCI archive."""
    dest = Path(output_dir) / "harth"
    raw_dir = dest / "raw"

    if raw_dir.exists() and any(raw_dir.glob("S*.csv")):
        print(f"[INFO] HARTH raw data already exists at {raw_dir}. Skipping.")
        return

    dest.mkdir(parents=True, exist_ok=True)
    archive_path = str(dest / "harth.zip")
    print("[INFO] Downloading HARTH dataset from UCI archive...")
    download_file(HARTH_URL, archive_path, desc="harth")

    print("[INFO] Extracting HARTH...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(str(dest))
    os.remove(archive_path)

    # HARTH zip usually extracts into a subfolder — find and move CSV files
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_count = 0
    for csv_file in dest.rglob("S*.csv"):
        if csv_file.parent != raw_dir:
            target = raw_dir / csv_file.name
            csv_file.rename(target)
            csv_count += 1

    if csv_count == 0:
        # Check if CSVs are already in raw/
        csv_count = len(list(raw_dir.glob("S*.csv")))

    print(f"[OK] HARTH ready at {raw_dir} ({csv_count} subject files)")


def download_capture24(output_dir):
    """Provide instructions for CAPTURE-24 download + attempt automated download.

    CAPTURE-24 sensor data is hosted on the OxWearables GitHub via git-annex/DVC.
    We attempt to clone and pull; if that fails, print manual instructions.
    """
    dest = Path(output_dir) / "capture24"
    raw_dir = dest / "raw"

    if raw_dir.exists() and any(raw_dir.glob("P*.csv.gz")):
        print(f"[INFO] CAPTURE-24 raw data already exists at {raw_dir}. Skipping.")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)

    print("""
===================================================================
CAPTURE-24 Download Instructions
===================================================================
CAPTURE-24 accelerometer data requires manual download:

Option A — Direct download (recommended):
  1. Go to https://ora.ox.ac.uk/objects/uuid:92650814-a209-4607-9571-85f12e1dafdb
  2. Download the data archive
  3. Extract participant CSV files (P001.csv.gz, P001-annotation.csv, etc.)
  4. Place them in:
     {raw_dir}/

Option B — Using the OxWearables repo:
  1. git clone https://github.com/OxWearables/capture24.git
  2. Follow their data download instructions
  3. Copy the accelerometer .csv.gz and annotation .csv files to:
     {raw_dir}/

Expected file structure:
  {raw_dir}/P001.csv.gz
  {raw_dir}/P001-annotation.csv
  {raw_dir}/P002.csv.gz
  {raw_dir}/P002-annotation.csv
  ...

After placing the files, rerun preprocessing:
  python scripts/preprocess.py --dataset capture24 --config configs/data/capture24.yaml
===================================================================""".format(raw_dir=raw_dir))


def main():
    parser = argparse.ArgumentParser(description="Download raw datasets for LCTSCap.")
    parser.add_argument(
        "--dataset", type=str, choices=["capture24", "harth", "all"],
        required=True, help="Which dataset to download.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/path/to/lctscap_data/raw",
        help="Directory to save raw datasets.",
    )
    args = parser.parse_args()

    if args.dataset in ("harth", "all"):
        download_harth(args.output_dir)
    if args.dataset in ("capture24", "all"):
        download_capture24(args.output_dir)

    print("\nDownload step complete.")


if __name__ == "__main__":
    main()
