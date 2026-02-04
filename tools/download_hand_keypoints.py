#!/usr/bin/env python3
"""下载并解压 Hand Keypoints 数据集到本项目 datasets/ 目录。"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Download hand keypoints dataset.")
    parser.add_argument(
        "--url",
        default="https://github.com/ultralytics/assets/releases/download/v0.0.0/hand-keypoints.zip",
        help="Dataset zip URL.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets",
        help="Directory to store dataset (relative to project root).",
    )
    return parser.parse_args()


def main() -> int:
    """脚本主入口。"""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / "hand-keypoints.zip"
    extract_dir = out_dir / "hand-keypoints"

    if not zip_path.exists():
        print(f"Downloading: {args.url}")
        urlretrieve(args.url, zip_path)
    else:
        print(f"Zip already exists: {zip_path}")

    if not extract_dir.exists():
        print(f"Extracting to: {extract_dir}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    else:
        print(f"Extract dir exists: {extract_dir}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
