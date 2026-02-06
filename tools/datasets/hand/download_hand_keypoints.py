#!/usr/bin/env python3
"""下载并解压 Hand Keypoints 数据集到本项目 datasets/ 目录。"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


def find_repo_root() -> Path:
    """向上搜索仓库根目录（包含 pyproject.toml 与 src/）。"""
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    raise RuntimeError(f"无法定位仓库根目录: {this_file}")


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
    repo_root = find_repo_root()
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
