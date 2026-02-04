#!/usr/bin/env python3
"""清理项目 runs 目录下的训练输出。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Remove runs directory under project.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files/dirs that would be removed without deleting them.",
    )
    return parser.parse_args()


def main() -> int:
    """脚本主入口。"""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = repo_root / "runs"

    if not runs_dir.exists():
        print(f"Runs directory not found: {runs_dir}")
        return 1

    if args.dry_run:
        print("Dry run: the following entries would be removed:")
        for path in runs_dir.iterdir():
            print(path)
        return 0

    shutil.rmtree(runs_dir)
    print(f"Removed runs directory: {runs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
