#!/usr/bin/env python3
"""清理 output/logs 下的日志文件。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def find_repo_root() -> Path:
    """向上搜索仓库根目录（包含 pyproject.toml 与 src/）。"""
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    raise RuntimeError(f"无法定位仓库根目录: {this_file}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Remove files under output/logs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be removed without deleting them.",
    )
    return parser.parse_args()


def main() -> int:
    """脚本主入口。

    返回:
        int: 0 表示正常完成，非 0 表示异常。
    """
    args = parse_args()
    repo_root = find_repo_root()
    log_dir = repo_root / "output" / "logs"

    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        return 1

    files = [p for p in log_dir.iterdir() if p.is_file()]

    if not files:
        print(f"No log files to remove in: {log_dir}")
        return 0

    if args.dry_run:
        print("Dry run: the following files would be removed:")
        for path in files:
            print(path)
        return 0

    removed = 0
    for path in files:
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            print(f"Failed to remove {path}: {exc}")

    print(f"Removed {removed} file(s) from: {log_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
