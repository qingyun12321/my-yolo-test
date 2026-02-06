#!/usr/bin/env python3
"""按官方 JSON2YOLO/Ultralytics 方式转换 COCO JSON 到 YOLO Seg 标签。

官方说明要点：
1) JSON2YOLO 转换能力已并入 Ultralytics；
2) 分割任务应使用 `convert_coco(..., use_segments=True)`；
3) 训练前先完成该转换，再配置 data yaml 并执行 `segment train`。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics.data.converter import convert_coco


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Convert COCO JSON annotations to YOLO segmentation labels.")
    parser.add_argument(
        "--labels-dir",
        default="datasets/coco-json/annotations",
        help=(
            "COCO annotation directory (official: labels_dir). "
            "The directory should contain instances_*.json files."
        ),
    )
    parser.add_argument(
        "--save-dir",
        default="datasets/yolo-seg",
        help="Output directory for converted YOLO labels/images references (official: save_dir).",
    )
    parser.add_argument(
        "--cls91to80",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Official converter option: map COCO91 class IDs to COCO80 IDs.",
    )
    parser.add_argument(
        "--lvis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Official converter option for LVIS dataset format.",
    )
    return parser.parse_args()


def main() -> int:
    """脚本主入口。"""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    labels_dir = (repo_root / args.labels_dir).resolve()
    save_dir = (repo_root / args.save_dir).resolve()

    if not labels_dir.exists():
        raise FileNotFoundError(f"labels_dir not found: {labels_dir}")

    json_files = sorted(labels_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in labels_dir: {labels_dir}. "
            "Expected files like instances_train.json / instances_val.json."
        )

    print("Converting COCO JSON to YOLO Seg labels with official converter...")
    print(f"labels_dir={labels_dir}")
    print(f"save_dir={save_dir}")
    convert_coco(
        labels_dir=str(labels_dir),
        save_dir=str(save_dir),
        use_segments=True,
        use_keypoints=False,
        cls91to80=bool(args.cls91to80),
        lvis=bool(args.lvis),
    )
    print("Conversion done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
