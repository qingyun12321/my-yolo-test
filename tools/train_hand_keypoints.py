#!/usr/bin/env python3
"""训练手部关键点模型，并将最佳权重拷贝到 models/ 目录。

设备选择策略（全自动）：
1) 多卡可用 -> 使用所有 CUDA 设备（如 "0,1,2"）
2) 单卡可用 -> 使用 "0"
3) 无 CUDA -> 使用 "cpu"
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Train hand keypoints model.")
    parser.add_argument(
        "--data",
        default="datasets/hand-keypoints.yaml",
        help="Dataset YAML path (relative to project root).",
    )
    parser.add_argument(
        "--model",
        default="models/yolo26n-pose.pt",
        help="Base model path or name (default: models/yolo26n-pose.pt).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device selection, e.g. 'cpu', '0', or '0,1'.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (>=1).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (positive int).",
    )
    parser.add_argument(
        "--out",
        default="models/hand-keypoints.pt",
        help="Output weight path to copy best.pt to.",
    )
    return parser.parse_args()


def main() -> int:
    """脚本主入口。"""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_path = (repo_root / args.data).resolve()
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 设备自动选择：优先使用多卡，其次单卡，最后 CPU
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count > 1:
                # 多卡时使用所有 GPU（0..N-1）
                device = ",".join(str(i) for i in range(count))
            else:
                # 单卡
                device = "0"
        else:
            # 无 CUDA
            device = "cpu"

    print(f"Using device: {device}")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=device,
    )

    # 训练结果中包含权重目录，取 best.pt
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"best.pt not found at: {best_path}")

    shutil.copy2(best_path, out_path)
    print(f"Saved best weight to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
