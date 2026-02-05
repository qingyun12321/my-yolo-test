#!/usr/bin/env python3
"""训练手部关键点模型，并将最佳权重拷贝到 models/ 目录。

设备选择策略（默认单卡）：
1) 指定 --multi-gpu 且多卡可用 -> 使用所有 CUDA 设备（如 "0,1,2"）
2) 其它情况有 CUDA -> 使用 "0"
3) 无 CUDA -> 使用 "cpu"
"""

from __future__ import annotations

import argparse
import os
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
        help=(
            "Base model path or name (default: models/yolo26n-pose.pt). "
            "If a name is provided, it will be downloaded into models/."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device selection, e.g. 'cpu', '0', or '0,1'.",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Enable multi-GPU training (DDP) when multiple GPUs are available.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable AMP. Note: AMP checks may download an extra non-pose model.",
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


def resolve_model_target(repo_root: Path, model_arg: str) -> tuple[Path, Path, str]:
    """Resolve model path and download directory.

    If a bare name is provided, download into repo_root/models.
    If a path is provided (absolute or relative), download into its parent.
    """
    raw_path = Path(model_arg).expanduser()
    if raw_path.is_absolute():
        target = raw_path
    elif raw_path.parent != Path("."):
        target = (repo_root / raw_path).resolve()
    else:
        target = (repo_root / "models" / raw_path.name).resolve()
    return target, target.parent, target.name


def main() -> int:
    """脚本主入口。"""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_path = (repo_root / args.data).resolve()
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 设备自动选择：默认单卡优先；--multi-gpu 时使用所有 GPU
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if args.multi_gpu and count > 1:
                device = ",".join(str(i) for i in range(count))
            else:
                device = "0"
        else:
            device = "cpu"

    print(f"Using device: {device}")

    model_path, download_dir, model_name = resolve_model_target(repo_root, args.model)
    if model_path.exists():
        model = YOLO(str(model_path))
    else:
        download_dir.mkdir(parents=True, exist_ok=True)
        print(
            "Base model not found at "
            f"{model_path}. Downloading into {download_dir}."
        )
        cwd = Path.cwd()
        try:
            os.chdir(download_dir)
            model = YOLO(model_name)
        finally:
            os.chdir(cwd)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=device,
        amp=args.amp,
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
