#!/usr/bin/env python3
"""按 Ultralytics 官方方式训练分割模型，并可选复制 best.pt。

对齐目标：
1) 与官方命令语义一致：`yolo segment train data=... model=... epochs=100 imgsz=640`；
2) 支持官方配置覆盖机制：`cfg=...` + `key=value` 形式覆盖；
3) 保留工程化便利：训练后可自动拷贝 `best.pt` 到固定路径。
"""

from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Train segmentation model with Ultralytics-official-compatible arguments."
    )
    parser.add_argument(
        "--data",
        default="training/configs/segment_data.yaml",
        help="Dataset YAML path (official: data=...).",
    )
    parser.add_argument(
        "--model",
        default="models/yolo26n-seg.pt",
        help=(
            "Model path or official model name (official: model=...). "
            "Example name: yolo11n-seg.pt."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (official example uses 100).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (official example uses 640).",
    )
    parser.add_argument(
        "--cfg",
        default=None,
        help=(
            "Optional Ultralytics cfg YAML (official: cfg=...). "
            "When provided, it is loaded first, then overridden by explicit args/overrides."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra official-style overrides, repeatable. "
            "Example: --override batch=8 --override lr0=0.005 --override cos_lr=True"
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Device override (official: device=...), e.g. cpu / 0 / 0,1.",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="If set and --device is empty, use all visible CUDA devices (device=0,1,...).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Optional batch override (official: batch=...).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional workers override (official: workers=...).",
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional cache override (official: cache=True/False).",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional amp override (official: amp=True/False).",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Optional project override (official: project=...).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional run name override (official: name=...).",
    )
    parser.add_argument(
        "--exist-ok",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Optional exist_ok override (official: exist_ok=True/False).",
    )

    parser.add_argument(
        "--out",
        default="models/object-seg-finetuned.pt",
        help="Target path for copying best.pt.",
    )
    parser.add_argument(
        "--no-copy-best",
        action="store_true",
        help="Disable best.pt copy step; keep only Ultralytics run outputs.",
    )
    return parser.parse_args()


def _resolve_repo_path(repo_root: Path, path_arg: str) -> Path:
    """将可能的相对路径解析为仓库内绝对路径。"""
    raw = Path(path_arg).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (repo_root / raw).resolve()


def _resolve_model_arg(repo_root: Path, model_arg: str) -> str:
    """解析模型参数：路径优先，否则按官方模型名处理。"""
    raw = Path(model_arg).expanduser()
    if raw.is_absolute():
        return str(raw.resolve())
    if raw.parent != Path("."):
        return str((repo_root / raw).resolve())

    # 当文件名在仓库根目录存在时，优先视为本地文件；否则按官方模型名处理。
    local = (repo_root / raw).resolve()
    if local.exists():
        return str(local)
    return model_arg


def _parse_kv_overrides(items: list[str]) -> dict[str, Any]:
    """解析 `KEY=VALUE` 覆盖项，尽量贴近官方 CLI 体验。"""
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --override '{item}', expected KEY=VALUE.")
        key, value_text = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --override '{item}', empty KEY is not allowed.")
        overrides[key] = _parse_literal_value(value_text.strip())
    return overrides


def _parse_literal_value(text: str) -> Any:
    """将字符串解析为 Python 值（bool/int/float/list/str）。"""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _resolve_device(device: str | None, multi_gpu: bool) -> str | None:
    """解析 device 参数。

    规则：
    1) 显式传入 --device 时直接使用；
    2) 否则仅在 --multi-gpu 时自动拼接所有 CUDA 设备；
    3) 其它情况返回 None，交给 Ultralytics 默认策略。
    """
    if device:
        return device
    if not multi_gpu:
        return None
    if not torch.cuda.is_available():
        return "cpu"
    count = torch.cuda.device_count()
    if count <= 1:
        return "0"
    return ",".join(str(i) for i in range(count))


def _build_train_kwargs(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    """构建传给 `YOLO.train()` 的参数字典。"""
    data_path = _resolve_repo_path(repo_root, args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    kwargs: dict[str, Any] = {
        "data": str(data_path),
        "epochs": max(1, int(args.epochs)),
        "imgsz": max(32, int(args.imgsz)),
    }

    if args.cfg:
        cfg_path = _resolve_repo_path(repo_root, args.cfg)
        if not cfg_path.exists():
            raise FileNotFoundError(f"cfg yaml not found: {cfg_path}")
        kwargs["cfg"] = str(cfg_path)

    resolved_device = _resolve_device(device=args.device, multi_gpu=bool(args.multi_gpu))
    if resolved_device is not None:
        kwargs["device"] = resolved_device

    if args.batch is not None:
        kwargs["batch"] = max(1, int(args.batch))
    if args.workers is not None:
        kwargs["workers"] = max(0, int(args.workers))
    if args.cache is not None:
        kwargs["cache"] = bool(args.cache)
    if args.amp is not None:
        kwargs["amp"] = bool(args.amp)
    if args.project:
        kwargs["project"] = str(_resolve_repo_path(repo_root, args.project))
    if args.name:
        kwargs["name"] = args.name
    if args.exist_ok is not None:
        kwargs["exist_ok"] = bool(args.exist_ok)

    # 最后应用 KEY=VALUE 覆盖，行为与官方 arg=value 一致（后者优先级更高）。
    kwargs.update(_parse_kv_overrides(args.override))
    return kwargs


def _format_cli_preview(model_arg: str, train_kwargs: dict[str, Any]) -> str:
    """输出等价的官方 CLI 预览，便于核对流程一致性。"""
    ordered_keys = [
        "data",
        "model",
        "epochs",
        "imgsz",
        "cfg",
        "device",
        "batch",
        "workers",
        "cache",
        "amp",
        "project",
        "name",
        "exist_ok",
    ]
    preview_pairs: dict[str, Any] = {"model": model_arg, **train_kwargs}
    parts = ["yolo", "segment", "train"]
    for key in ordered_keys:
        if key not in preview_pairs:
            continue
        value = preview_pairs[key]
        if isinstance(value, str) and " " in value:
            parts.append(f'{key}="{value}"')
        else:
            parts.append(f"{key}={value}")

    remaining = [key for key in preview_pairs if key not in ordered_keys]
    for key in sorted(remaining):
        value = preview_pairs[key]
        if isinstance(value, str) and " " in value:
            parts.append(f'{key}="{value}"')
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def main() -> int:
    """脚本主入口。"""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    model_arg = _resolve_model_arg(repo_root, args.model)
    train_kwargs = _build_train_kwargs(args, repo_root)

    print("Official-equivalent CLI preview:")
    print(_format_cli_preview(model_arg=model_arg, train_kwargs=train_kwargs))

    model = YOLO(model_arg)
    results = model.train(**train_kwargs)

    if args.no_copy_best:
        print(f"Training finished. Run directory: {results.save_dir}")
        return 0

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"best.pt not found at: {best_path}")

    out_path = _resolve_repo_path(repo_root, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, out_path)
    print(f"Saved best weight to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
