from __future__ import annotations

"""示例：YOLO 目标检测（摄像头输入，输出 mp4）。"""

import argparse
import platform
import time
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Run YOLO object detection on a webcam and save output as mp4."
    )
    parser.add_argument(
        "--model",
        default="models/yolo26n.pt",
        help="Model path or name (default: models/yolo26n.pt).",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Webcam index (e.g. 0) or device path (e.g. /dev/video0).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (e.g. 0, 0,1, or cpu).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output mp4 path. Default: output/webcam_detect_<timestamp>.mp4",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the live preview window.",
    )
    return parser.parse_args()


def coerce_source(value: str) -> int | str:
    """将来源字符串转换为 int 或保持原样。

    参数:
        value: 摄像头来源字符串（如 "0" 或 "/dev/video0"）。

    返回:
        int | str: 转换后的来源。
    """
    value = value.strip()
    return int(value) if value.isdigit() else value


def build_output_path(output_arg: str | None) -> Path:
    """构建输出文件路径。

    参数:
        output_arg: 用户指定的输出路径（可为空，None 表示自动生成）。
    """
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return output_dir / f"webcam_detect_{timestamp}.mp4"


def resolve_model_arg(model_arg: str, project_root: Path) -> Path | str:
    """解析模型参数为路径或模型名。

    参数:
        model_arg: 模型路径或模型名。
        project_root: 项目根目录。
    """
    model_arg = model_arg.strip()
    model_path = Path(model_arg).expanduser()

    if model_path.is_absolute():
        return model_path

    if model_path.exists():
        return model_path.resolve()

    if any(sep in model_arg for sep in ("/", "\\")):
        return (project_root / model_path).resolve()

    return model_arg


def prepare_model_arg(model_arg: str, project_root: Path) -> Path | str:
    """准备模型参数，必要时下载权重。

    参数:
        model_arg: 模型路径或模型名。
        project_root: 项目根目录。
    """
    resolved = resolve_model_arg(model_arg, project_root)
    if isinstance(resolved, str):
        return resolved

    if resolved.exists():
        return resolved

    resolved.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(attempt_download_asset(resolved, release="latest"))
    if not downloaded.exists():
        raise FileNotFoundError(f"Model not found after download attempt: {resolved}")
    return downloaded


def is_windows() -> bool:
    """判断是否为 Windows。"""
    return platform.system().lower() == "windows"


def default_source() -> str:
    """默认摄像头来源（Windows 下为 '0'）。"""
    return "0"


def backend_candidates() -> list[tuple[str, int]]:
    """返回推荐的视频后端列表。"""
    if is_windows():
        candidates = []
        if hasattr(cv2, "CAP_DSHOW"):
            candidates.append(("DSHOW", cv2.CAP_DSHOW))
        if hasattr(cv2, "CAP_MSMF"):
            candidates.append(("MSMF", cv2.CAP_MSMF))
        candidates.append(("ANY", cv2.CAP_ANY))
        return candidates
    return [("ANY", cv2.CAP_ANY)]


def open_capture(source: int | str) -> tuple[cv2.VideoCapture | None, str]:
    """打开摄像头或视频流。

    参数:
        source: 摄像头索引（int）或设备路径/URL（str）。
    """
    last_backend = "ANY"
    for name, backend in backend_candidates():
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened():
            return cap, name
        cap.release()
        last_backend = name
    return None, last_backend


def main() -> int:
    """示例主流程。

    返回:
        int: 0 表示正常退出。
    """
    args = parse_args()
    source_arg = args.source if args.source is not None else default_source()
    source = coerce_source(source_arg)
    project_root = Path(__file__).resolve().parents[1]
    output_path = build_output_path(args.output)
    model_arg = prepare_model_arg(args.model, project_root)

    model = YOLO(model_arg)
    cap, backend_name = open_capture(source)
    if not cap:
        raise RuntimeError(f"Unable to open webcam source: {source_arg}")

    target_fps = 60.0
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    fps = target_fps

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError("Unable to read from webcam source.")

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open video writer at: {output_path}")

    show_preview = not args.no_preview

    try:
        while True:
            # 逐帧推理
            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            annotated = results[0].plot()
            writer.write(annotated)

            if show_preview:
                cv2.imshow("YOLO Webcam Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            ok, frame = cap.read()
            if not ok or frame is None:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        writer.release()
        if show_preview:
            cv2.destroyAllWindows()

    print(f"Saved output to: {output_path} (backend: {backend_name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
