from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset


def parse_args() -> argparse.Namespace:
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
        default="0",
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
    value = value.strip()
    return int(value) if value.isdigit() else value


def build_output_path(output_arg: str | None) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return output_dir / f"webcam_detect_{timestamp}.mp4"


def resolve_model_path(model_arg: str, project_root: Path) -> Path:
    model_path = Path(model_arg).expanduser()
    if not model_path.is_absolute():
        model_path = (project_root / model_path).resolve()
    return model_path


def ensure_model_path(model_path: Path) -> Path:
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(attempt_download_asset(model_path))
    if not downloaded.exists():
        raise FileNotFoundError(f"Model not found after download attempt: {model_path}")
    return downloaded


def main() -> int:
    args = parse_args()
    source = coerce_source(args.source)
    project_root = Path(__file__).resolve().parents[1]
    output_path = build_output_path(args.output)
    model_path = ensure_model_path(resolve_model_path(args.model, project_root))

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open webcam source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 30.0

    ok, frame = cap.read()
    if not ok:
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
            results = model.predict(
                frame,
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
            if not ok:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        writer.release()
        if show_preview:
            cv2.destroyAllWindows()

    print(f"Saved output to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
