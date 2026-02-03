from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from pathlib import Path

from ultralytics.utils.downloads import attempt_download_asset


@dataclass(frozen=True)
class AppConfig:
    pose_model: str
    det_model: str
    source: str
    conf: float
    imgsz: int
    device: str | None
    output_interval: float
    keypoint_conf: float
    contact_expand: int
    no_preview: bool


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def models_dir() -> Path:
    return project_root() / "models"


def output_dir() -> Path:
    out = project_root() / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


def log_dir() -> Path:
    logs = output_dir() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return logs


def is_windows() -> bool:
    return platform.system().lower() == "windows"


def default_source() -> str:
    return "0" if is_windows() else "/dev/video0"


def coerce_source(value: str) -> int | str:
    value = value.strip()
    return int(value) if value.isdigit() else value


def resolve_model_path(model_arg: str) -> Path:
    model_path = Path(model_arg).expanduser()
    if not model_path.is_absolute():
        model_path = (project_root() / model_path).resolve()
    return model_path


def ensure_model_path(model_path: Path) -> Path:
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(attempt_download_asset(model_path))
    if not downloaded.exists():
        raise FileNotFoundError(
            "Model not found after download attempt. "
            "Place weights under models/ or pass --pose-model/--det-model."
        )
    return downloaded


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
