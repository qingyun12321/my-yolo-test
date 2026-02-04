from __future__ import annotations

"""应用级配置与路径/模型处理工具。

包含：
- 项目路径与输出路径管理
- 默认摄像头来源设置
- 模型参数解析（路径或模型名）
- 模型权重下载与校验
"""

import platform
import time
from dataclasses import dataclass
from pathlib import Path

from ultralytics.utils.downloads import attempt_download_asset


@dataclass(frozen=True)
class AppConfig:
    """应用配置结构体（可用于将 CLI 参数固化为配置对象）。

    字段说明：
        pose_model: 姿态模型路径或模型名，例如 "models/yolo26n-pose.pt" 或 "yolo26n-pose.pt"。
        det_model: 目标检测/分割模型路径或模型名，例如 "models/yolo26n-seg.pt" 或 "yolo26n-seg.pt"。
        hand_model: 手部关键点模型路径（通常是训练得到的 best.pt）。
        source: 摄像头来源（字符串形式），如 "0" 或 "/dev/video0"。
        conf: 置信度阈值，范围 [0.0, 1.0]。
        imgsz: 推理输入尺寸（正整数，如 640）。
        hand_conf: 手部关键点置信度阈值，范围 [0.0, 1.0]。
        hand_imgsz: 手部模型推理尺寸（正整数，如 640）。
        hand_scale: 手部裁剪尺寸占人体 bbox 高度的比例（0.1-1.0）。
        device: 推理设备，如 "cpu"、"0"、"0,1" 等，None 表示由库自动选择。
        output_interval: 输出节流间隔（秒），0 表示每帧输出。
        keypoint_conf: 关键点置信度阈值，范围 [0.0, 1.0]。
        contact_expand: 手与物体接触判断时对 bbox 的像素扩张量（>=0）。
        no_preview: 是否禁用可视化窗口。
    """
    pose_model: str
    det_model: str
    hand_model: str
    source: str
    conf: float
    imgsz: int
    hand_conf: float
    hand_imgsz: int
    hand_scale: float
    device: str | None
    output_interval: float
    keypoint_conf: float
    contact_expand: int
    no_preview: bool


def project_root() -> Path:
    """返回项目根目录路径（src 的上一级）。

    返回:
        Path: 项目根目录路径。
    """
    return Path(__file__).resolve().parents[1]


def models_dir() -> Path:
    """返回模型目录（models/）。

    返回:
        Path: 模型目录路径。
    """
    return project_root() / "models"


def output_dir() -> Path:
    """返回输出目录（output/），如不存在会自动创建。

    返回:
        Path: 输出目录路径。
    """
    out = project_root() / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


def log_dir() -> Path:
    """返回日志目录（output/logs/），如不存在会自动创建。

    返回:
        Path: 日志目录路径。
    """
    logs = output_dir() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return logs


def is_windows() -> bool:
    """判断当前系统是否为 Windows。

    返回:
        bool: True 表示 Windows，否则为 False。
    """
    return platform.system().lower() == "windows"


def default_source() -> str:
    """返回默认摄像头来源。

    Windows 使用字符串 "0"，其他系统使用 "/dev/video0"。

    返回:
        str: 摄像头来源字符串。
    """
    return "0" if is_windows() else "/dev/video0"


def coerce_source(value: str) -> int | str:
    """将来源字符串转换为 int 或保持为字符串。

    说明：
        仅当字符串为纯数字时转为 int（例如 "0" -> 0），
        否则保持原样（例如 "/dev/video0"）。

    参数:
        value: 摄像头来源字符串。

    返回:
        int | str: 转换后的来源。
    """
    value = value.strip()
    return int(value) if value.isdigit() else value


def resolve_model_arg(model_arg: str) -> Path | str:
    """解析模型参数，支持路径或模型名。

    规则：
        - 绝对路径：直接返回 Path
        - 相对路径且文件存在：返回解析后的 Path
        - 含路径分隔符但文件不存在：按项目根目录拼接后返回 Path
        - 纯模型名：返回原字符串（交由 Ultralytics 自动解析/下载）

    参数:
        model_arg: 模型路径或模型名。

    返回:
        Path | str: 解析后的路径或模型名字符串。
    """
    model_arg = model_arg.strip()
    model_path = Path(model_arg).expanduser()

    if model_path.is_absolute():
        return model_path

    if model_path.exists():
        return model_path.resolve()

    if any(sep in model_arg for sep in ("/", "\\")):
        return (project_root() / model_path).resolve()

    return model_arg


def prepare_model_arg(model_arg: str) -> Path | str:
    """准备模型参数，必要时触发权重下载。

    行为：
        - 若是模型名字符串，直接返回（Ultralytics 内部会处理下载）
        - 若是路径且存在，直接返回
        - 若是路径且不存在，尝试从官方 release 下载到指定路径

    参数:
        model_arg: 模型路径或模型名。

    返回:
        Path | str: 最终可用于 YOLO 构造的模型参数。

    异常:
        FileNotFoundError: 下载失败或目标文件不存在。
    """
    resolved = resolve_model_arg(model_arg)
    if isinstance(resolved, str):
        return resolved

    if resolved.exists():
        return resolved

    resolved.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(attempt_download_asset(resolved, release="latest"))
    if not downloaded.exists():
        raise FileNotFoundError(
            "Model not found after download attempt. "
            "Place weights under models/ or pass --pose-model/--det-model."
        )
    return downloaded


def timestamp() -> str:
    """生成当前时间戳（YYYYmmdd_HHMMSS）。

    返回:
        str: 时间戳字符串。
    """
    return time.strftime("%Y%m%d_%H%M%S")
