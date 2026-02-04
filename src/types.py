from __future__ import annotations

"""全局数据结构与类型定义。"""

from dataclasses import dataclass
from typing import Any


# BBox: (x1, y1, x2, y2)，单位为像素
BBox = tuple[float, float, float, float]
# Keypoints: [(x, y, conf), ...]，conf 范围 [0.0, 1.0]
Keypoints = list[tuple[float, float, float]]
# HandKeypoints: 21 个手部关键点列表（wrist + 4*5 指节）
HandKeypoints = list[tuple[float, float, float]]


@dataclass(frozen=True)
class PoseDetection:
    """单帧姿态检测结果（未追踪）。"""
    bbox: BBox
    keypoints: Keypoints
    score: float


@dataclass(frozen=True)
class TrackedPerson:
    """追踪后的人员信息。"""
    track_id: int
    bbox: BBox
    keypoints: Keypoints
    score: float


@dataclass(frozen=True)
class DetectedObject:
    """检测到的物体信息。"""
    name: str
    bbox: BBox
    score: float
    mask: Any | None


@dataclass(frozen=True)
class ActionResult:
    """动作识别结果。"""
    action: str
    confidence: float


@dataclass(frozen=True)
class ContactResult:
    """接触检测结果。"""
    active: bool
    object_name: str | None
    object_score: float | None
    wrist_name: str | None
