from __future__ import annotations

from dataclasses import dataclass
from typing import Any


BBox = tuple[float, float, float, float]
Keypoints = list[tuple[float, float, float]]


@dataclass(frozen=True)
class PoseDetection:
    bbox: BBox
    keypoints: Keypoints
    score: float


@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    bbox: BBox
    keypoints: Keypoints
    score: float


@dataclass(frozen=True)
class DetectedObject:
    name: str
    bbox: BBox
    score: float
    mask: Any | None


@dataclass(frozen=True)
class ActionResult:
    action: str
    confidence: float


@dataclass(frozen=True)
class ContactResult:
    active: bool
    object_name: str | None
    object_score: float | None
    wrist_name: str | None
