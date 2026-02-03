from __future__ import annotations

from typing import Iterable

from ..types import Keypoints


COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
KEYPOINT_INDEX = {name: idx for idx, name in enumerate(COCO_KEYPOINTS)}


def get_keypoint(keypoints: Keypoints, name: str) -> tuple[float, float, float] | None:
    idx = KEYPOINT_INDEX.get(name)
    if idx is None or idx >= len(keypoints):
        return None
    return keypoints[idx]


def _valid_points(points: Iterable[tuple[float, float, float]], min_conf: float) -> list[tuple[float, float, float]]:
    return [point for point in points if point[2] >= min_conf]


def torso_center(keypoints: Keypoints, min_conf: float) -> tuple[float, float] | None:
    left_shoulder = get_keypoint(keypoints, "left_shoulder")
    right_shoulder = get_keypoint(keypoints, "right_shoulder")
    left_hip = get_keypoint(keypoints, "left_hip")
    right_hip = get_keypoint(keypoints, "right_hip")

    candidates: list[tuple[float, float, float]] = []
    for point in (left_shoulder, right_shoulder, left_hip, right_hip):
        if point is not None:
            candidates.append(point)

    valid = _valid_points(candidates, min_conf)
    if valid:
        xs = [p[0] for p in valid]
        ys = [p[1] for p in valid]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    nose = get_keypoint(keypoints, "nose")
    if nose and nose[2] >= min_conf:
        return nose[0], nose[1]

    return None


def torso_scale(keypoints: Keypoints, min_conf: float) -> float:
    left_shoulder = get_keypoint(keypoints, "left_shoulder")
    right_shoulder = get_keypoint(keypoints, "right_shoulder")
    left_hip = get_keypoint(keypoints, "left_hip")
    right_hip = get_keypoint(keypoints, "right_hip")

    distances: list[float] = []
    if left_shoulder and right_shoulder:
        if left_shoulder[2] >= min_conf and right_shoulder[2] >= min_conf:
            distances.append(abs(left_shoulder[0] - right_shoulder[0]))
    if left_hip and right_hip:
        if left_hip[2] >= min_conf and right_hip[2] >= min_conf:
            distances.append(abs(left_hip[0] - right_hip[0]))

    if distances:
        return max(distances)

    return 1.0


def normalize_keypoints(
    keypoints: Keypoints,
    min_conf: float,
) -> dict[str, tuple[float, float, float]]:
    center = torso_center(keypoints, min_conf)
    scale = torso_scale(keypoints, min_conf)
    if center is None:
        return {}

    normalized: dict[str, tuple[float, float, float]] = {}
    for name in COCO_KEYPOINTS:
        point = get_keypoint(keypoints, name)
        if point is None:
            continue
        x, y, conf = point
        normalized[name] = ((x - center[0]) / scale, (y - center[1]) / scale, conf)
    return normalized
