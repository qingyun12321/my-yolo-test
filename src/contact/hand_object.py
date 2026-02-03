from __future__ import annotations

import numpy as np

from ..types import ContactResult, DetectedObject, Keypoints


def detect_contact(
    keypoints: Keypoints,
    objects: list[DetectedObject],
    min_conf: float,
    expand: int,
) -> ContactResult:
    wrists = []
    left = _get_keypoint(keypoints, 9, min_conf)
    right = _get_keypoint(keypoints, 10, min_conf)
    if left:
        wrists.append(("left_wrist", left))
    if right:
        wrists.append(("right_wrist", right))

    best: tuple[str, DetectedObject] | None = None
    for wrist_name, point in wrists:
        for obj in objects:
            if _point_in_object(point, obj, expand):
                if best is None or obj.score > best[1].score:
                    best = (wrist_name, obj)

    if best is None:
        return ContactResult(active=False, object_name=None, object_score=None, wrist_name=None)

    wrist_name, obj = best
    return ContactResult(
        active=True,
        object_name=obj.name,
        object_score=obj.score,
        wrist_name=wrist_name,
    )


def _get_keypoint(
    keypoints: Keypoints,
    index: int,
    min_conf: float,
) -> tuple[float, float] | None:
    if index >= len(keypoints):
        return None
    x, y, conf = keypoints[index]
    if conf < min_conf:
        return None
    return (x, y)


def _point_in_object(
    point: tuple[float, float],
    obj: DetectedObject,
    expand: int,
) -> bool:
    if obj.mask is not None:
        return _point_in_mask(point, obj.mask)
    return _point_in_bbox(point, obj.bbox, expand)


def _point_in_bbox(
    point: tuple[float, float],
    bbox: tuple[float, float, float, float],
    expand: int,
) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return (x1 - expand) <= x <= (x2 + expand) and (y1 - expand) <= y <= (y2 + expand)


def _point_in_mask(point: tuple[float, float], mask: np.ndarray) -> bool:
    x, y = point
    if mask.ndim != 2:
        return False
    h, w = mask.shape
    xi = int(round(x))
    yi = int(round(y))
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return False
    return bool(mask[yi, xi])
