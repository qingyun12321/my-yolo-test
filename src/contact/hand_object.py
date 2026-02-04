from __future__ import annotations

"""手与物体接触判定。

策略：
1) 提取左右手腕关键点
2) 判断手腕是否落在物体 bbox 或 mask 中
3) 若同时命中多个物体，选择置信度最高者
"""

import numpy as np

from ..types import ContactResult, DetectedObject, Keypoints


def detect_contact(
    keypoints: Keypoints,
    objects: list[DetectedObject],
    min_conf: float,
    expand: int,
) -> ContactResult:
    """检测手腕是否与任意物体接触。

    参数:
        keypoints: 姿态关键点列表（x, y, conf）。
        objects: 检测到的物体列表。
        min_conf: 手腕关键点最低置信度阈值，范围 [0.0, 1.0]。
        expand: bbox 扩张像素，>=0；值越大判定越宽松。

    返回:
        ContactResult: 接触判定结果。
    """
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
    """获取指定索引的关键点（仅返回坐标）。

    参数:
        keypoints: 关键点列表。
        index: 关键点索引（COCO：左手腕=9，右手腕=10）。
        min_conf: 置信度阈值，范围 [0.0, 1.0]。

    返回:
        tuple[float, float] | None: 关键点坐标或 None。
    """
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
    """判断点是否落在物体范围内（mask 或 bbox）。

    参数:
        point: (x, y) 像素坐标。
        obj: 物体检测结果。
        expand: bbox 扩张像素，>=0。
    """
    if obj.mask is not None:
        return _point_in_mask(point, obj.mask)
    return _point_in_bbox(point, obj.bbox, expand)


def _point_in_bbox(
    point: tuple[float, float],
    bbox: tuple[float, float, float, float],
    expand: int,
) -> bool:
    """判断点是否落在 bbox 内（支持像素扩张）。

    参数:
        point: (x, y) 像素坐标。
        bbox: (x1, y1, x2, y2) 边框。
        expand: bbox 扩张像素，>=0。
    """
    x, y = point
    x1, y1, x2, y2 = bbox
    return (x1 - expand) <= x <= (x2 + expand) and (y1 - expand) <= y <= (y2 + expand)


def _point_in_mask(point: tuple[float, float], mask: np.ndarray) -> bool:
    """判断点是否落在分割 mask 内（不做扩张）。

    参数:
        point: (x, y) 像素坐标。
        mask: 二值 mask，形状为 (H, W)。
    """
    x, y = point
    if mask.ndim != 2:
        return False
    h, w = mask.shape
    xi = int(round(x))
    yi = int(round(y))
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return False
    return bool(mask[yi, xi])
