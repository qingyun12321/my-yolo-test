from __future__ import annotations

"""目标后处理工具：去重与跨类别冲突抑制。"""

from ..types import BBox, DetectedObject


def deduplicate_objects(
    objects: list[DetectedObject],
    iou_threshold: float,
    center_ratio: float,
    conflict_suppress: bool,
    conflict_overlap: float,
    conflict_area_ratio: float,
    conflict_score_gap: float,
) -> list[DetectedObject]:
    """对检测结果做两类后处理。

    1. 同类别去重：处理同一物体被重复框出的情况；
    2. 跨类别冲突抑制：处理“整体物体 + 局部误分类”同时出现的情况。
    """
    if not objects:
        return []

    iou_threshold = float(max(0.0, min(iou_threshold, 1.0)))
    center_ratio = float(max(0.0, center_ratio))
    conflict_overlap = float(max(0.0, min(conflict_overlap, 1.0)))
    conflict_area_ratio = float(max(1.0, conflict_area_ratio))
    conflict_score_gap = float(max(0.0, conflict_score_gap))

    ranked = sorted(objects, key=_object_quality, reverse=True)
    unique: list[DetectedObject] = []
    for candidate in ranked:
        dropped = False
        idx = 0
        while idx < len(unique):
            kept = unique[idx]

            if candidate.name == kept.name and _is_duplicate_object(
                candidate.bbox,
                kept.bbox,
                iou_threshold=iou_threshold,
                center_ratio=center_ratio,
            ):
                better = _prefer_same_class(candidate, kept)
                if better is kept:
                    dropped = True
                    break
                unique.pop(idx)
                continue

            if (
                conflict_suppress
                and candidate.name != kept.name
                and _is_cross_class_conflict(
                    candidate.bbox,
                    kept.bbox,
                    overlap_threshold=conflict_overlap,
                    area_ratio_threshold=conflict_area_ratio,
                )
            ):
                better = _prefer_cross_class(
                    candidate,
                    kept,
                    score_gap=conflict_score_gap,
                )
                if better is kept:
                    dropped = True
                    break
                unique.pop(idx)
                continue

            idx += 1

        if not dropped:
            unique.append(candidate)
    return unique


def _object_quality(obj: DetectedObject) -> tuple[int, float, float]:
    """定义目标质量排序：优先 mask，其次分数，再次面积。"""
    return (
        1 if obj.mask is not None else 0,
        float(obj.score),
        _bbox_area(obj.bbox),
    )


def _prefer_same_class(a: DetectedObject, b: DetectedObject) -> DetectedObject:
    """同类别冲突时，保留质量更高者。"""
    return a if _object_quality(a) >= _object_quality(b) else b


def _prefer_cross_class(
    a: DetectedObject,
    b: DetectedObject,
    score_gap: float,
) -> DetectedObject:
    """跨类别冲突时的保留策略。"""
    if abs(a.score - b.score) >= score_gap:
        return a if a.score > b.score else b

    area_a = _bbox_area(a.bbox)
    area_b = _bbox_area(b.bbox)
    if area_a > area_b * 1.15:
        return a
    if area_b > area_a * 1.15:
        return b

    return a if _object_quality(a) >= _object_quality(b) else b


def _is_cross_class_conflict(
    bbox_a: BBox,
    bbox_b: BBox,
    overlap_threshold: float,
    area_ratio_threshold: float,
) -> bool:
    """判断两个不同类别目标是否构成“嵌套冲突”。"""
    area_a = _bbox_area(bbox_a)
    area_b = _bbox_area(bbox_b)
    if area_a <= 1e-6 or area_b <= 1e-6:
        return False

    smaller = min(area_a, area_b)
    larger = max(area_a, area_b)
    if (larger / smaller) < area_ratio_threshold:
        return False

    inter = _bbox_intersection_area(bbox_a, bbox_b)
    if inter <= 0:
        return False

    overlap_small = inter / smaller
    if overlap_small < overlap_threshold:
        return False

    small_bbox = bbox_a if area_a <= area_b else bbox_b
    large_bbox = bbox_b if area_a <= area_b else bbox_a
    sx, sy = _bbox_center(small_bbox)
    return _point_in_bbox((sx, sy), large_bbox, expand=2.0)


def _is_duplicate_object(
    bbox_a: BBox,
    bbox_b: BBox,
    iou_threshold: float,
    center_ratio: float,
) -> bool:
    """判断两个同类别目标是否应视作重复。"""
    iou = _bbox_iou(bbox_a, bbox_b)
    if iou >= iou_threshold:
        return True

    # 局部重叠兜底：当较小框几乎被较大框包含时，通常是同物体“整体+局部”重复。
    inter = _bbox_intersection_area(bbox_a, bbox_b)
    if inter > 0:
        small = min(_bbox_area(bbox_a), _bbox_area(bbox_b))
        if small > 1e-6 and (inter / small) >= 0.88:
            return True

    if center_ratio <= 0:
        return False

    center_dist = _center_distance(bbox_a, bbox_b)
    min_diag = min(_bbox_diagonal(bbox_a), _bbox_diagonal(bbox_b))
    if min_diag <= 1e-6:
        return False
    if center_dist > center_ratio * min_diag:
        return False

    area_a = _bbox_area(bbox_a)
    area_b = _bbox_area(bbox_b)
    if area_a <= 1e-6 or area_b <= 1e-6:
        return False
    area_ratio = area_a / area_b
    return 0.25 <= area_ratio <= 4.0


def _bbox_iou(bbox_a: BBox, bbox_b: BBox) -> float:
    """计算 IoU。"""
    inter = _bbox_intersection_area(bbox_a, bbox_b)
    if inter <= 0:
        return 0.0
    union = _bbox_area(bbox_a) + _bbox_area(bbox_b) - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _bbox_intersection_area(bbox_a: BBox, bbox_b: BBox) -> float:
    """计算交叠面积。"""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    return float(inter_w * inter_h)


def _bbox_area(bbox: BBox) -> float:
    """计算边框面积。"""
    x1, y1, x2, y2 = bbox
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def _bbox_diagonal(bbox: BBox) -> float:
    """计算边框对角线。"""
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return float((w * w + h * h) ** 0.5)


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    """计算边框中心。"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _center_distance(bbox_a: BBox, bbox_b: BBox) -> float:
    """计算两个边框中心点距离。"""
    return _point_distance(_bbox_center(bbox_a), _bbox_center(bbox_b))


def _point_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    """计算两点欧氏距离。"""
    dx = float(point_a[0] - point_b[0])
    dy = float(point_a[1] - point_b[1])
    return float((dx * dx + dy * dy) ** 0.5)


def _point_in_bbox(point: tuple[float, float], bbox: BBox, expand: float = 0.0) -> bool:
    """判断点是否落在边框内（支持扩张边界）。"""
    x, y = point
    x1, y1, x2, y2 = bbox
    return (x1 - expand) <= x <= (x2 + expand) and (y1 - expand) <= y <= (y2 + expand)
