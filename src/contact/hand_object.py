from __future__ import annotations

"""手与物体接触判定。

本模块针对真实场景中的遮挡问题做了增强：
1) 接触点不只使用指尖，还使用掌心、掌根、MCP 关节与指骨采样点；
2) 对候选接触点做空间去重，减少近邻点重复计数；
3) 手关键点缺失时回退到人体手腕，保证最小可用能力；
4) mask/bbox 联合距离判定，兼容分割质量波动。
"""

import cv2
import numpy as np

from ..types import ContactResult, DetectedObject, Keypoints


def detect_contact(
    keypoints: Keypoints,
    objects: list[DetectedObject],
    min_conf: float,
    expand: int,
    hand_keypoints: dict[str, list[tuple[float, float, float]] | None] | None = None,
    dist_threshold: float = 12.0,
    min_points: int = 2,
) -> ContactResult:
    """检测手与物体是否接触（鲁棒版）。

    策略：
        - 优先使用手部关键点（指尖 + 手掌中心）
        - 若手部关键点缺失，则回退使用人体手腕关键点
        - 使用“点到物体距离”判定，并支持 mask 优先
        - 需要达到一定数量的关键点接近才判定接触

    参数:
        keypoints: 人体姿态关键点列表（x, y, conf）。
        objects: 检测到的物体列表。
        min_conf: 关键点最低置信度阈值，范围 [0.0, 1.0]。
        expand: bbox 扩张像素，>=0；值越大判定越宽松。
        hand_keypoints: {"left": kps|None, "right": kps|None}。
        dist_threshold: 点到物体的距离阈值（像素），<=该值视为接触。
        min_points: 最少命中点数（鲁棒性更高，建议 2）。

    返回:
        ContactResult: 接触判定结果。
    """
    candidates: list[tuple[str, tuple[float, float]]] = []

    # 1) 先尝试使用手部关键点（多点位增强，适应遮挡场景）
    if hand_keypoints:
        for side in ("left", "right"):
            kps = hand_keypoints.get(side)
            if not kps:
                continue
            points = _select_hand_points(kps, min_conf)
            for name, point in points:
                candidates.append((f"{side}_{name}", point))

    # 2) 若手部关键点缺失，则回退使用人体手腕
    if not candidates:
        left = _get_keypoint(keypoints, 9, min_conf)
        right = _get_keypoint(keypoints, 10, min_conf)
        if left:
            candidates.append(("left_wrist", left))
        if right:
            candidates.append(("right_wrist", right))

    candidates = _dedupe_candidate_points(candidates, min_distance=6.0)

    if not candidates or not objects:
        return ContactResult(active=False, object_name=None, object_score=None, wrist_name=None)

    # 3) 对每个物体统计命中点数与最短距离
    best: tuple[DetectedObject, int, float, str] | None = None
    for obj in objects:
        hit_points = 0
        min_dist = float("inf")
        best_point_name = ""
        for name, point in candidates:
            dist = _distance_to_object(point, obj, expand)
            if dist <= dist_threshold:
                hit_points += 1
                if dist < min_dist:
                    min_dist = dist
                    best_point_name = name
            else:
                min_dist = min(min_dist, dist)

        required = min_points if len(candidates) >= min_points else 1
        if hit_points >= required:
            # 选择命中点更多、距离更近的物体
            if best is None or hit_points > best[1] or (
                hit_points == best[1] and min_dist < best[2]
            ):
                best = (obj, hit_points, min_dist, best_point_name)

    if best is None:
        return ContactResult(active=False, object_name=None, object_score=None, wrist_name=None)

    obj, _, _, point_name = best
    return ContactResult(
        active=True,
        object_name=obj.name,
        object_score=obj.score,
        wrist_name=point_name,
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


def _distance_to_object(
    point: tuple[float, float],
    obj: DetectedObject,
    expand: int,
) -> float:
    """计算点到物体的最小距离（像素）。

    - 若在 mask 内（含扩张），距离为 0
    - 否则使用 bbox 距离作为近似
    """
    if obj.mask is not None:
        if _point_in_mask_with_expand(point, obj.mask, expand):
            return 0.0
    return _distance_to_bbox(point, obj.bbox, expand)


def _distance_to_bbox(
    point: tuple[float, float],
    bbox: tuple[float, float, float, float],
    expand: int,
) -> float:
    """点到 bbox 的距离（支持扩张）。"""
    x, y = point
    x1, y1, x2, y2 = bbox
    x1 -= expand
    y1 -= expand
    x2 += expand
    y2 += expand
    dx = 0.0
    if x < x1:
        dx = x1 - x
    elif x > x2:
        dx = x - x2
    dy = 0.0
    if y < y1:
        dy = y1 - y
    elif y > y2:
        dy = y - y2
    return float((dx * dx + dy * dy) ** 0.5)


def _point_in_mask_with_expand(
    point: tuple[float, float],
    mask: np.ndarray,
    expand: int,
) -> bool:
    """判断点是否在 mask 内（支持像素扩张）。"""
    if expand <= 0:
        return _point_in_mask(point, mask)

    mask_uint8 = (mask > 0.5).astype("uint8") * 255
    kernel = np.ones((expand * 2 + 1, expand * 2 + 1), dtype=np.uint8)
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    return _point_in_mask(point, dilated)


def _select_hand_points(
    kps: list[tuple[float, float, float]],
    min_conf: float,
) -> list[tuple[str, tuple[float, float]]]:
    """挑选用于接触判定的手部点位（遮挡鲁棒版）。

    选点策略：
    1) 指尖：覆盖“指尖点触”场景；
    2) MCP（掌指关节）+ 手腕：覆盖“手掌抓握”场景；
    3) 掌心中心：覆盖“物体贴掌心”场景；
    4) 指骨采样点：覆盖“指尖被挡住但指节仍可见”场景。
    """
    points: list[tuple[str, tuple[float, float]]] = []

    tip_ids = [
        ("thumb_tip", 4),
        ("index_tip", 8),
        ("middle_tip", 12),
        ("ring_tip", 16),
        ("pinky_tip", 20),
    ]
    mcp_ids = [
        ("thumb_mcp", 2),
        ("index_mcp", 5),
        ("middle_mcp", 9),
        ("ring_mcp", 13),
        ("pinky_mcp", 17),
    ]
    core_ids = [
        ("wrist", 0),
    ]

    for name, idx in tip_ids + mcp_ids + core_ids:
        point = _get_hand_point(kps, idx, min_conf)
        if point is not None:
            points.append((name, point))

    # 掌心中心：腕点 + 四个 MCP 的平均，适合抓握型接触。
    palm_anchor_ids = [0, 5, 9, 13, 17]
    palm_points = [pt for idx in palm_anchor_ids if (pt := _get_hand_point(kps, idx, min_conf)) is not None]
    if palm_points:
        px = sum(point[0] for point in palm_points) / len(palm_points)
        py = sum(point[1] for point in palm_points) / len(palm_points)
        points.append(("palm_center", (px, py)))

    # 指骨采样：在 MCP->TIP 连线上取两个样本点，提高遮挡下的命中率。
    shaft_pairs = [
        ("index_shaft", 5, 8),
        ("middle_shaft", 9, 12),
        ("ring_shaft", 13, 16),
    ]
    for name, mcp_idx, tip_idx in shaft_pairs:
        mcp_pt = _get_hand_point(kps, mcp_idx, min_conf)
        tip_pt = _get_hand_point(kps, tip_idx, min_conf)
        if mcp_pt is None or tip_pt is None:
            continue
        for ratio in (0.35, 0.65):
            sx = float(mcp_pt[0] * (1.0 - ratio) + tip_pt[0] * ratio)
            sy = float(mcp_pt[1] * (1.0 - ratio) + tip_pt[1] * ratio)
            points.append((f"{name}_{int(ratio * 100)}", (sx, sy)))

    return points


def _get_hand_point(
    kps: list[tuple[float, float, float]],
    index: int,
    min_conf: float,
) -> tuple[float, float] | None:
    """从手关键点序列中安全读取单点坐标。"""
    if index < 0 or index >= len(kps):
        return None
    x, y, conf = kps[index]
    if conf < min_conf:
        return None
    return (float(x), float(y))


def _dedupe_candidate_points(
    points: list[tuple[str, tuple[float, float]]],
    min_distance: float,
) -> list[tuple[str, tuple[float, float]]]:
    """按空间距离去重候选点，避免近邻点重复加分。

    说明：
        手部点位会同时包含 TIP/MCP/掌心/采样点，若全部直接计数，
        在某些角度下会出现多个点几乎重合，导致命中数被“虚高”。
        这里按最小距离做一次抑制，保留先到先得的代表点。
    """
    if not points:
        return []
    if min_distance <= 0:
        return points

    filtered: list[tuple[str, tuple[float, float]]] = []
    min_dist2 = float(min_distance * min_distance)
    for name, point in points:
        duplicate = False
        for _, kept in filtered:
            dx = point[0] - kept[0]
            dy = point[1] - kept[1]
            if (dx * dx + dy * dy) <= min_dist2:
                duplicate = True
                break
        if not duplicate:
            filtered.append((name, point))
    return filtered
