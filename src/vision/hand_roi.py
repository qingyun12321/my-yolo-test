from __future__ import annotations

"""基于手关键点构建 ROI（强调前向外扩，并保留少量向内补偿）。"""

from dataclasses import dataclass

from ..types import HandKeypoints


@dataclass
class _SideRoiState:
    """单侧手 ROI 的时序状态。"""

    center: tuple[float, float] | None = None
    size: float | None = None
    direction: tuple[float, float] = (0.0, -1.0)
    misses: int = 0


class HandRoiBuilder:
    """手 ROI 构建器。

    设计要点：
    1. 用手关键点估计中心、尺度和“手腕 -> 手指”的前向方向；
    2. 构建单个不对称 ROI：前向外扩较多，后向内扩较少；
    3. 对方向进行时序平滑与翻转保护，减少外扩方向帧间跳变；
    4. 在关键点缺失时短暂保持上一帧 ROI，提升鲁棒性。
    """

    def __init__(
        self,
        padding_ratio: float,
        min_size: int,
        min_size_ratio: float = 0.12,
        context_scale: float = 1.9,
        forward_shift: float = 0.42,
        inward_scale: float = 1.18,
        inward_shift: float = 0.1,
        direction_smooth: float = 0.35,
        merge_iou: float = 0.7,
        hold_frames: int = 1,
        shrink_floor: float = 0.82,
        size_smooth: float = 0.35,
        global_merge_iou: float = 0.78,
        cross_side_merge_ratio: float = 0.45,
    ) -> None:
        self.padding_ratio = float(max(0.0, min(padding_ratio, 1.0)))
        self.min_size = max(32, int(min_size))
        self.min_size_ratio = float(max(0.0, min(min_size_ratio, 1.0)))
        self.context_scale = float(max(1.0, context_scale))
        self.forward_shift = float(max(0.0, forward_shift))
        self.inward_scale = float(max(1.0, inward_scale))
        self.inward_shift = float(max(0.0, inward_shift))
        self.direction_smooth = float(max(0.0, min(direction_smooth, 1.0)))
        self.merge_iou = float(max(0.0, min(merge_iou, 1.0)))
        self.hold_frames = max(0, int(hold_frames))
        self.shrink_floor = float(max(0.0, min(shrink_floor, 1.0)))
        self.size_smooth = float(max(0.0, min(size_smooth, 1.0)))
        self.global_merge_iou = float(max(0.0, min(global_merge_iou, 1.0)))
        self.cross_side_merge_ratio = float(max(0.0, cross_side_merge_ratio))
        self._state = {
            "left": _SideRoiState(),
            "right": _SideRoiState(),
        }

    def reset(self) -> None:
        """重置左右手 ROI 状态。"""
        self._state["left"] = _SideRoiState()
        self._state["right"] = _SideRoiState()

    def build(
        self,
        hands_lr: dict[str, HandKeypoints | None],
        frame_shape: tuple[int, int],
    ) -> list[tuple[int, int, int, int]]:
        """根据左右手关键点构建 ROI 列表。"""
        h, w = frame_shape
        all_rois: list[tuple[int, int, int, int]] = []
        fresh_sides: list[str] = []

        for side in ("left", "right"):
            side_rois = self._build_side_rois(side, hands_lr.get(side), h=h, w=w)
            if hands_lr.get(side):
                fresh_sides.append(side)
            all_rois.extend(side_rois)

        cross_side_overlap = self._is_cross_side_duplicate()
        merge_iou = self.global_merge_iou
        if len(fresh_sides) <= 1 or cross_side_overlap:
            merge_iou = min(merge_iou, 0.62)

        all_rois = _merge_rois(all_rois, merge_iou=merge_iou)

        if (len(fresh_sides) == 1 or cross_side_overlap) and len(all_rois) > 1:
            all_rois = sorted(all_rois, key=_roi_area, reverse=True)[:1]

        return all_rois

    def _build_side_rois(
        self,
        side: str,
        keypoints: HandKeypoints | None,
        h: int,
        w: int,
    ) -> list[tuple[int, int, int, int]]:
        """构建单侧手 ROI（含时序平滑、丢失保持、方向稳定）。"""
        state = self._state[side]

        if keypoints:
            dyn_min = max(self.min_size, int(round(min(h, w) * self.min_size_ratio)))
            center, size, direction = _estimate_hand_roi_seed(
                keypoints,
                padding_ratio=self.padding_ratio,
                min_size=dyn_min,
            )

            if state.size is not None and state.center is not None:
                size = max(size, state.size * self.shrink_floor)
                if self.size_smooth > 0:
                    alpha = self.size_smooth
                    center = (
                        (1.0 - alpha) * state.center[0] + alpha * center[0],
                        (1.0 - alpha) * state.center[1] + alpha * center[1],
                    )
                    size = (1.0 - alpha) * state.size + alpha * size

            if self.direction_smooth > 0:
                direction = _smooth_direction(
                    previous=state.direction,
                    current=direction,
                    alpha=self.direction_smooth,
                )
            else:
                direction = _align_direction(previous=state.direction, current=direction)

            state.center = center
            state.size = size
            state.direction = direction
            state.misses = 0
        else:
            if state.center is None or state.size is None:
                return []
            if state.misses >= self.hold_frames:
                state.center = None
                state.size = None
                state.misses = self.hold_frames + 1
                return []
            state.misses += 1

        roi = _make_asymmetric_roi_from_seed(
            center=state.center,
            size=state.size,
            direction=state.direction,
            context_scale=self.context_scale,
            forward_shift=self.forward_shift,
            inward_scale=self.inward_scale,
            inward_shift=self.inward_shift,
            h=h,
            w=w,
        )
        if roi is None:
            return []
        return [roi]

    def _is_cross_side_duplicate(self) -> bool:
        """判断左右手状态是否近似重叠（同一只手误识别成两只）。"""
        left = self._state["left"]
        right = self._state["right"]
        if left.center is None or right.center is None:
            return False
        if left.size is None or right.size is None:
            return False

        dx = left.center[0] - right.center[0]
        dy = left.center[1] - right.center[1]
        dist = (dx * dx + dy * dy) ** 0.5
        limit = self.cross_side_merge_ratio * min(left.size, right.size)
        return dist <= limit


def _estimate_hand_roi_seed(
    keypoints: HandKeypoints,
    padding_ratio: float,
    min_size: int,
) -> tuple[tuple[float, float], float, tuple[float, float]]:
    """从手关键点估计 ROI 种子参数。"""
    x1, y1, x2, y2 = _robust_bbox(keypoints)
    center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    ref = max(box_w, box_h)
    palm_span = _palm_span(keypoints)
    if palm_span > 1.0:
        ref = max(ref, palm_span * 1.6)

    size = max(float(min_size), ref * (1.0 + 2.0 * padding_ratio))
    direction = _hand_forward_dir(keypoints)
    return center, size, direction


def _robust_bbox(keypoints: HandKeypoints) -> tuple[float, float, float, float]:
    """稳健外接框：偏向稳定关节点并截尾去离群。"""
    preferred = [0, 5, 9, 13, 17, 8, 12, 16, 20, 4]
    pts: list[tuple[float, float]] = []
    for idx in preferred:
        if idx >= len(keypoints):
            continue
        x, y, conf = keypoints[idx]
        if conf <= 0:
            continue
        pts.append((float(x), float(y)))

    if len(pts) < 6:
        pts = [(float(x), float(y)) for x, y, _ in keypoints]

    xs = sorted(p[0] for p in pts)
    ys = sorted(p[1] for p in pts)
    x1, x2 = _trimmed_range(xs, trim_ratio=0.08)
    y1, y2 = _trimmed_range(ys, trim_ratio=0.08)
    return (x1, y1, x2, y2)


def _trimmed_range(values: list[float], trim_ratio: float) -> tuple[float, float]:
    """对坐标序列做截尾范围估计。"""
    if not values:
        return (0.0, 1.0)
    if len(values) <= 4:
        return (float(min(values)), float(max(values)))
    n = len(values)
    lo = int((n - 1) * trim_ratio)
    hi = int((n - 1) * (1.0 - trim_ratio))
    lo = max(0, min(n - 1, lo))
    hi = max(lo, min(n - 1, hi))
    return (float(values[lo]), float(values[hi]))


def _palm_span(keypoints: HandKeypoints) -> float:
    """估计掌心尺度，用于 ROI 尺寸兜底。"""
    pairs = [(0, 5), (0, 9), (0, 13), (0, 17), (5, 17)]
    dists: list[float] = []
    for i, j in pairs:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        xi, yi, ci = keypoints[i]
        xj, yj, cj = keypoints[j]
        if ci <= 0 or cj <= 0:
            continue
        dx = float(xi - xj)
        dy = float(yi - yj)
        dists.append((dx * dx + dy * dy) ** 0.5)
    if not dists:
        return 0.0
    dists.sort()
    return float(dists[len(dists) // 2])


def _hand_forward_dir(keypoints: HandKeypoints) -> tuple[float, float]:
    """估计手前向方向（手腕/掌根 -> 指尖）。"""
    wrist = _avg_points(keypoints, [0])
    if wrist is None:
        wrist = _avg_points(keypoints, [5, 9, 13, 17])

    tips = _avg_points(keypoints, [8, 12, 16, 20])
    if wrist is not None and tips is not None:
        return _normalize_direction((tips[0] - wrist[0], tips[1] - wrist[1]), fallback=(0.0, -1.0))

    palm = _avg_points(keypoints, [0, 5, 9, 13, 17])
    if palm is not None and tips is not None:
        return _normalize_direction((tips[0] - palm[0], tips[1] - palm[1]), fallback=(0.0, -1.0))

    return (0.0, -1.0)


def _avg_points(
    keypoints: HandKeypoints,
    indices: list[int],
) -> tuple[float, float] | None:
    """计算指定关键点集合的平均位置。"""
    pts: list[tuple[float, float]] = []
    for idx in indices:
        if idx >= len(keypoints):
            continue
        x, y, conf = keypoints[idx]
        if conf <= 0:
            continue
        pts.append((float(x), float(y)))
    if not pts:
        return None
    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
    )


def _make_asymmetric_roi_from_seed(
    center: tuple[float, float] | None,
    size: float | None,
    direction: tuple[float, float],
    context_scale: float,
    forward_shift: float,
    inward_scale: float,
    inward_shift: float,
    h: int,
    w: int,
) -> tuple[int, int, int, int] | None:
    """构建单个不对称 ROI。

    该 ROI 在前向（手指延长方向）扩展更大，
    在后向（手腕方向）仅保留少量补偿。
    """
    if center is None or size is None:
        return None

    cx, cy = center
    base = max(2.0, float(size))

    u = _normalize_direction(direction, fallback=(0.0, -1.0))
    v = (-u[1], u[0])

    # 以 base 为基准计算前向/后向/横向半径。
    outward_extent = base * (forward_shift + 0.5 * context_scale)
    inward_extent = base * (inward_shift + 0.5 * inward_scale)
    lateral_extent = base * 0.5 * max(1.0, context_scale, inward_scale)

    points = [
        (cx + u[0] * outward_extent + v[0] * lateral_extent, cy + u[1] * outward_extent + v[1] * lateral_extent),
        (cx + u[0] * outward_extent - v[0] * lateral_extent, cy + u[1] * outward_extent - v[1] * lateral_extent),
        (cx - u[0] * inward_extent + v[0] * lateral_extent, cy - u[1] * inward_extent + v[1] * lateral_extent),
        (cx - u[0] * inward_extent - v[0] * lateral_extent, cy - u[1] * inward_extent - v[1] * lateral_extent),
    ]

    x1 = int(round(min(p[0] for p in points)))
    y1 = int(round(min(p[1] for p in points)))
    x2 = int(round(max(p[0] for p in points)))
    y2 = int(round(max(p[1] for p in points)))

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return None
    return (x1, y1, x2, y2)


def _smooth_direction(
    previous: tuple[float, float],
    current: tuple[float, float],
    alpha: float,
) -> tuple[float, float]:
    """对方向向量做平滑，并避免帧间 180 度翻转。"""
    prev = _normalize_direction(previous, fallback=(0.0, -1.0))
    curr = _align_direction(previous=prev, current=current)
    if alpha <= 0:
        return prev
    if alpha >= 1:
        return curr
    blended = (
        (1.0 - alpha) * prev[0] + alpha * curr[0],
        (1.0 - alpha) * prev[1] + alpha * curr[1],
    )
    return _normalize_direction(blended, fallback=prev)


def _align_direction(
    previous: tuple[float, float],
    current: tuple[float, float],
) -> tuple[float, float]:
    """将当前方向对齐到上一帧方向半球，避免方向反跳。"""
    prev = _normalize_direction(previous, fallback=(0.0, -1.0))
    curr = _normalize_direction(current, fallback=prev)
    if (prev[0] * curr[0] + prev[1] * curr[1]) < 0:
        return (-curr[0], -curr[1])
    return curr


def _normalize_direction(
    direction: tuple[float, float],
    fallback: tuple[float, float],
) -> tuple[float, float]:
    """归一化方向向量。"""
    dx, dy = float(direction[0]), float(direction[1])
    norm = (dx * dx + dy * dy) ** 0.5
    if norm < 1e-6:
        return fallback
    return (dx / norm, dy / norm)


def _merge_rois(
    rois: list[tuple[int, int, int, int]],
    merge_iou: float,
) -> list[tuple[int, int, int, int]]:
    """合并重叠 ROI。"""
    if not rois:
        return []
    merge_iou = float(max(0.0, min(merge_iou, 1.0)))
    merged: list[tuple[int, int, int, int]] = []
    for roi in sorted(rois, key=_roi_area, reverse=True):
        inserted = False
        for i, kept in enumerate(merged):
            if _roi_iou(roi, kept) >= merge_iou or _overlap_small_ratio(roi, kept) >= 0.92:
                merged[i] = _union_roi(roi, kept)
                inserted = True
                break
        if not inserted:
            merged.append(roi)
    return merged


def _roi_area(roi: tuple[int, int, int, int]) -> int:
    """计算 ROI 面积。"""
    x1, y1, x2, y2 = roi
    return max(0, x2 - x1) * max(0, y2 - y1)


def _roi_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """计算两个 ROI 的 IoU。"""
    inter = _intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = _roi_area(a) + _roi_area(b) - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _overlap_small_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """计算交叠面积占较小 ROI 面积的比例。"""
    inter = _intersection_area(a, b)
    if inter <= 0:
        return 0.0
    small = min(_roi_area(a), _roi_area(b))
    if small <= 0:
        return 0.0
    return float(inter / small)


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """计算两个 ROI 的交叠面积。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return max(0, x2 - x1) * max(0, y2 - y1)


def _union_roi(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """返回两个 ROI 的并集外接框。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2))