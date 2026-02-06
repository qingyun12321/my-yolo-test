from __future__ import annotations

"""手关键点时序稳定器：稳定左右手归属并降低抖动。"""

from dataclasses import dataclass
from typing import Iterable

from ..types import BBox, HandKeypoints
from .hand_mediapipe import HandResult


@dataclass(frozen=True)
class _HandCandidate:
    """单个手候选，包含几何信息与 handedness 先验。"""

    keypoints: HandKeypoints
    bbox: BBox
    center: tuple[float, float]
    area: float
    diagonal: float
    handedness: str


@dataclass
class _SideState:
    """单侧手（left/right）的时序状态。"""

    keypoints: HandKeypoints | None = None
    misses: int = 0


class HandStabilizer:
    """对手部关键点进行时序稳定。

    功能包括：
    1. 左右手分配稳定，减少颜色互跳。
    2. EMA 平滑，兼顾稳定性与响应速度。
    3. 丢帧保持，减少短时遮挡导致的闪烁。
    4. 重叠手抑制，避免同一只手被显示为两只。
    """

    def __init__(
        self,
        smooth_alpha: float = 0.55,
        hold_frames: int = 2,
        side_merge_ratio: float = 0.35,
        adaptive_alpha: bool = True,
        fast_alpha: float = 0.88,
        motion_scale: float = 0.12,
    ) -> None:
        self.smooth_alpha = float(max(0.0, min(smooth_alpha, 1.0)))
        self.hold_frames = max(0, int(hold_frames))
        self.side_merge_ratio = float(max(0.0, side_merge_ratio))
        self.adaptive_alpha = bool(adaptive_alpha)
        self.fast_alpha = float(max(0.0, min(fast_alpha, 1.0)))
        self.motion_scale = float(max(1e-6, motion_scale))
        self._state = {
            "left": _SideState(),
            "right": _SideState(),
        }

    def reset(self) -> None:
        """重置左右手状态。"""
        self._state["left"] = _SideState()
        self._state["right"] = _SideState()

    def update(
        self,
        hands: Iterable[HandResult],
        frame_shape: tuple[int, int],
    ) -> dict[str, HandKeypoints | None]:
        """输入当前帧手检测结果，输出稳定后的 left/right 映射。"""
        candidates = self._extract_candidates(hands)
        candidates = self._merge_close_candidates(candidates)
        if len(candidates) > 2:
            candidates = candidates[:2]

        assignments = self._assign_sides(candidates, frame_width=int(frame_shape[1]))
        self._update_state(assignments)
        self._suppress_overlap(assignments)

        return {
            "left": self._state["left"].keypoints,
            "right": self._state["right"].keypoints,
        }

    def _extract_candidates(self, hands: Iterable[HandResult]) -> list[_HandCandidate]:
        """过滤并提取候选手。"""
        candidates: list[_HandCandidate] = []
        for item in hands:
            if not item.keypoints:
                continue
            bbox = _keypoints_bbox(item.keypoints)
            area = _bbox_area(bbox)
            diagonal = _bbox_diagonal(bbox)
            if area <= 1.0 or diagonal <= 1.0:
                continue
            candidates.append(
                _HandCandidate(
                    keypoints=item.keypoints,
                    bbox=bbox,
                    center=_keypoints_center(item.keypoints),
                    area=area,
                    diagonal=diagonal,
                    handedness=(item.handedness or "unknown").lower(),
                )
            )
        candidates.sort(key=lambda c: c.area, reverse=True)
        return candidates

    def _merge_close_candidates(self, candidates: list[_HandCandidate]) -> list[_HandCandidate]:
        """合并明显重复的候选手。"""
        if self.side_merge_ratio <= 0:
            return candidates
        merged: list[_HandCandidate] = []
        for candidate in candidates:
            duplicate = False
            for kept in merged:
                limit = self.side_merge_ratio * min(candidate.diagonal, kept.diagonal)
                close_center = _point_distance(candidate.center, kept.center) <= limit
                overlap_small = _bbox_overlap_small_ratio(candidate.bbox, kept.bbox) >= 0.72
                overlap_iou = _bbox_iou(candidate.bbox, kept.bbox) >= 0.45
                if close_center or overlap_small or overlap_iou:
                    duplicate = True
                    break
            if not duplicate:
                merged.append(candidate)
        return merged

    def _assign_sides(
        self,
        candidates: list[_HandCandidate],
        frame_width: int,
    ) -> dict[str, HandKeypoints]:
        """为候选手分配 left/right。"""
        assignments: dict[str, HandKeypoints] = {}
        if not candidates:
            return assignments

        prev_centers: dict[str, tuple[float, float]] = {}
        for side in ("left", "right"):
            prev = self._state[side].keypoints
            if prev is not None:
                prev_centers[side] = _keypoints_center(prev)

        if len(candidates) == 1:
            candidate = candidates[0]
            if candidate.handedness in ("left", "right"):
                side = candidate.handedness
            elif prev_centers:
                distances = {
                    side: _point_distance(candidate.center, center)
                    for side, center in prev_centers.items()
                }
                side = min(distances, key=distances.get)
            else:
                side = "left" if candidate.center[0] <= frame_width * 0.5 else "right"
            assignments[side] = candidate.keypoints
            return assignments

        c0, c1 = candidates[0], candidates[1]
        if {c0.handedness, c1.handedness} == {"left", "right"}:
            assignments[c0.handedness] = c0.keypoints
            assignments[c1.handedness] = c1.keypoints
            return assignments

        if "left" in prev_centers and "right" in prev_centers:
            c0_left = _point_distance(c0.center, prev_centers["left"])
            c0_right = _point_distance(c0.center, prev_centers["right"])
            c1_left = _point_distance(c1.center, prev_centers["left"])
            c1_right = _point_distance(c1.center, prev_centers["right"])
            if (c0_left + c1_right) <= (c0_right + c1_left):
                assignments["left"] = c0.keypoints
                assignments["right"] = c1.keypoints
            else:
                assignments["left"] = c1.keypoints
                assignments["right"] = c0.keypoints
            return assignments

        if len(prev_centers) == 1:
            known_side = next(iter(prev_centers))
            other_side = "right" if known_side == "left" else "left"
            d0 = _point_distance(c0.center, prev_centers[known_side])
            d1 = _point_distance(c1.center, prev_centers[known_side])
            if d0 <= d1:
                assignments[known_side] = c0.keypoints
                assignments[other_side] = c1.keypoints
            else:
                assignments[known_side] = c1.keypoints
                assignments[other_side] = c0.keypoints
            return assignments

        if c0.center[0] <= c1.center[0]:
            assignments["left"] = c0.keypoints
            assignments["right"] = c1.keypoints
        else:
            assignments["left"] = c1.keypoints
            assignments["right"] = c0.keypoints
        return assignments

    def _update_state(self, assignments: dict[str, HandKeypoints]) -> None:
        """更新左右手状态并应用平滑。"""
        for side in ("left", "right"):
            assigned = assignments.get(side)
            state = self._state[side]
            if assigned is not None:
                if state.keypoints is not None:
                    alpha = self._resolve_alpha(state.keypoints, assigned)
                    assigned = _smooth_keypoints(state.keypoints, assigned, alpha)
                state.keypoints = assigned
                state.misses = 0
                continue

            if state.keypoints is not None and state.misses < self.hold_frames:
                state.misses += 1
            else:
                state.keypoints = None
                state.misses = self.hold_frames + 1

    def _resolve_alpha(self, previous: HandKeypoints, current: HandKeypoints) -> float:
        """根据运动幅度自适应决定平滑强度。"""
        base = self.smooth_alpha
        if not self.adaptive_alpha:
            return base
        if self.fast_alpha <= base:
            return base

        prev_center = _keypoints_center(previous)
        curr_center = _keypoints_center(current)
        prev_diag = _bbox_diagonal(_keypoints_bbox(previous))
        curr_diag = _bbox_diagonal(_keypoints_bbox(current))
        scale = max(1.0, min(prev_diag, curr_diag))
        motion_ratio = _point_distance(prev_center, curr_center) / scale
        boost = min(1.0, motion_ratio / self.motion_scale)
        return float(base + (self.fast_alpha - base) * boost)

    def _suppress_overlap(self, assignments: dict[str, HandKeypoints]) -> None:
        """对高度重叠的左右手结果做抑制。"""
        left = self._state["left"].keypoints
        right = self._state["right"].keypoints
        if left is None or right is None:
            return

        left_bbox = _keypoints_bbox(left)
        right_bbox = _keypoints_bbox(right)
        center_dist = _point_distance(_keypoints_center(left), _keypoints_center(right))
        min_diag = min(_bbox_diagonal(left_bbox), _bbox_diagonal(right_bbox))
        overlap_small = _bbox_overlap_small_ratio(left_bbox, right_bbox)
        overlap_iou = _bbox_iou(left_bbox, right_bbox)
        if (
            center_dist > self.side_merge_ratio * min_diag
            and overlap_small < 0.7
            and overlap_iou < 0.45
        ):
            return

        # 本帧仅有一侧更新时，优先丢弃另一侧的陈旧状态。
        if len(assignments) == 1:
            only_side = next(iter(assignments))
            stale_side = "right" if only_side == "left" else "left"
            self._drop_side(stale_side)
            return

        left_area = _bbox_area(left_bbox)
        right_area = _bbox_area(right_bbox)
        if min(left_area, right_area) <= max(left_area, right_area) * 0.72:
            # 强嵌套通常表示同一只手的重复候选。
            if left_area >= right_area:
                self._drop_side("right")
            else:
                self._drop_side("left")
            return

        left_state = self._state["left"]
        right_state = self._state["right"]
        if left_state.misses <= right_state.misses:
            self._drop_side("right")
        else:
            self._drop_side("left")

    def _drop_side(self, side: str) -> None:
        """清空指定侧并标记为失效。"""
        state = self._state[side]
        state.keypoints = None
        state.misses = self.hold_frames + 1


def _smooth_keypoints(
    previous: HandKeypoints,
    current: HandKeypoints,
    alpha: float,
) -> HandKeypoints:
    """对关键点执行逐点 EMA 平滑。"""
    if len(previous) != len(current):
        return current
    if alpha >= 1.0:
        return current
    if alpha <= 0.0:
        return previous

    one_minus = 1.0 - alpha
    smoothed: HandKeypoints = []
    for p, c in zip(previous, current):
        smoothed.append(
            (
                float(one_minus * p[0] + alpha * c[0]),
                float(one_minus * p[1] + alpha * c[1]),
                float(one_minus * p[2] + alpha * c[2]),
            )
        )
    return smoothed


def _keypoints_bbox(keypoints: HandKeypoints) -> BBox:
    """计算关键点外接框。"""
    xs = [x for x, _, _ in keypoints]
    ys = [y for _, y, _ in keypoints]
    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))


def _keypoints_center(keypoints: HandKeypoints) -> tuple[float, float]:
    """计算关键点几何中心。"""
    xs = [x for x, _, _ in keypoints]
    ys = [y for _, y, _ in keypoints]
    return (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))


def _bbox_area(bbox: BBox) -> float:
    """计算边框面积。"""
    x1, y1, x2, y2 = bbox
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def _bbox_diagonal(bbox: BBox) -> float:
    """计算边框对角线长度。"""
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return float((w * w + h * h) ** 0.5)


def _bbox_iou(a: BBox, b: BBox) -> float:
    """计算两个边框的 IoU。"""
    inter = _bbox_intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _bbox_overlap_small_ratio(a: BBox, b: BBox) -> float:
    """计算交叠面积占较小边框面积的比例。"""
    inter = _bbox_intersection_area(a, b)
    if inter <= 0:
        return 0.0
    small = min(_bbox_area(a), _bbox_area(b))
    if small <= 1e-6:
        return 0.0
    return float(inter / small)


def _bbox_intersection_area(a: BBox, b: BBox) -> float:
    """计算两个边框交叠面积。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def _point_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    """计算两点欧氏距离。"""
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return float((dx * dx + dy * dy) ** 0.5)