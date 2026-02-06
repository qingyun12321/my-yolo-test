from __future__ import annotations

"""目标时序稳定器。

用于提升“手和物体互相遮挡”场景下的检测稳定性，核心能力：
1. 跨帧关联同一物体，减少短时漏检造成的闪烁；
2. 对 bbox 与置信度做时序平滑，降低抖动；
3. 对类别分数做累积投票，抑制同一物体在相邻帧反复变类；
4. 在短时遮挡时保留目标若干帧，避免接触状态瞬断。
"""

from dataclasses import dataclass, field
from typing import Any

from ..types import BBox, DetectedObject


@dataclass
class _ObjectTrack:
    """单个物体轨迹状态。"""

    name: str
    bbox: BBox
    score: float
    mask: Any | None
    hits: int = 1
    misses: int = 0
    class_scores: dict[str, float] = field(default_factory=dict)


class ObjectTemporalStabilizer:
    """对检测结果做轻量时序稳定。"""

    def __init__(
        self,
        hold_frames: int = 3,
        min_hits: int = 1,
        match_iou: float = 0.32,
        match_center_ratio: float = 0.72,
        bbox_alpha: float = 0.62,
        class_decay: float = 0.9,
        score_decay: float = 0.9,
    ) -> None:
        self.hold_frames = max(0, int(hold_frames))
        self.min_hits = max(1, int(min_hits))
        self.match_iou = float(max(0.0, min(match_iou, 1.0)))
        self.match_center_ratio = float(max(0.0, match_center_ratio))
        self.bbox_alpha = float(max(0.0, min(bbox_alpha, 1.0)))
        self.class_decay = float(max(0.0, min(class_decay, 1.0)))
        self.score_decay = float(max(0.0, min(score_decay, 1.0)))
        self._tracks: list[_ObjectTrack] = []

    def reset(self) -> None:
        """清空全部时序状态。"""
        self._tracks.clear()

    def update(self, objects: list[DetectedObject]) -> list[DetectedObject]:
        """输入当前帧检测结果，输出稳定后的目标列表。"""
        self._decay_tracks()
        used_track_indices: set[int] = set()

        for obj in sorted(objects, key=lambda item: float(item.score), reverse=True):
            match_idx = self._find_match_track(obj, used_track_indices)
            if match_idx is None:
                self._tracks.append(
                    _ObjectTrack(
                        name=obj.name,
                        bbox=obj.bbox,
                        score=float(obj.score),
                        mask=obj.mask,
                        hits=1,
                        misses=0,
                        class_scores={obj.name: float(obj.score)},
                    )
                )
                used_track_indices.add(len(self._tracks) - 1)
                continue

            track = self._tracks[match_idx]
            self._update_track(track, obj)
            used_track_indices.add(match_idx)

        for idx, track in enumerate(self._tracks):
            if idx in used_track_indices:
                continue
            track.misses += 1
            track.score = float(track.score * self.score_decay)

        self._tracks = [track for track in self._tracks if track.misses <= self.hold_frames]
        return self._emit_objects()

    def _decay_tracks(self) -> None:
        """衰减各轨迹的类别历史分数，避免过久历史主导当前判断。"""
        if self.class_decay >= 1.0:
            return
        for track in self._tracks:
            decayed: dict[str, float] = {}
            for name, score in track.class_scores.items():
                value = float(score * self.class_decay)
                if value >= 1e-4:
                    decayed[name] = value
            track.class_scores = decayed

    def _find_match_track(
        self,
        obj: DetectedObject,
        used_track_indices: set[int],
    ) -> int | None:
        """在未占用轨迹中寻找与当前目标最匹配的一条。"""
        best_idx: int | None = None
        best_score = -1.0
        for idx, track in enumerate(self._tracks):
            if idx in used_track_indices:
                continue

            iou = _bbox_iou(obj.bbox, track.bbox)
            center_ratio = _bbox_center_ratio(obj.bbox, track.bbox)
            if iou < self.match_iou and center_ratio > self.match_center_ratio:
                continue

            # 几何匹配分 + 类别一致奖励
            geom = iou + max(0.0, 1.0 - center_ratio / max(1e-6, self.match_center_ratio)) * 0.35
            class_bonus = 0.12 if obj.name == track.name else 0.0
            score = geom + class_bonus
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _update_track(self, track: _ObjectTrack, obj: DetectedObject) -> None:
        """将当前帧观测融合到轨迹状态。"""
        alpha = self.bbox_alpha
        if alpha <= 0.0:
            fused_bbox = track.bbox
            fused_score = track.score
        elif alpha >= 1.0:
            fused_bbox = obj.bbox
            fused_score = float(obj.score)
        else:
            fused_bbox = (
                float((1.0 - alpha) * track.bbox[0] + alpha * obj.bbox[0]),
                float((1.0 - alpha) * track.bbox[1] + alpha * obj.bbox[1]),
                float((1.0 - alpha) * track.bbox[2] + alpha * obj.bbox[2]),
                float((1.0 - alpha) * track.bbox[3] + alpha * obj.bbox[3]),
            )
            fused_score = float((1.0 - alpha) * track.score + alpha * float(obj.score))

        track.bbox = fused_bbox
        track.score = fused_score
        if obj.mask is not None:
            track.mask = obj.mask
        track.misses = 0
        track.hits += 1

        track.class_scores[obj.name] = float(track.class_scores.get(obj.name, 0.0) + obj.score)
        if track.class_scores:
            track.name = max(track.class_scores.items(), key=lambda item: item[1])[0]
        else:
            track.name = obj.name

    def _emit_objects(self) -> list[DetectedObject]:
        """将轨迹状态转换为可用的检测目标列表。"""
        emitted: list[DetectedObject] = []
        for track in self._tracks:
            if track.hits < self.min_hits:
                continue

            # 对遮挡保持阶段的分数做可控衰减，避免“僵尸框”长期存在。
            score = float(track.score * (self.score_decay ** track.misses))
            score = max(0.0, min(1.0, score))
            emitted.append(
                DetectedObject(
                    name=track.name,
                    bbox=track.bbox,
                    score=score,
                    mask=track.mask,
                )
            )

        return _dedupe_emitted(emitted)


def _dedupe_emitted(objects: list[DetectedObject]) -> list[DetectedObject]:
    """对输出做一次轻量去重，避免轨迹短时分裂带来双框。"""
    if not objects:
        return []
    ranked = sorted(objects, key=lambda item: float(item.score), reverse=True)
    kept: list[DetectedObject] = []
    for obj in ranked:
        duplicate = False
        for other in kept:
            if obj.name != other.name:
                continue
            if _bbox_iou(obj.bbox, other.bbox) >= 0.8 or _overlap_small_ratio(obj.bbox, other.bbox) >= 0.9:
                duplicate = True
                break
        if not duplicate:
            kept.append(obj)
    return kept


def _bbox_iou(a: BBox, b: BBox) -> float:
    """计算边框 IoU。"""
    inter = _bbox_intersection(a, b)
    if inter <= 0.0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _overlap_small_ratio(a: BBox, b: BBox) -> float:
    """计算交叠面积占较小框面积的比例。"""
    inter = _bbox_intersection(a, b)
    if inter <= 0.0:
        return 0.0
    small = min(_bbox_area(a), _bbox_area(b))
    if small <= 1e-6:
        return 0.0
    return float(inter / small)


def _bbox_center_ratio(a: BBox, b: BBox) -> float:
    """计算中心距离相对尺度比值。"""
    ac = _bbox_center(a)
    bc = _bbox_center(b)
    dist = _point_distance(ac, bc)
    scale = min(_bbox_diagonal(a), _bbox_diagonal(b))
    if scale <= 1e-6:
        return float("inf")
    return float(dist / scale)


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


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    """计算边框中心点。"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _bbox_intersection(a: BBox, b: BBox) -> float:
    """计算边框交集面积。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def _point_distance(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    """计算两点欧氏距离。"""
    dx = float(p0[0] - p1[0])
    dy = float(p0[1] - p1[1])
    return float((dx * dx + dy * dy) ** 0.5)
