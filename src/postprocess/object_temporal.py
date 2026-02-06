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
        single_detection_mode = len(objects) == 1

        for obj in sorted(objects, key=lambda item: float(item.score), reverse=True):
            match_idx = self._find_match_track(obj, used_track_indices)
            if match_idx is None and single_detection_mode:
                # 单目标帧下启用更宽松重关联，优先保持轨迹连续性。
                match_idx = self._find_relaxed_track(obj, used_track_indices)
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
        return self._emit_objects(has_current_objects=bool(objects))

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
            if iou < self.match_iou:
                # 快速运动重关联兜底：
                # 当 IoU 一时不足，但中心位移与尺度仍在可接受范围时，允许继续关联，
                # 避免同一物体被切成多条短轨迹（移动时多标签的主要来源）。
                area_ratio = _bbox_area_ratio(obj.bbox, track.bbox)
                relaxed_center_limit = max(1e-6, self.match_center_ratio * 1.9)
                fast_motion_relink = (
                    center_ratio <= relaxed_center_limit
                    and 0.35 <= area_ratio <= 2.8
                )
                if not fast_motion_relink:
                    continue

            # 几何匹配分 + 类别一致奖励
            overlap = _overlap_small_ratio(obj.bbox, track.bbox)
            geom = max(iou, overlap) + max(
                0.0, 1.0 - center_ratio / max(1e-6, self.match_center_ratio)
            ) * 0.35
            class_bonus = 0.12 if obj.name == track.name else 0.0
            score = geom + class_bonus
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _find_relaxed_track(
        self,
        obj: DetectedObject,
        used_track_indices: set[int],
    ) -> int | None:
        """单目标场景下的宽松重关联（用于抗快速位移分裂）。"""
        best_idx: int | None = None
        best_cost = float("inf")
        center_limit = max(1e-6, self.match_center_ratio * 2.6)
        for idx, track in enumerate(self._tracks):
            if idx in used_track_indices:
                continue

            center_ratio = _bbox_center_ratio(obj.bbox, track.bbox)
            if center_ratio > center_limit:
                continue

            area_ratio = _bbox_area_ratio(obj.bbox, track.bbox)
            if area_ratio > 4.0:
                continue

            # cost 越小越好：优先中心更近、miss 更少、类别更一致。
            cost = center_ratio + float(track.misses) * 0.35
            if obj.name == track.name:
                cost -= 0.15
            if cost < best_cost:
                best_cost = cost
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

    def _emit_objects(self, has_current_objects: bool) -> list[DetectedObject]:
        """将轨迹状态转换为可用的检测目标列表。

        设计要点：
        1. 只要当前帧存在观测目标，就不输出 miss 轨迹，避免旧位置残影；
        2. 当仅靠 hold 保留轨迹时，不再输出历史 mask，防止旧轮廓残留。
        """
        emitted: list[DetectedObject] = []
        for track in self._tracks:
            if track.hits < self.min_hits:
                continue

            if track.misses > 1:
                # 无检测阶段最多只保留 1 帧输出，后续帧隐藏以消除旧位置残留。
                continue

            if track.misses > 0 and has_current_objects:
                # 当前帧已有真实观测时，直接隐藏 miss 轨迹，避免“旧位置还挂着一层轮廓”。
                continue

            # 对遮挡保持阶段的分数做可控衰减，避免“僵尸框”长期存在。
            score = float(track.score * (self.score_decay ** track.misses))
            score = max(0.0, min(1.0, score))
            if score < 0.08:
                continue

            # miss 阶段不输出历史 mask，防止画面出现旧轮廓残影。
            emit_mask = track.mask if track.misses <= 0 else None
            emitted.append(
                DetectedObject(
                    name=track.name,
                    bbox=track.bbox,
                    score=score,
                    mask=emit_mask,
                )
            )

        return _dedupe_emitted(emitted)


def _dedupe_emitted(objects: list[DetectedObject]) -> list[DetectedObject]:
    """对输出做一次轻量去重，避免轨迹短时分裂带来多标签。"""
    if not objects:
        return []

    # 优先保留“已知类别 + 有 mask + 高分”的目标，unknown 与低分候选靠后。
    ranked = sorted(objects, key=_emitted_quality, reverse=True)
    kept: list[DetectedObject] = []
    for obj in ranked:
        dropped = False
        idx = 0
        while idx < len(kept):
            other = kept[idx]
            iou = _bbox_iou(obj.bbox, other.bbox)
            overlap = _overlap_small_ratio(obj.bbox, other.bbox)
            center_ratio = _bbox_center_ratio(obj.bbox, other.bbox)
            area_ratio = _bbox_area_ratio(obj.bbox, other.bbox)
            same_class = obj.name == other.name

            if same_class:
                same_class_duplicate = (
                    iou >= 0.65
                    or overlap >= 0.82
                    or (center_ratio <= 0.36 and 0.40 <= area_ratio <= 2.5)
                )
                if same_class_duplicate:
                    dropped = True
                    break
                idx += 1
                continue

            # 跨类别强冲突：高重叠 + 中心接近，通常是同一物体的类别抖动。
            cross_class_conflict = overlap >= 0.90 or (
                iou >= 0.68 and center_ratio <= 0.32 and 0.35 <= area_ratio <= 2.8
            )
            if not cross_class_conflict:
                idx += 1
                continue

            better = _prefer_emitted_object(obj, other)
            if better is other:
                dropped = True
                break
            kept.pop(idx)

        if not dropped:
            kept.append(obj)
    return kept


def _emitted_quality(obj: DetectedObject) -> tuple[int, int, float, float]:
    """定义时序输出质量排序：已知类别优先，其次 mask、分数、面积。"""
    return (
        0 if obj.name == "unknown" else 1,
        1 if obj.mask is not None else 0,
        float(obj.score),
        _bbox_area(obj.bbox),
    )


def _prefer_emitted_object(a: DetectedObject, b: DetectedObject) -> DetectedObject:
    """跨类别冲突时保留更可信目标。"""
    # 已知类别优先于 unknown，避免 unknown 抢占已知目标。
    if a.name == "unknown" and b.name != "unknown":
        return b
    if b.name == "unknown" and a.name != "unknown":
        return a

    if abs(float(a.score) - float(b.score)) >= 0.12:
        return a if a.score > b.score else b
    return a if _emitted_quality(a) >= _emitted_quality(b) else b


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


def _bbox_area_ratio(a: BBox, b: BBox) -> float:
    """计算边框面积比（大/小）。"""
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    small = min(area_a, area_b)
    if small <= 1e-6:
        return float("inf")
    large = max(area_a, area_b)
    return float(large / small)


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
