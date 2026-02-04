from __future__ import annotations

"""基于 IoU 的轻量级单/多人追踪器。"""

from dataclasses import dataclass

from ..types import PoseDetection, TrackedPerson


@dataclass
class _Track:
    """内部追踪状态（不对外暴露）。"""
    track_id: int
    bbox: tuple[float, float, float, float]
    keypoints: list[tuple[float, float, float]]
    score: float
    last_seen: float


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """计算两个 bbox 的 IoU。

    参数:
        a: 第一个 bbox (x1, y1, x2, y2)。
        b: 第二个 bbox (x1, y1, x2, y2)。

    返回:
        float: IoU 值，范围 [0.0, 1.0]。
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: float = 1.0) -> None:
        """初始化追踪器。

        参数:
            iou_threshold: 匹配阈值，范围 [0.0, 1.0]，越大越严格。
            max_age: 允许的最大丢失时间（秒），>0。
        """
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: list[PoseDetection], timestamp: float) -> list[TrackedPerson]:
        """更新追踪状态并返回追踪结果。

        参数:
            detections: 当前帧的姿态检测列表。
            timestamp: 当前时间戳（秒）。

        返回:
            list[TrackedPerson]: 更新后的追踪对象列表。
        """
        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()

        for det_idx, detection in enumerate(detections):
            best_iou = 0.0
            best_track = None
            for track_idx, track in enumerate(self._tracks):
                if track_idx in used_tracks:
                    continue
                iou = _iou(detection.bbox, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track = track_idx
            if best_track is not None and best_iou >= self._iou_threshold:
                assignments[det_idx] = best_track
                used_tracks.add(best_track)

        updated_tracks: list[_Track] = []
        for det_idx, detection in enumerate(detections):
            if det_idx in assignments:
                track = self._tracks[assignments[det_idx]]
                track.bbox = detection.bbox
                track.keypoints = detection.keypoints
                track.score = detection.score
                track.last_seen = timestamp
                updated_tracks.append(track)
            else:
                updated_tracks.append(
                    _Track(
                        track_id=self._next_id,
                        bbox=detection.bbox,
                        keypoints=detection.keypoints,
                        score=detection.score,
                        last_seen=timestamp,
                    )
                )
                self._next_id += 1

        self._tracks = [
            track for track in updated_tracks if timestamp - track.last_seen <= self._max_age
        ]

        return [
            TrackedPerson(
                track_id=track.track_id,
                bbox=track.bbox,
                keypoints=track.keypoints,
                score=track.score,
            )
            for track in self._tracks
        ]
