from __future__ import annotations

"""基于规则的简单动作识别（挥手/点头/摇头）。"""

from collections import deque

from ..types import ActionResult


class ActionRuleEngine:
    def __init__(
        self,
        max_seconds: float = 2.0,
        min_frames: int = 6,
        keypoint_conf: float = 0.3,
    ) -> None:
        """初始化动作规则引擎。

        参数:
            max_seconds: 滑动窗口最大时长（秒），>0。
            min_frames: 最小帧数要求（整数，>=1）。
            keypoint_conf: 关键点置信度阈值，范围 [0.0, 1.0]。
        """
        self._history: dict[int, deque[tuple[float, dict[str, tuple[float, float, float]]]]] = {}
        self._max_seconds = max_seconds
        self._min_frames = min_frames
        self._keypoint_conf = keypoint_conf

    def update(
        self,
        track_id: int,
        normalized_keypoints: dict[str, tuple[float, float, float]],
        timestamp: float,
    ) -> None:
        """更新指定人员的关键点历史。

        参数:
            track_id: 追踪 ID（整数）。
            normalized_keypoints: 归一化关键点字典。
            timestamp: 当前时间戳（秒）。
        """
        if track_id not in self._history:
            self._history[track_id] = deque()
        history = self._history[track_id]
        history.append((timestamp, normalized_keypoints))
        while history and timestamp - history[0][0] > self._max_seconds:
            history.popleft()

    def classify(self, track_id: int) -> ActionResult:
        """基于历史关键点判断动作类别。

        参数:
            track_id: 追踪 ID（整数）。

        返回:
            ActionResult: 动作与置信度。
        """
        history = self._history.get(track_id)
        if not history or len(history) < self._min_frames:
            return ActionResult(action="none", confidence=0.0)

        wave_score = max(
            _wave_score(history, "left", self._min_frames, self._keypoint_conf),
            _wave_score(history, "right", self._min_frames, self._keypoint_conf),
        )
        nod_score = _nod_score(history, self._min_frames, self._keypoint_conf)
        shake_score = _shake_score(history, self._min_frames, self._keypoint_conf)

        scores = {
            "wave": wave_score,
            "nod": nod_score,
            "shake": shake_score,
        }
        best_action = max(scores, key=scores.get)
        best_score = scores[best_action]
        if best_score <= 0:
            return ActionResult(action="none", confidence=0.0)
        return ActionResult(action=best_action, confidence=best_score)


def _sign_changes(values: list[float], min_delta: float) -> int:
    """统计数值序列的方向变化次数（忽略微小变化）。

    参数:
        values: 数值序列。
        min_delta: 变化阈值（>0），小于该值的变化会被忽略。
    """
    signs: list[int] = []
    for idx in range(1, len(values)):
        delta = values[idx] - values[idx - 1]
        if abs(delta) < min_delta:
            continue
        signs.append(1 if delta > 0 else -1)
    changes = 0
    for idx in range(1, len(signs)):
        if signs[idx] != signs[idx - 1]:
            changes += 1
    return changes


def _wave_score(
    history: deque[tuple[float, dict[str, tuple[float, float, float]]]],
    side: str,
    min_frames: int,
    min_conf: float,
) -> float:
    """计算挥手动作评分。

    参数:
        history: 关键点历史。
        side: "left" 或 "right"。
        min_frames: 最小帧数要求（整数，>=1）。
        min_conf: 关键点置信度阈值，范围 [0.0, 1.0]。
    """
    wrist_key = f"{side}_wrist"
    shoulder_key = f"{side}_shoulder"
    samples: list[tuple[float, float, float]] = []
    for _, kps in history:
        wrist = kps.get(wrist_key)
        shoulder = kps.get(shoulder_key)
        if wrist and shoulder and wrist[2] >= min_conf and shoulder[2] >= min_conf:
            samples.append((wrist[0], wrist[1], shoulder[1]))

    if len(samples) < min_frames:
        return 0.0

    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]
    shoulders = [s[2] for s in samples]
    amplitude = max(xs) - min(xs)
    changes = _sign_changes(xs, min_delta=0.03)
    above_ratio = sum(1 for y, sy in zip(ys, shoulders) if y < sy - 0.1) / len(samples)

    if amplitude < 0.25 or changes < 2 or above_ratio < 0.6:
        return 0.0

    score = min(1.0, amplitude / 0.5) * min(1.0, changes / 3.0) * above_ratio
    return float(score)


def _nod_score(
    history: deque[tuple[float, dict[str, tuple[float, float, float]]]],
    min_frames: int,
    min_conf: float,
) -> float:
    """计算点头动作评分。

    参数:
        history: 关键点历史。
        min_frames: 最小帧数要求（整数，>=1）。
        min_conf: 关键点置信度阈值，范围 [0.0, 1.0]。
    """
    values = _series(history, "nose", axis="y", min_conf=min_conf)
    if len(values) < min_frames:
        return 0.0
    amplitude = max(values) - min(values)
    changes = _sign_changes(values, min_delta=0.03)
    if amplitude < 0.2 or changes < 2:
        return 0.0
    score = min(1.0, amplitude / 0.4) * min(1.0, changes / 3.0)
    return float(score)


def _shake_score(
    history: deque[tuple[float, dict[str, tuple[float, float, float]]]],
    min_frames: int,
    min_conf: float,
) -> float:
    """计算摇头动作评分。

    参数:
        history: 关键点历史。
        min_frames: 最小帧数要求（整数，>=1）。
        min_conf: 关键点置信度阈值，范围 [0.0, 1.0]。
    """
    values = _series(history, "nose", axis="x", min_conf=min_conf)
    if len(values) < min_frames:
        return 0.0
    amplitude = max(values) - min(values)
    changes = _sign_changes(values, min_delta=0.03)
    if amplitude < 0.2 or changes < 2:
        return 0.0
    score = min(1.0, amplitude / 0.4) * min(1.0, changes / 3.0)
    return float(score)


def _series(
    history: deque[tuple[float, dict[str, tuple[float, float, float]]]],
    key: str,
    axis: str,
    min_conf: float,
) -> list[float]:
    """从历史中提取指定关键点在某轴上的序列。

    参数:
        history: 关键点历史。
        key: 关键点名称（如 "nose"）。
        axis: "x" 或 "y"。
        min_conf: 关键点置信度阈值，范围 [0.0, 1.0]。
    """
    values: list[float] = []
    index = 0 if axis == "x" else 1
    for _, kps in history:
        point = kps.get(key)
        if point is None or point[2] < min_conf:
            continue
        values.append(point[index])
    return values
