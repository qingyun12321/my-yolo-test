from __future__ import annotations

"""MediaPipe Hand Landmarker 封装。

该模块负责：
1. 自动准备 hand_landmarker.task 模型文件；
2. 创建 VIDEO 模式的 Hand Landmarker；
3. 将检测结果转换为像素坐标关键点；
4. 提供 left/right 映射工具供下游稳定器使用。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass(frozen=True)
class HandResult:
    """单只手的检测结果。"""

    handedness: str  # "left"、"right" 或 "unknown"
    keypoints: list[tuple[float, float, float]]


def ensure_hand_model(model_path: Path) -> Path:
    """确保 hand_landmarker.task 可用，不存在则自动下载。"""
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(HAND_LANDMARKER_URL, model_path)
    return model_path


def create_hand_landmarker(
    model_path: Path,
    num_hands: int = 2,
    min_detection_confidence: float = 0.5,
    min_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> "mp.tasks.vision.HandLandmarker":
    """创建 MediaPipe Hand Landmarker（VIDEO 模式）。"""
    base_options = mp.tasks.BaseOptions
    hand_landmarker = mp.tasks.vision.HandLandmarker
    hand_options = mp.tasks.vision.HandLandmarkerOptions
    running_mode = mp.tasks.vision.RunningMode

    options = hand_options(
        base_options=base_options(model_asset_path=str(model_path)),
        running_mode=running_mode.VIDEO,
        num_hands=max(1, int(num_hands)),
        min_hand_detection_confidence=float(min_detection_confidence),
        min_hand_presence_confidence=float(min_presence_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
    )
    return hand_landmarker.create_from_options(options)


def detect_hands(
    landmarker: "mp.tasks.vision.HandLandmarker",
    frame_bgr: np.ndarray,
    timestamp_ms: int,
) -> list[HandResult]:
    """检测当前帧的手关键点并输出像素坐标。"""
    height, width = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.hand_landmarks:
        return []

    hand_results: list[HandResult] = []
    for hand_idx, landmarks in enumerate(result.hand_landmarks):
        handedness = "unknown"
        if result.handedness and hand_idx < len(result.handedness):
            if result.handedness[hand_idx]:
                handedness = result.handedness[hand_idx][0].category_name.lower()

        keypoints: list[tuple[float, float, float]] = []
        for lm in landmarks:
            x = float(lm.x) * width
            y = float(lm.y) * height
            # Tasks API 未提供稳定的单关键点置信度，这里统一设为 1.0。
            # 这样可以避免绘制和接触判定因为“伪低置信度”而误丢点。
            conf = 1.0
            keypoints.append((x, y, conf))

        hand_results.append(HandResult(handedness=handedness, keypoints=keypoints))

    return hand_results


def to_left_right_map(
    hands: Iterable[HandResult],
) -> dict[str, list[tuple[float, float, float]] | None]:
    """将多手检测结果映射为 left/right 两个槽位。"""
    output: dict[str, list[tuple[float, float, float]] | None] = {
        "left": None,
        "right": None,
    }
    for item in hands:
        if item.handedness in ("left", "right") and output[item.handedness] is None:
            output[item.handedness] = item.keypoints
    return output