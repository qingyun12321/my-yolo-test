from __future__ import annotations

"""MediaPipe Hand Landmarker 推理封装。

说明：
    - 使用官方 Hand Landmarker 任务模型（.task）
    - 采用 VIDEO 运行模式，以复用上一帧的手部 ROI（官方推荐的追踪方式）
    - 若模型文件不存在，会自动下载到指定路径
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np


# 官方模型下载地址（Hand Landmarker 模型包）
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass(frozen=True)
class HandResult:
    """单手关键点检测结果。"""

    handedness: str  # "left" 或 "right"
    keypoints: list[tuple[float, float, float]]


def ensure_hand_model(model_path: Path) -> Path:
    """确保 hand_landmarker.task 存在，不存在则自动下载。

    参数:
        model_path: 模型文件路径（应为 .task）。

    返回:
        Path: 实际可用的模型文件路径。
    """
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
    """创建 MediaPipe Hand Landmarker 实例（VIDEO 模式）。

    参数:
        model_path: 模型路径（.task）。
        num_hands: 最大检测手数量（>=1）。
        min_detection_confidence: 检测阈值（0-1）。
        min_presence_confidence: 关键点存在阈值（0-1）。
        min_tracking_confidence: 追踪阈值（0-1）。

    返回:
        HandLandmarker: 任务实例。
    """
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=max(1, int(num_hands)),
        min_hand_detection_confidence=float(min_detection_confidence),
        min_hand_presence_confidence=float(min_presence_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
    )
    return HandLandmarker.create_from_options(options)


def detect_hands(
    landmarker: "mp.tasks.vision.HandLandmarker",
    frame_bgr: np.ndarray,
    timestamp_ms: int,
) -> list[HandResult]:
    """在当前帧执行手部关键点检测。

    参数:
        landmarker: HandLandmarker 实例（VIDEO 模式）。
        frame_bgr: OpenCV BGR 图像。
        timestamp_ms: 视频时间戳（毫秒）。

    返回:
        list[HandResult]: 多手关键点结果（像素坐标）。
    """
    height, width = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    hand_results: list[HandResult] = []

    if not result.hand_landmarks:
        return hand_results

    for hand_idx, landmarks in enumerate(result.hand_landmarks):
        handedness = "unknown"
        if result.handedness and hand_idx < len(result.handedness):
            if result.handedness[hand_idx]:
                handedness = result.handedness[hand_idx][0].category_name.lower()

        keypoints: list[tuple[float, float, float]] = []
        for lm in landmarks:
            x = float(lm.x) * width
            y = float(lm.y) * height
            # z 为相对深度，先映射到 conf 字段便于下游绘制阈值使用
            conf = 1.0
            if hasattr(lm, "visibility") and lm.visibility is not None:
                conf = float(lm.visibility)
            keypoints.append((x, y, conf))

        hand_results.append(HandResult(handedness=handedness, keypoints=keypoints))

    return hand_results


def to_left_right_map(
    hands: Iterable[HandResult],
) -> dict[str, list[tuple[float, float, float]] | None]:
    """将多手结果整理为 left/right 映射（优先保留置信度更稳定的一只手）。

    参数:
        hands: HandResult 列表。

    返回:
        dict: {"left": keypoints|None, "right": keypoints|None}
    """
    output: dict[str, list[tuple[float, float, float]] | None] = {"left": None, "right": None}
    for item in hands:
        if item.handedness in ("left", "right") and output[item.handedness] is None:
            output[item.handedness] = item.keypoints
    return output
