from __future__ import annotations

"""手部关键点推理与结果整理。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from ..types import HandKeypoints


@dataclass(frozen=True)
class HandDetection:
    """单手关键点检测结果。"""

    keypoints: HandKeypoints
    score: float
    result: Any | None = None


def load_hand_model(model_arg: str | Path) -> YOLO:
    """加载手部关键点模型。

    参数:
        model_arg: 模型路径或模型名（例如手部关键点 best.pt）。

    返回:
        YOLO: Ultralytics YOLO 模型实例。
    """
    return YOLO(model_arg)


def infer_hand_keypoints(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    imgsz: int,
    device: str | None,
    return_result: bool = False,
    top_k: int | None = 1,
) -> list[HandDetection]:
    """执行手部关键点推理并返回检测结果列表。

    参数:
        model: 已加载的手部关键点模型。
        frame: OpenCV 图像帧（BGR）。
        conf: 置信度阈值，范围 [0.0, 1.0]。
        imgsz: 推理输入尺寸（正整数）。
        device: 推理设备，如 "cpu"、"0"、"0,1" 等；None 表示自动选择。
        return_result: 是否保留原始结果对象。
        top_k: 仅保留置信度最高的前 K 个检测；
               1 表示单手，None 或 <=0 表示保留全部。

    返回:
        list[HandDetection]: 手部关键点检测结果列表。
    """
    results = model.predict(
        source=frame,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    result = results[0]
    if result.keypoints is None:
        return []

    kps_xy = result.keypoints.xy.cpu().numpy()
    kps_conf = None
    if result.keypoints.conf is not None:
        kps_conf = result.keypoints.conf.cpu().numpy()
    scores = None
    if result.boxes is not None and result.boxes.conf is not None:
        scores = result.boxes.conf.cpu().numpy()

    detections: list[HandDetection] = []
    for idx, kp_xy in enumerate(kps_xy):
        kp_conf = kps_conf[idx] if kps_conf is not None else np.ones(kp_xy.shape[0])
        keypoints = [(float(x), float(y), float(conf)) for (x, y), conf in zip(kp_xy, kp_conf)]
        score = float(scores[idx]) if scores is not None else float(np.mean(kp_conf))
        detections.append(
            HandDetection(
                keypoints=keypoints,
                score=score,
                result=result if return_result else None,
            )
        )

    if top_k is not None and top_k > 0 and len(detections) > top_k:
        detections = sorted(detections, key=lambda det: det.score, reverse=True)[:top_k]

    return detections


def infer_hand_on_crop(
    model: YOLO,
    frame: np.ndarray,
    crop: tuple[int, int, int, int],
    conf: float,
    imgsz: int,
    device: str | None,
    top_k: int | None = 1,
) -> HandDetection | None:
    """在指定裁剪区域内推理手部关键点，并映射回原图坐标。

    参数:
        model: 已加载的手部关键点模型。
        frame: 原始图像帧（BGR）。
        crop: 裁剪区域 (x1, y1, x2, y2)。
        conf: 置信度阈值，范围 [0.0, 1.0]。
        imgsz: 推理输入尺寸（正整数）。
        device: 推理设备。
        top_k: 仅保留置信度最高的前 K 个检测。

    返回:
        HandDetection | None: 映射回原图坐标后的检测结果，或 None。
    """
    x1, y1, x2, y2 = crop
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    detections = infer_hand_keypoints(
        model,
        roi,
        conf=conf,
        imgsz=imgsz,
        device=device,
        return_result=False,
        top_k=top_k,
    )
    if not detections:
        return None

    det = detections[0]
    mapped = [(x + x1, y + y1, conf) for x, y, conf in det.keypoints]
    return HandDetection(keypoints=mapped, score=det.score, result=None)
