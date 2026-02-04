from __future__ import annotations

"""姿态估计推理与结果整理。

说明：
    通过 Ultralytics YOLO 姿态模型获取人体检测框与关键点，
    并转换为项目内部的数据结构。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from ..types import PoseDetection


@dataclass(frozen=True)
class PoseBatch:
    """姿态推理批次结果。"""
    detections: list[PoseDetection]
    result: Any | None


def load_pose_model(model_arg: str | Path) -> YOLO:
    """加载姿态模型。

    参数:
        model_arg: 模型路径或模型名（如 "yolo26n-pose.pt"）。

    返回:
        YOLO: Ultralytics YOLO 模型实例。
    """
    return YOLO(model_arg)


def infer_pose(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    imgsz: int,
    device: str | None,
    return_result: bool = False,
    top_k: int | None = 1,
) -> PoseBatch:
    """执行姿态推理并返回结构化结果。

    参数:
        model: 已加载的 YOLO 姿态模型。
        frame: OpenCV 图像帧（BGR）。
        conf: 置信度阈值，范围 [0.0, 1.0]。
        imgsz: 推理输入尺寸（正整数，如 640）。
        device: 推理设备，如 "cpu"、"0"、"0,1" 等；None 表示自动选择。
        return_result: 是否保留原始结果对象（用于可视化）。
        top_k: 仅保留置信度最高的前 K 个检测；
               1 表示单人，None 或 <=0 表示保留全部。

    返回:
        PoseBatch: 姿态检测列表与可选原始结果。
    """
    results = model.predict(
        source=frame,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    result = results[0]
    detections: list[PoseDetection] = []

    if result.boxes is None or result.keypoints is None:
        return PoseBatch(detections=detections, result=result if return_result else None)

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
    kps_xy = result.keypoints.xy.cpu().numpy()
    kps_conf = None
    if result.keypoints.conf is not None:
        kps_conf = result.keypoints.conf.cpu().numpy()

    for idx, box in enumerate(boxes):
        score = float(scores[idx]) if scores is not None else 0.0
        kp_xy = kps_xy[idx]
        kp_conf = kps_conf[idx] if kps_conf is not None else np.ones(kp_xy.shape[0])
        keypoints = [
            (float(x), float(y), float(conf))
            for (x, y), conf in zip(kp_xy, kp_conf)
        ]
        detections.append(
            PoseDetection(
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                keypoints=keypoints,
                score=score,
            )
        )

    # 默认仅保留最高置信度的单人检测，便于“单人模式”
    if top_k is not None and top_k > 0 and len(detections) > top_k:
        detections = sorted(detections, key=lambda det: det.score, reverse=True)[:top_k]

    return PoseBatch(detections=detections, result=result if return_result else None)
