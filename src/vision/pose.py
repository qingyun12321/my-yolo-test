from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ultralytics import YOLO

from ..types import PoseDetection


@dataclass(frozen=True)
class PoseBatch:
    detections: list[PoseDetection]
    result: Any | None


def load_pose_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def infer_pose(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    imgsz: int,
    device: str | None,
    return_result: bool = False,
) -> PoseBatch:
    results = model.predict(
        frame,
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

    return PoseBatch(detections=detections, result=result if return_result else None)
