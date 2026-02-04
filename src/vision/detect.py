from __future__ import annotations

"""目标检测/分割推理与结果整理。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from ultralytics import YOLO

from ..types import DetectedObject


@dataclass(frozen=True)
class DetectionBatch:
    """检测批次结果。"""
    objects: list[DetectedObject]
    result: Any | None


def load_det_model(model_arg: str | Path) -> YOLO:
    """加载检测/分割模型。

    参数:
        model_arg: 模型路径或模型名（如 "yolo26n-seg.pt"）。

    返回:
        YOLO: Ultralytics YOLO 模型实例。
    """
    return YOLO(model_arg)


def infer_objects(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    imgsz: int,
    device: str | None,
    exclude_names: Iterable[str] = ("person",),
    return_result: bool = False,
) -> DetectionBatch:
    """执行目标检测/分割并整理为对象列表。

    参数:
        model: 已加载的 YOLO 模型。
        frame: OpenCV 图像帧（BGR）。
        conf: 置信度阈值，范围 [0.0, 1.0]。
        imgsz: 推理输入尺寸（正整数）。
        device: 推理设备，如 "cpu"、"0"、"0,1" 等；None 表示自动选择。
        exclude_names: 需要过滤的类别名集合（默认过滤 "person"）。
        return_result: 是否返回原始结果对象（用于可视化）。

    返回:
        DetectionBatch: 检测对象与可选原始结果。
    """
    results = model.predict(
        source=frame,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    result = results[0]
    objects: list[DetectedObject] = []

    if result.boxes is None:
        return DetectionBatch(objects=objects, result=result if return_result else None)

    names = result.names or {}
    if isinstance(names, (list, tuple)):
        names = {idx: name for idx, name in enumerate(names)}
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
    classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else None

    masks = None
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()

    for idx, box in enumerate(boxes):
        class_id = int(classes[idx]) if classes is not None else -1
        name = names.get(class_id, str(class_id))
        if name in exclude_names:
            continue
        score = float(scores[idx]) if scores is not None else 0.0
        mask = None
        if masks is not None and idx < masks.shape[0]:
            mask = masks[idx]
        objects.append(
            DetectedObject(
                name=name,
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                score=score,
                mask=mask,
            )
        )

    return DetectionBatch(objects=objects, result=result if return_result else None)
