from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from ultralytics import YOLO

from ..types import DetectedObject


@dataclass(frozen=True)
class DetectionBatch:
    objects: list[DetectedObject]
    result: Any | None


def load_det_model(model_arg: str | Path) -> YOLO:
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
