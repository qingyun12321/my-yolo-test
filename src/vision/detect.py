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
    """单次检测输出：结构化目标列表 + 可选原始结果。"""

    objects: list[DetectedObject]
    result: Any | None


def load_det_model(model_arg: str | Path) -> YOLO:
    """加载 YOLO 检测/分割模型。"""
    return YOLO(model_arg)


def _to_name_set(names: Iterable[str] | None) -> set[str]:
    """将类别名序列标准化为小写集合。"""
    if not names:
        return set()
    return {name.strip().lower() for name in names if name and name.strip()}


def _normalize_names_dict(names: Any) -> dict[int, str]:
    """将 YOLO names 统一为 {class_id: class_name}。"""
    if not names:
        return {}
    if isinstance(names, dict):
        return {int(idx): str(name) for idx, name in names.items()}
    if isinstance(names, (list, tuple)):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {}


def _resolve_class_ids(
    model: YOLO,
    include_names: set[str],
    exclude_names: set[str],
) -> list[int] | None:
    """根据类别名过滤条件，解析需要传给 YOLO 的 class_id 列表。"""
    model_names = _normalize_names_dict(getattr(model, "names", None))
    if not model_names:
        return None

    available_ids = set(model_names.keys())
    if include_names:
        available_ids &= {idx for idx, name in model_names.items() if name.lower() in include_names}
    if exclude_names:
        available_ids -= {idx for idx, name in model_names.items() if name.lower() in exclude_names}

    if not include_names and not exclude_names:
        return None
    return sorted(available_ids)


def infer_objects(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    imgsz: int,
    device: str | None,
    include_names: Iterable[str] | None = None,
    exclude_names: Iterable[str] = ("person",),
    return_result: bool = False,
) -> DetectionBatch:
    """执行目标检测/分割，并转换为 DetectedObject 列表。"""
    include_set = _to_name_set(include_names)
    exclude_set = _to_name_set(exclude_names)
    class_ids = _resolve_class_ids(
        model,
        include_names=include_set,
        exclude_names=exclude_set,
    )
    if class_ids is not None and len(class_ids) == 0:
        return DetectionBatch(objects=[], result=None)

    predict_kwargs: dict[str, Any] = {
        "source": frame,
        "conf": conf,
        "imgsz": imgsz,
        "device": device,
        "verbose": False,
    }
    if class_ids is not None:
        predict_kwargs["classes"] = class_ids

    results = model.predict(
        **predict_kwargs,
    )
    result = results[0]
    objects: list[DetectedObject] = []

    if result.boxes is None:
        return DetectionBatch(objects=objects, result=result if return_result else None)

    names = _normalize_names_dict(result.names)
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
    classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else None

    masks = None
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()

    for idx, box in enumerate(boxes):
        class_id = int(classes[idx]) if classes is not None else -1
        name = names.get(class_id, str(class_id))
        name_lc = name.lower()
        if include_set and name_lc not in include_set:
            continue
        if name_lc in exclude_set:
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