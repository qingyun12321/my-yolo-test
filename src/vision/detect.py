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


@dataclass(frozen=True)
class PredictOptions:
    """Ultralytics `predict` 的关键参数（命名与官方保持一致）。"""

    iou: float = 0.7
    max_det: int = 300
    agnostic_nms: bool = False
    half: bool = False
    retina_masks: bool = False
    batch: int = 1


def load_det_model(model_arg: str | Path) -> YOLO:
    """加载 YOLO 检测/分割模型。"""
    return YOLO(model_arg)


def _to_name_set(names: Iterable[str] | None) -> set[str]:
    """将类别名序列标准化为小写集合。"""
    if not names:
        return set()
    return {name.strip().lower() for name in names if name and name.strip()}


def _normalize_names_dict(names: Any) -> dict[int, str]:
    """将 YOLO names 统一为 `{class_id: class_name}`。"""
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
    """根据类别名过滤条件，解析需要传给 YOLO 的 `class_id` 列表。"""
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


def _build_predict_kwargs(
    source: Any,
    conf: float,
    imgsz: int,
    device: str | None,
    options: PredictOptions,
    class_ids: list[int] | None,
) -> dict[str, Any]:
    """构造 `YOLO.predict` 参数字典，避免单帧/批量逻辑重复。"""
    kwargs: dict[str, Any] = {
        "source": source,
        "conf": float(max(0.0, min(conf, 1.0))),
        "iou": float(max(0.0, min(options.iou, 1.0))),
        "imgsz": max(32, int(imgsz)),
        "max_det": max(1, int(options.max_det)),
        "agnostic_nms": bool(options.agnostic_nms),
        "half": bool(options.half),
        "retina_masks": bool(options.retina_masks),
        "batch": max(1, int(options.batch)),
        "device": device,
        "verbose": False,
    }
    if class_ids is not None:
        kwargs["classes"] = class_ids
    return kwargs


def _result_to_objects(
    result: Any,
    include_set: set[str],
    exclude_set: set[str],
) -> list[DetectedObject]:
    """将单帧 Ultralytics 结果转换为项目内 `DetectedObject` 列表。"""
    objects: list[DetectedObject] = []
    if result.boxes is None:
        return objects

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
    return objects


def infer_objects(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    imgsz: int,
    device: str | None,
    include_names: Iterable[str] | None = None,
    exclude_names: Iterable[str] = ("person",),
    return_result: bool = False,
    predict_options: PredictOptions | None = None,
) -> DetectionBatch:
    """执行目标检测/分割，并转换为 `DetectedObject` 列表。"""
    options = predict_options or PredictOptions()
    include_set = _to_name_set(include_names)
    exclude_set = _to_name_set(exclude_names)
    class_ids = _resolve_class_ids(
        model,
        include_names=include_set,
        exclude_names=exclude_set,
    )
    if class_ids is not None and len(class_ids) == 0:
        return DetectionBatch(objects=[], result=None)

    predict_kwargs = _build_predict_kwargs(
        source=frame,
        conf=conf,
        imgsz=imgsz,
        device=device,
        options=options,
        class_ids=class_ids,
    )
    results = model.predict(**predict_kwargs)
    result = results[0]
    objects = _result_to_objects(result, include_set=include_set, exclude_set=exclude_set)
    return DetectionBatch(objects=objects, result=result if return_result else None)


def infer_objects_batch(
    model: YOLO,
    frames: Iterable[np.ndarray],
    conf: float,
    imgsz: int,
    device: str | None,
    include_names: Iterable[str] | None = None,
    exclude_names: Iterable[str] = ("person",),
    return_result: bool = False,
    predict_options: PredictOptions | None = None,
) -> list[DetectionBatch]:
    """对多帧/多 ROI 一次性批量推理，减少重复调用带来的 CPU 开销。"""
    frame_list = [frame for frame in frames if frame is not None and frame.size > 0]
    if not frame_list:
        return []

    options = predict_options or PredictOptions()
    include_set = _to_name_set(include_names)
    exclude_set = _to_name_set(exclude_names)
    class_ids = _resolve_class_ids(
        model,
        include_names=include_set,
        exclude_names=exclude_set,
    )
    if class_ids is not None and len(class_ids) == 0:
        return [DetectionBatch(objects=[], result=None) for _ in frame_list]

    predict_kwargs = _build_predict_kwargs(
        source=frame_list,
        conf=conf,
        imgsz=imgsz,
        device=device,
        options=options,
        class_ids=class_ids,
    )
    results = model.predict(**predict_kwargs)

    batches: list[DetectionBatch] = []
    for result in results:
        objects = _result_to_objects(result, include_set=include_set, exclude_set=exclude_set)
        batches.append(
            DetectionBatch(
                objects=objects,
                result=result if return_result else None,
            )
        )

    # 异常情况下补齐空结果，避免上层映射 ROI 时出现越界。
    while len(batches) < len(frame_list):
        batches.append(DetectionBatch(objects=[], result=None))
    return batches[: len(frame_list)]
