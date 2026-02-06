from __future__ import annotations

"""主应用入口：动作识别 + 手与物体接触检测。

核心流程：
1. 读取摄像头帧；
2. 执行姿态推理与目标检测/分割；
3. 执行手关键点检测，并通过时序稳定器降低抖动；
4. 基于手关键点生成 ROI，在 ROI 内补充目标检测；
5. 合并全图与 ROI 检测结果，并做去重/冲突抑制；
6. 输出动作与接触判定结果（stdout + 可选 JSONL）。
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .actions.rules import ActionRuleEngine
from .config import (
    coerce_source,
    default_source,
    log_dir,
    prepare_model_arg,
    project_root,
    timestamp,
)
from .contact.hand_object import detect_contact
from .features.hand_skeleton import HAND_EDGES
from .features.skeleton import normalize_keypoints
from .io.capture import open_capture
from .io.output import JsonlEmitter
from .postprocess.object_filter import deduplicate_objects
from .postprocess.object_temporal import ObjectTemporalStabilizer
from .types import ActionResult, ContactResult, DetectedObject, TrackedPerson
from .vision.class_whitelist import load_whitelist_from_config
from .vision.detect import infer_objects, load_det_model
from .vision.hand_mediapipe import (
    create_hand_landmarker,
    detect_hands,
    ensure_hand_model,
    to_left_right_map,
)
from .vision.hand_roi import HandRoiBuilder
from .vision.hand_stabilizer import HandStabilizer
from .vision.pose import infer_pose, load_pose_model
from .vision.track import SimpleTracker


def parse_args() -> argparse.Namespace:
    """解析命令行参数并返回配置对象。"""
    parser = argparse.ArgumentParser(
        description="Lightweight action + contact detection using YOLO pose and detection."
    )
    parser.add_argument(
        "--pose-model",
        default="models/yolo26n-pose.pt",
        help="Pose model path or name (default: models/yolo26n-pose.pt).",
    )
    parser.add_argument(
        "--det-model",
        default="models/yolo26n-seg.pt",
        help="Detection/seg model path or name (default: models/yolo26n-seg.pt).",
    )
    parser.add_argument(
        "--hand-model",
        default="models/hand_landmarker.task",
        help="MediaPipe hand landmarker model path (default: models/hand_landmarker.task).",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Webcam index (e.g. 0) or device path (e.g. /dev/video0).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Confidence threshold for pose/detection.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (e.g. 0, 0,1, or cpu).",
    )

    parser.add_argument(
        "--hand-num",
        type=int,
        default=2,
        help="Max number of hands to detect (>=1).",
    )
    parser.add_argument(
        "--hand-det-conf",
        type=float,
        default=0.5,
        help="Hand detection confidence (0-1).",
    )
    parser.add_argument(
        "--hand-presence-conf",
        type=float,
        default=0.5,
        help="Hand presence confidence (0-1).",
    )
    parser.add_argument(
        "--hand-track-conf",
        type=float,
        default=0.5,
        help="Hand tracking confidence (0-1).",
    )

    parser.add_argument(
        "--hand-stabilize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable temporal stabilization for hand keypoints and left/right assignment.",
    )
    parser.add_argument(
        "--hand-smooth-alpha",
        type=float,
        default=0.55,
        help="Base EMA alpha (0-1) for hand keypoint smoothing.",
    )
    parser.add_argument(
        "--hand-smooth-adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable adaptive alpha to reduce lag during fast hand motion.",
    )
    parser.add_argument(
        "--hand-smooth-fast-alpha",
        type=float,
        default=0.88,
        help="Upper alpha bound used when hand motion is fast (0-1).",
    )
    parser.add_argument(
        "--hand-smooth-motion-scale",
        type=float,
        default=0.12,
        help="Motion ratio scale for adaptive alpha. Smaller means more responsive.",
    )
    parser.add_argument(
        "--hand-hold-frames",
        type=int,
        default=2,
        help="Keep previous hand for N missed frames to reduce flicker.",
    )
    parser.add_argument(
        "--hand-side-merge-ratio",
        type=float,
        default=0.45,
        help=(
            "If left/right hand centers are too close, merge into one side "
            "(distance <= ratio * min(diagonal))."
        ),
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Capture FPS target (>=1).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Output interval in seconds.",
    )

    parser.add_argument(
        "--keypoint-conf",
        type=float,
        default=0.2,
        help="Minimum keypoint confidence for rules.",
    )
    parser.add_argument(
        "--contact-expand",
        type=int,
        default=30,
        help="Expand bbox by N pixels for contact check.",
    )
    parser.add_argument(
        "--contact-dist",
        type=float,
        default=20.0,
        help="Max distance (px) between keypoint and object to count as contact.",
    )
    parser.add_argument(
        "--contact-min-points",
        type=int,
        default=1,
        help="Minimum number of hand points required to confirm contact.",
    )

    parser.add_argument(
        "--det-include",
        default=None,
        help=(
            "Comma-separated object class names to keep only (case-insensitive). "
            "Example: cup,bottle. Empty keeps all classes."
        ),
    )
    parser.add_argument(
        "--det-exclude",
        default="person,hand,left hand,right hand,left_hand,right_hand",
        help=(
            "Comma-separated class names to exclude (case-insensitive). "
            "Default excludes person/hand classes."
        ),
    )
    parser.add_argument(
        "--det-whitelist-config",
        default=None,
        help=(
            "YAML config path to load whitelist classes (usually your segmentation data yaml). "
            "Supports direct `names` or train yaml with `data: xxx.yaml`."
        ),
    )
    parser.add_argument(
        "--det-whitelist-field",
        default="names",
        help=(
            "Field path for whitelist classes inside YAML (default: names). "
            "Supports dotted path, e.g. data.names."
        ),
    )
    parser.add_argument(
        "--det-whitelist-mode",
        choices=("override", "union"),
        default="override",
        help=(
            "How to combine --det-include with config whitelist: "
            "override (use config only) or union (merge both)."
        ),
    )

    parser.add_argument(
        "--obj-dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable object deduplication across full-frame and ROI detections.",
    )
    parser.add_argument(
        "--obj-dedup-iou",
        type=float,
        default=0.45,
        help="IoU threshold (0-1) for considering two same-class objects as duplicates.",
    )
    parser.add_argument(
        "--obj-dedup-center-ratio",
        type=float,
        default=0.35,
        help=(
            "Center-distance ratio for dedup fallback (distance <= ratio * min(diagonal)). "
            "Set 0 to disable this fallback."
        ),
    )
    parser.add_argument(
        "--obj-conflict-suppress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress cross-class nested conflicts (whole object + part-of-object false class).",
    )
    parser.add_argument(
        "--obj-conflict-overlap",
        type=float,
        default=0.75,
        help="Min intersection-over-smaller-area (0-1) to treat two classes as conflicting.",
    )
    parser.add_argument(
        "--obj-conflict-area-ratio",
        type=float,
        default=1.8,
        help="Min larger/smaller area ratio for cross-class conflict suppression.",
    )
    parser.add_argument(
        "--obj-conflict-score-gap",
        type=float,
        default=0.15,
        help="If score gap exceeds this, keep higher score in cross-class conflict.",
    )
    parser.add_argument(
        "--obj-temporal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable temporal stabilizer for detected objects.",
    )
    parser.add_argument(
        "--obj-temporal-hold-frames",
        type=int,
        default=3,
        help="Keep objects for N frames when temporarily occluded/missed.",
    )
    parser.add_argument(
        "--obj-temporal-min-hits",
        type=int,
        default=1,
        help="Minimum matched frames before an object is emitted.",
    )
    parser.add_argument(
        "--obj-temporal-iou",
        type=float,
        default=0.32,
        help="Minimum IoU to match object tracks across frames.",
    )
    parser.add_argument(
        "--obj-temporal-center-ratio",
        type=float,
        default=0.72,
        help=(
            "Fallback center-distance ratio for temporal match "
            "(distance <= ratio * min(diagonal))."
        ),
    )
    parser.add_argument(
        "--obj-temporal-bbox-alpha",
        type=float,
        default=0.62,
        help="EMA alpha (0-1) for temporal bbox/score smoothing.",
    )
    parser.add_argument(
        "--obj-temporal-class-decay",
        type=float,
        default=0.9,
        help="Decay factor (0-1) for class voting history in temporal tracks.",
    )
    parser.add_argument(
        "--obj-temporal-score-decay",
        type=float,
        default=0.9,
        help="Per-frame score decay (0-1) for held tracks when missed.",
    )

    parser.add_argument(
        "--hand-roi-det",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable hand ROI object detection (default: True).",
    )
    parser.add_argument(
        "--hand-roi-only",
        action="store_true",
        help="Use ONLY hand ROI objects for contact (skip full-frame objects).",
    )
    parser.add_argument(
        "--hand-roi-padding",
        type=float,
        default=0.35,
        help="Hand ROI padding ratio based on hand box size (0-1).",
    )
    parser.add_argument(
        "--hand-roi-min-size",
        type=int,
        default=96,
        help="Minimum hand ROI size in pixels (>=32).",
    )
    parser.add_argument(
        "--hand-roi-min-size-ratio",
        type=float,
        default=0.12,
        help="Minimum hand ROI size as frame short-side ratio (0-1).",
    )
    parser.add_argument(
        "--hand-roi-context-scale",
        type=float,
        default=1.9,
        help="Scale multiplier for an additional context ROI per hand.",
    )
    parser.add_argument(
        "--hand-roi-forward-shift",
        type=float,
        default=0.42,
        help="Shift context ROI along hand forward direction (relative to hand ROI size).",
    )
    parser.add_argument(
        "--hand-roi-inward-scale",
        type=float,
        default=1.18,
        help="Scale multiplier for an inward (toward wrist) compensation ROI per hand.",
    )
    parser.add_argument(
        "--hand-roi-inward-shift",
        type=float,
        default=0.1,
        help="Shift inward compensation ROI opposite to hand forward direction.",
    )
    parser.add_argument(
        "--hand-roi-direction-smooth",
        type=float,
        default=0.35,
        help="Smoothing factor (0-1) for ROI forward direction stabilization.",
    )
    parser.add_argument(
        "--hand-roi-merge-iou",
        type=float,
        default=0.7,
        help="IoU threshold to merge overlapping ROIs for each hand.",
    )
    parser.add_argument(
        "--hand-roi-global-merge-iou",
        type=float,
        default=0.78,
        help="IoU threshold to merge ROIs across both hands (suppresses duplicate ROIs).",
    )
    parser.add_argument(
        "--hand-roi-cross-side-merge-ratio",
        type=float,
        default=0.45,
        help=(
            "If left/right ROI seeds are too close, collapse them as one hand "
            "(distance <= ratio * min(size))."
        ),
    )
    parser.add_argument(
        "--hand-roi-hold-frames",
        type=int,
        default=2,
        help="Keep previous ROI for N frames when keypoints are temporarily missing.",
    )
    parser.add_argument(
        "--hand-roi-shrink-floor",
        type=float,
        default=0.9,
        help="Per-frame minimum ROI size ratio vs previous frame (prevents sudden tiny ROI).",
    )
    parser.add_argument(
        "--hand-roi-size-smooth",
        type=float,
        default=0.35,
        help="Smoothing factor for ROI center/size (0-1). Higher means more responsive.",
    )

    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable all debug overlays (ROI boxes, counters, diagnostics).",
    )
    parser.add_argument(
        "--hand-roi-debug",
        action="store_true",
        help="Show hand ROI boxes in preview.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable preview window.",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="Optional JSONL log path. Default: output/logs/action_contact_<timestamp>.jsonl",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable log file output (stdout still enabled).",
    )

    return parser.parse_args()


def _parse_name_csv(value: str | None) -> tuple[str, ...]:
    """将逗号分隔的类别名字符串解析为元组。"""
    if value is None:
        return tuple()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _merge_name_tuples(
    primary: tuple[str, ...],
    secondary: tuple[str, ...],
) -> tuple[str, ...]:
    """合并两组类别名并按大小写不敏感去重。"""
    merged: list[str] = []
    seen: set[str] = set()
    for name in (*primary, *secondary):
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(name)
    return tuple(merged)


def main() -> int:
    """主程序入口。

    返回：
        int: 0 表示正常退出。
    """
    args = parse_args()
    det_include_names = _parse_name_csv(args.det_include)
    det_exclude_names = _parse_name_csv(args.det_exclude)

    if args.det_whitelist_config:
        whitelist_path = Path(args.det_whitelist_config).expanduser()
        if not whitelist_path.is_absolute():
            whitelist_path = (project_root() / whitelist_path).resolve()

        whitelist_names = load_whitelist_from_config(
            whitelist_path,
            field_path=args.det_whitelist_field,
        )
        if args.det_whitelist_mode == "union":
            det_include_names = _merge_name_tuples(det_include_names, whitelist_names)
        else:
            det_include_names = whitelist_names
        if not det_include_names:
            raise RuntimeError(
                "det whitelist config 已加载，但解析结果为空。"
                f" path={whitelist_path}, field={args.det_whitelist_field}"
            )

    source_arg = args.source if args.source is not None else default_source()
    source = coerce_source(source_arg)

    pose_model_arg = prepare_model_arg(args.pose_model)
    det_model_arg = prepare_model_arg(args.det_model)
    pose_model = load_pose_model(pose_model_arg)
    det_model = load_det_model(det_model_arg)

    hand_model_arg = Path(args.hand_model).expanduser()
    if not hand_model_arg.is_absolute():
        hand_model_arg = (project_root() / hand_model_arg).resolve()
    hand_model_path = ensure_hand_model(hand_model_arg)
    hand_landmarker = create_hand_landmarker(
        hand_model_path,
        num_hands=args.hand_num,
        min_detection_confidence=args.hand_det_conf,
        min_presence_confidence=args.hand_presence_conf,
        min_tracking_confidence=args.hand_track_conf,
    )

    cap, backend_name = open_capture(source)
    if not cap:
        raise RuntimeError(f"Unable to open webcam source: {source_arg}")

    # 尝试设置采集帧率；部分设备可能忽略该设置。
    if args.fps and args.fps >= 1:
        cap.set(cv2.CAP_PROP_FPS, float(args.fps))

    tracker = SimpleTracker(iou_threshold=0.3, max_age=1.0)
    action_engine = ActionRuleEngine(
        max_seconds=2.0,
        min_frames=6,
        keypoint_conf=args.keypoint_conf,
    )

    emitters = [JsonlEmitter()]
    log_handle = None
    if not args.no_log:
        if args.log_path:
            log_path = Path(args.log_path)
            if not log_path.is_absolute():
                log_path = (project_root() / log_path).resolve()
        else:
            log_path = log_dir() / f"action_contact_{timestamp()}.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        emitters.append(JsonlEmitter(log_handle))

    show_preview = not args.no_preview
    debug_all = bool(args.debug)
    roi_debug = debug_all or args.hand_roi_debug

    hand_stabilizer = HandStabilizer(
        smooth_alpha=args.hand_smooth_alpha,
        hold_frames=args.hand_hold_frames,
        side_merge_ratio=args.hand_side_merge_ratio,
        adaptive_alpha=args.hand_smooth_adaptive,
        fast_alpha=args.hand_smooth_fast_alpha,
        motion_scale=args.hand_smooth_motion_scale,
    )
    hand_roi_builder = HandRoiBuilder(
        padding_ratio=args.hand_roi_padding,
        min_size=args.hand_roi_min_size,
        min_size_ratio=args.hand_roi_min_size_ratio,
        context_scale=args.hand_roi_context_scale,
        forward_shift=args.hand_roi_forward_shift,
        inward_scale=args.hand_roi_inward_scale,
        inward_shift=args.hand_roi_inward_shift,
        direction_smooth=args.hand_roi_direction_smooth,
        merge_iou=args.hand_roi_merge_iou,
        hold_frames=args.hand_roi_hold_frames,
        shrink_floor=args.hand_roi_shrink_floor,
        size_smooth=args.hand_roi_size_smooth,
        global_merge_iou=args.hand_roi_global_merge_iou,
        cross_side_merge_ratio=args.hand_roi_cross_side_merge_ratio,
    )
    object_temporal = None
    if args.obj_temporal:
        object_temporal = ObjectTemporalStabilizer(
            hold_frames=args.obj_temporal_hold_frames,
            min_hits=args.obj_temporal_min_hits,
            match_iou=args.obj_temporal_iou,
            match_center_ratio=args.obj_temporal_center_ratio,
            bbox_alpha=args.obj_temporal_bbox_alpha,
            class_decay=args.obj_temporal_class_decay,
            score_decay=args.obj_temporal_score_decay,
        )

    if det_include_names:
        print("Detection include classes:", ", ".join(det_include_names))

    last_emit = 0.0
    frame_index = 0
    prev_frame_ts = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            now = time.time()
            # 姿态推理：默认仅保留置信度最高的单人结果。
            pose_batch = infer_pose(
                pose_model,
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                return_result=show_preview,
                top_k=1,
            )

            # 全图目标检测/分割；若 hand_roi_only 开启，则跳过全图检测。
            det_batch = None
            if not args.hand_roi_only:
                det_batch = infer_objects(
                    det_model,
                    frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=args.device,
                    include_names=det_include_names,
                    exclude_names=det_exclude_names,
                    return_result=False,
                )

            # 单人追踪：用于维持动作状态的时间连续性。
            persons = tracker.update(pose_batch.detections, now)

            actions: dict[int, ActionResult] = {}
            contacts: dict[int, ContactResult] = {}
            hand_points: dict[int, dict[str, list[tuple[float, float, float]] | None]] = {}
            roi_objects: list[DetectedObject] = []
            rois: list[tuple[int, int, int, int]] = []

            timestamp_ms = int(now * 1000)
            hands = detect_hands(hand_landmarker, frame, timestamp_ms)
            if args.hand_stabilize:
                hands_lr = hand_stabilizer.update(hands, frame_shape=frame.shape[:2])
            else:
                hands_lr = to_left_right_map(hands)

            if hands and hands_lr.get("left") is None and hands_lr.get("right") is None:
                # 兜底策略：稳定器短时给空时，回退到原始 left/right 映射。
                hands_lr = to_left_right_map(hands)

            use_roi_det = args.hand_roi_det or args.hand_roi_only
            has_hand_roi_seed = hands_lr.get("left") is not None or hands_lr.get("right") is not None
            if use_roi_det and has_hand_roi_seed:
                rois = hand_roi_builder.build(hands_lr, frame.shape[:2])
                roi_objects = _infer_objects_on_rois(
                    det_model,
                    frame,
                    rois,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=args.device,
                    include_names=det_include_names,
                    exclude_names=det_exclude_names,
                )

            full_frame_objects = det_batch.objects if det_batch else []
            all_objects = roi_objects if args.hand_roi_only else (full_frame_objects + roi_objects)
            raw_object_count = len(all_objects)
            dedup_object_count = raw_object_count

            if args.obj_dedup:
                all_objects = deduplicate_objects(
                    all_objects,
                    iou_threshold=args.obj_dedup_iou,
                    center_ratio=args.obj_dedup_center_ratio,
                    conflict_suppress=args.obj_conflict_suppress,
                    conflict_overlap=args.obj_conflict_overlap,
                    conflict_area_ratio=args.obj_conflict_area_ratio,
                    conflict_score_gap=args.obj_conflict_score_gap,
                )
            dedup_object_count = len(all_objects)

            if object_temporal is not None:
                objects_for_contact = object_temporal.update(all_objects)
            else:
                objects_for_contact = all_objects

            for person in persons:
                # 当前默认单人场景，直接复用全帧手结果到当前 track。
                hand_points[person.track_id] = hands_lr
                normalized = normalize_keypoints(person.keypoints, args.keypoint_conf)
                action_engine.update(person.track_id, normalized, now)
                actions[person.track_id] = action_engine.classify(person.track_id)
                contacts[person.track_id] = detect_contact(
                    person.keypoints,
                    objects_for_contact,
                    min_conf=args.keypoint_conf,
                    expand=args.contact_expand,
                    hand_keypoints=hand_points.get(person.track_id),
                    dist_threshold=args.contact_dist,
                    min_points=args.contact_min_points,
                )

            # 输出节流：interval <= 0 表示每帧输出。
            if args.interval <= 0:
                should_emit = True
            else:
                should_emit = (now - last_emit) >= args.interval

            # 仅当存在人体时输出记录，避免空帧刷日志。
            if should_emit and persons:
                records = [
                    _build_record(
                        now,
                        frame_index,
                        person,
                        actions.get(person.track_id),
                        contacts.get(person.track_id),
                        hand_points.get(person.track_id),
                    )
                    for person in persons
                ]
                for emitter in emitters:
                    emitter.emit_many(records)
                last_emit = now

            # 可视化预览：叠加姿态、目标、手关键点和调试信息。
            if show_preview:
                annotated = frame
                if pose_batch.result is not None:
                    annotated = pose_batch.result.plot()
                if roi_debug and rois:
                    _draw_rois(annotated, rois, color=(255, 128, 0))
                _draw_overlay(
                    annotated,
                    persons,
                    actions,
                    contacts,
                    objects_for_contact,
                    hand_points,
                    global_hands=hands_lr,
                )
                if debug_all:
                    dt = max(1e-6, now - prev_frame_ts)
                    _draw_debug_lines(
                        annotated,
                        lines=[
                            f"fps:{1.0 / dt:.1f}",
                            (
                                "obj raw:"
                                f"{raw_object_count} "
                                f"dedup:{dedup_object_count} "
                                f"temporal:{len(objects_for_contact)}"
                            ),
                            (
                                "hands raw:"
                                f"{len(hands)} "
                                f"lr:{int(hands_lr.get('left') is not None)}/"
                                f"{int(hands_lr.get('right') is not None)}"
                            ),
                            f"rois:{len(rois)}",
                        ],
                    )
                cv2.imshow("Action + Contact", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            prev_frame_ts = now
            frame_index += 1

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        if log_handle is not None:
            log_handle.close()

    print(f"Capture ended (backend: {backend_name}).")
    return 0


def _build_record(
    timestamp: float,
    frame_index: int,
    person: TrackedPerson,
    action: ActionResult | None,
    contact: ContactResult | None,
    hands: dict[str, list[tuple[float, float, float]] | None] | None,
) -> dict:
    """构建单条结构化输出记录（用于 stdout/JSONL）。"""
    action = action or ActionResult(action="none", confidence=0.0)
    contact = contact or ContactResult(
        active=False,
        object_name=None,
        object_score=None,
        wrist_name=None,
    )
    return {
        "ts": round(timestamp, 3),
        "frame": frame_index,
        "person_id": person.track_id,
        "action": action.action,
        "action_conf": round(action.confidence, 3),
        "contact": {
            "active": contact.active,
            "object": contact.object_name,
            "score": None if contact.object_score is None else round(contact.object_score, 3),
            "wrist": contact.wrist_name,
        },
        "hands": _serialize_hands(hands),
    }


def _draw_overlay(
    frame,
    persons: list[TrackedPerson],
    actions: dict[int, ActionResult],
    contacts: dict[int, ContactResult],
    objects,
    hands: dict[int, dict[str, list[tuple[float, float, float]] | None]],
    global_hands: dict[str, list[tuple[float, float, float]] | None] | None = None,
) -> None:
    """在预览画面绘制目标框、动作标签、接触结果与手关键点。"""
    for obj_idx, obj in enumerate(objects):
        x1, y1, x2, y2 = [int(v) for v in obj.bbox]
        if obj.mask is not None:
            _draw_mask_edges(frame, obj.mask, color=(0, 200, 0), thickness=2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label_y = y2 + 18 + obj_idx * 14
        _draw_label(
            frame,
            f"{obj.name} {obj.score:.2f}",
            x1,
            label_y,
            text_color=(0, 90, 0),
            bg_color=(180, 255, 180),
            font_scale=0.5,
            thickness=1,
        )

    for person_idx, person in enumerate(persons):
        x1, y1, x2, y2 = [int(v) for v in person.bbox]
        action = actions.get(person.track_id, ActionResult("none", 0.0))
        contact = contacts.get(person.track_id, ContactResult(False, None, None, None))
        label = f"ID {person.track_id} {action.action} {action.confidence:.2f}"
        if contact.active and contact.object_name:
            label += f" contact:{contact.object_name}"
        label_y = y1 - 10 - person_idx * 18
        _draw_label(
            frame,
            label,
            x1,
            label_y,
            text_color=(20, 20, 20),
            bg_color=(255, 255, 0),
            font_scale=0.6,
            thickness=2,
        )

    # 手关键点绘制与人体框解耦：即使姿态框短时不稳定也尽量保持可见。
    if global_hands:
        drew = False
        left_kps = global_hands.get("left")
        right_kps = global_hands.get("right")
        if left_kps:
            _draw_hand_keypoints(frame, left_kps, color=(0, 128, 255))
            drew = True
        if right_kps:
            _draw_hand_keypoints(frame, right_kps, color=(0, 255, 128))
            drew = True
        if drew:
            return

    # 兼容回退：若 global_hands 为空，则从按人缓存中取一份手数据绘制。
    for hand_data in hands.values():
        if hand_data:
            _draw_hand_keypoints(frame, hand_data.get("left"), color=(0, 128, 255))
            _draw_hand_keypoints(frame, hand_data.get("right"), color=(0, 255, 128))
            break


def _draw_label(
    frame,
    text: str,
    x: int,
    y: int,
    text_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    """绘制带背景底色的文本标签，避免复杂背景下文字难以辨认。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    height, width = frame.shape[:2]
    x = max(0, min(x, width - text_w - 4))
    y = max(text_h + 4, min(y, height - 4))
    top_left = (x, y - text_h - 4)
    bottom_right = (x + text_w + 4, y + baseline + 2)
    cv2.rectangle(frame, top_left, bottom_right, bg_color, thickness=-1)
    cv2.putText(frame, text, (x + 2, y), font, font_scale, text_color, thickness)


def _draw_debug_lines(
    frame,
    lines: list[str],
) -> None:
    """在左上角绘制紧凑调试面板（FPS、目标数、ROI 数等）。"""
    if not lines:
        return
    x = 10
    y = 22
    for line in lines:
        _draw_label(
            frame,
            line,
            x,
            y,
            text_color=(20, 20, 20),
            bg_color=(210, 255, 210),
            font_scale=0.5,
            thickness=1,
        )
        y += 18


def _draw_mask_edges(
    frame,
    mask,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """绘制分割掩码的轮廓线，用于直观显示分割边界。"""
    mask_uint8 = (mask > 0.5).astype("uint8") * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(frame, contours, -1, color, thickness)


def _draw_hand_keypoints(
    frame,
    keypoints: list[tuple[float, float, float]] | None,
    color: tuple[int, int, int],
    conf_threshold: float = 0.0,
) -> None:
    """绘制手关键点和骨架连线。"""
    if not keypoints:
        return
    for i, j in HAND_EDGES:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        x1, y1, c1 = keypoints[i]
        x2, y2, c2 = keypoints[j]
        if c1 >= conf_threshold and c2 >= conf_threshold:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    for x, y, conf in keypoints:
        if conf >= conf_threshold:
            cv2.circle(frame, (int(x), int(y)), 2, color, -1)


def _infer_objects_on_rois(
    det_model,
    frame: np.ndarray,
    rois: list[tuple[int, int, int, int]],
    conf: float,
    imgsz: int,
    device: str | None,
    include_names: tuple[str, ...],
    exclude_names: tuple[str, ...],
) -> list[DetectedObject]:
    """在多个 ROI 上执行检测，并将结果映射回原图坐标系。"""
    results: list[DetectedObject] = []
    h, w = frame.shape[:2]
    for roi in rois:
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        batch = infer_objects(
            det_model,
            crop,
            conf=conf,
            imgsz=imgsz,
            device=device,
            include_names=include_names,
            exclude_names=exclude_names,
            return_result=False,
        )
        for obj in batch.objects:
            bx1, by1, bx2, by2 = obj.bbox
            mapped_bbox = (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)
            mapped_mask = None
            if obj.mask is not None:
                # 将 ROI 内的掩码映射回整帧坐标，便于统一后处理。
                mask_full = np.zeros((h, w), dtype=obj.mask.dtype)
                roi_h = y2 - y1
                roi_w = x2 - x1
                mask = obj.mask
                # 若模型输出掩码尺寸与 ROI 尺寸不一致，先 resize 再回贴。
                if mask.shape[:2] != (roi_h, roi_w):
                    mask = cv2.resize(mask, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                rx2 = min(x1 + roi_w, w)
                ry2 = min(y1 + roi_h, h)
                mask_full[y1:ry2, x1:rx2] = mask[: ry2 - y1, : rx2 - x1]
                mapped_mask = mask_full
            results.append(
                DetectedObject(
                    name=obj.name,
                    bbox=mapped_bbox,
                    score=obj.score,
                    mask=mapped_mask,
                )
            )
    return results


def _draw_rois(
    frame,
    rois: list[tuple[int, int, int, int]],
    color: tuple[int, int, int],
) -> None:
    """绘制 ROI 矩形（调试用途）。"""
    for x1, y1, x2, y2 in rois:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)


def _serialize_hands(
    hands: dict[str, list[tuple[float, float, float]] | None] | None,
) -> dict[str, list[dict] | None] | None:
    """将手关键点转换为可 JSON 序列化的数据结构。"""
    if hands is None:
        return None
    output: dict[str, list[dict] | None] = {}
    for side, kps in hands.items():
        if not kps:
            output[side] = None
            continue
        output[side] = [
            {"x": round(x, 2), "y": round(y, 2), "conf": round(conf, 3)}
            for x, y, conf in kps
        ]
    return output


if __name__ == "__main__":
    raise SystemExit(main())
