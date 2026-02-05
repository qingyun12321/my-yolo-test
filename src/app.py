from __future__ import annotations

"""主应用入口：动作 + 物体接触检测。

流程：
1) 加载姿态模型与检测/分割模型
2) 读取摄像头帧
3) 计算姿态 + 物体
4) 基于关键点规则识别动作
5) 判断手与物体接触
6) 输出结果（stdout + 可选 JSONL 文件）并可视化
"""

import argparse
import time
from pathlib import Path

import cv2

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
from .types import ActionResult, ContactResult, TrackedPerson
from .vision.detect import infer_objects, load_det_model
from .vision.hand_mediapipe import (
    create_hand_landmarker,
    detect_hands,
    ensure_hand_model,
    to_left_right_map,
)
from .vision.pose import infer_pose, load_pose_model
from .vision.track import SimpleTracker


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    返回:
        argparse.Namespace: 解析后的参数对象。
    """
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


def main() -> int:
    """主程序入口。

    返回:
        int: 0 表示正常退出，非 0 表示异常退出。
    """
    args = parse_args()
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

    # 尝试设置摄像头采集帧率（部分设备可能会忽略该设置）
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

    last_emit = 0.0
    frame_index = 0
    show_preview = not args.no_preview

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            now = time.time()
            # 姿态推理：默认只保留置信度最高的单人结果
            pose_batch = infer_pose(
                pose_model,
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                return_result=show_preview,
                top_k=1,
            )
            # 物体检测/分割推理
            det_batch = infer_objects(
                det_model,
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                return_result=False,
            )

            # 追踪单人（当前默认 top-1），用于持续的动作识别
            persons = tracker.update(pose_batch.detections, now)
            actions: dict[int, ActionResult] = {}
            contacts: dict[int, ContactResult] = {}
            hand_points: dict[int, dict[str, list[tuple[float, float, float]] | None]] = {}
            timestamp_ms = int(now * 1000)
            hands = detect_hands(hand_landmarker, frame, timestamp_ms)
            hands_lr = to_left_right_map(hands)

            for person in persons:
                # 归一化关键点，便于动作规则处理
                normalized = normalize_keypoints(person.keypoints, args.keypoint_conf)
                action_engine.update(person.track_id, normalized, now)
                actions[person.track_id] = action_engine.classify(person.track_id)
                contacts[person.track_id] = detect_contact(
                    person.keypoints,
                    det_batch.objects,
                    min_conf=args.keypoint_conf,
                    expand=args.contact_expand,
                    hand_keypoints=hand_points.get(person.track_id),
                    dist_threshold=args.contact_dist,
                    min_points=args.contact_min_points,
                )
                # MediaPipe 已内置检测 + 追踪裁剪策略，直接使用全帧检测结果
                hand_points[person.track_id] = hands_lr

            # 结果输出节流：interval <= 0 表示每帧输出
            if args.interval <= 0:
                should_emit = True
            else:
                should_emit = (now - last_emit) >= args.interval

            # 仅当存在检测到的人时输出记录
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

            # 可视化预览
            if show_preview:
                annotated = frame
                if pose_batch.result is not None:
                    annotated = pose_batch.result.plot()
                _draw_overlay(
                    annotated,
                    persons,
                    actions,
                    contacts,
                    det_batch.objects,
                    hand_points,
                )
                cv2.imshow("Action + Contact", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

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
    """构建输出记录（JSONL 字典）。

    参数:
        timestamp: 当前时间戳（秒）。
        frame_index: 帧序号（从 0 开始）。
        person: 追踪到的人员信息。
        action: 动作识别结果（可为 None）。
        contact: 接触检测结果（可为 None）。
        hands: 手部关键点（左右手，可能为 None）。

    返回:
        dict: 结构化输出记录。
    """
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
) -> None:
    """在画面上绘制检测框、动作与接触标记。

    参数:
        frame: 当前图像帧（OpenCV BGR）。
        persons: 追踪到的人员列表。
        actions: track_id -> 动作识别结果。
        contacts: track_id -> 接触检测结果。
        objects: 物体检测结果列表。
        hands: track_id -> 左右手关键点。
    """
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
        hand_data = hands.get(person.track_id)
        if hand_data:
            _draw_hand_keypoints(frame, hand_data.get("left"), color=(0, 128, 255))
            _draw_hand_keypoints(frame, hand_data.get("right"), color=(0, 255, 128))


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
    """绘制带背景色的文本标签。

    参数:
        frame: OpenCV 图像帧（BGR）。
        text: 文本内容。
        x: 文本左上角 X 坐标（像素）。
        y: 文本基线 Y 坐标（像素）。
        text_color: 字体颜色（B, G, R）。
        bg_color: 背景色（B, G, R）。
        font_scale: 字体缩放比例（>0）。
        thickness: 字体线宽（>=1）。
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    height, width = frame.shape[:2]
    x = max(0, min(x, width - text_w - 4))
    y = max(text_h + 4, min(y, height - 4))
    top_left = (x, y - text_h - 4)
    bottom_right = (x + text_w + 4, y + baseline + 2)
    cv2.rectangle(frame, top_left, bottom_right, bg_color, thickness=-1)
    cv2.putText(frame, text, (x + 2, y), font, font_scale, text_color, thickness)


def _draw_mask_edges(
    frame,
    mask,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """绘制分割 mask 的边缘轮廓。

    参数:
        frame: OpenCV 图像帧（BGR）。
        mask: 二值或概率 mask（H, W）。
        color: 边缘颜色（B, G, R）。
        thickness: 线宽（>=1）。
    """
    mask_uint8 = (mask > 0.5).astype("uint8") * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(frame, contours, -1, color, thickness)


def _draw_hand_keypoints(
    frame,
    keypoints: list[tuple[float, float, float]] | None,
    color: tuple[int, int, int],
    conf_threshold: float = 0.1,
) -> None:
    """绘制手部关键点与连线。

    参数:
        frame: OpenCV 图像帧（BGR）。
        keypoints: 手部关键点列表（21 点）。
        color: 绘制颜色（B, G, R）。
        conf_threshold: 绘制阈值（0-1），小于该值不画点。
    """
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


def _serialize_hands(
    hands: dict[str, list[tuple[float, float, float]] | None] | None,
) -> dict[str, list[dict] | None] | None:
    """将手部关键点转换为可 JSON 化的结构。"""
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
