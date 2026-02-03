from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .actions.rules import ActionRuleEngine
from .config import (
    coerce_source,
    default_source,
    ensure_model_path,
    log_dir,
    project_root,
    resolve_model_path,
    timestamp,
)
from .contact.hand_object import detect_contact
from .features.skeleton import normalize_keypoints
from .io.capture import open_capture
from .io.output import JsonlEmitter
from .types import ActionResult, ContactResult, TrackedPerson
from .vision.detect import infer_objects, load_det_model
from .vision.pose import infer_pose, load_pose_model
from .vision.track import SimpleTracker


def parse_args() -> argparse.Namespace:
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
        "--source",
        default=None,
        help="Webcam index (e.g. 0) or device path (e.g. /dev/video0).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
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
        "--interval",
        type=float,
        default=0.5,
        help="Output interval in seconds.",
    )
    parser.add_argument(
        "--keypoint-conf",
        type=float,
        default=0.3,
        help="Minimum keypoint confidence for rules.",
    )
    parser.add_argument(
        "--contact-expand",
        type=int,
        default=10,
        help="Expand bbox by N pixels for contact check.",
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
    args = parse_args()
    source_arg = args.source if args.source is not None else default_source()
    source = coerce_source(source_arg)

    pose_model_path = ensure_model_path(resolve_model_path(args.pose_model))
    det_model_path = ensure_model_path(resolve_model_path(args.det_model))

    pose_model = load_pose_model(str(pose_model_path))
    det_model = load_det_model(str(det_model_path))

    cap, backend_name = open_capture(source)
    if not cap:
        raise RuntimeError(f"Unable to open webcam source: {source_arg}")

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
            pose_batch = infer_pose(
                pose_model,
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                return_result=show_preview,
            )
            det_batch = infer_objects(
                det_model,
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                return_result=False,
            )

            persons = tracker.update(pose_batch.detections, now)
            actions: dict[int, ActionResult] = {}
            contacts: dict[int, ContactResult] = {}

            for person in persons:
                normalized = normalize_keypoints(person.keypoints, args.keypoint_conf)
                action_engine.update(person.track_id, normalized, now)
                actions[person.track_id] = action_engine.classify(person.track_id)
                contacts[person.track_id] = detect_contact(
                    person.keypoints,
                    det_batch.objects,
                    min_conf=args.keypoint_conf,
                    expand=args.contact_expand,
                )

            if args.interval <= 0:
                should_emit = True
            else:
                should_emit = (now - last_emit) >= args.interval

            if should_emit and persons:
                records = [
                    _build_record(
                        now,
                        frame_index,
                        person,
                        actions.get(person.track_id),
                        contacts.get(person.track_id),
                    )
                    for person in persons
                ]
                for emitter in emitters:
                    emitter.emit_many(records)
                last_emit = now

            if show_preview:
                annotated = frame
                if pose_batch.result is not None:
                    annotated = pose_batch.result.plot()
                _draw_overlay(annotated, persons, actions, contacts, det_batch.objects)
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
) -> dict:
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
    }


def _draw_overlay(
    frame,
    persons: list[TrackedPerson],
    actions: dict[int, ActionResult],
    contacts: dict[int, ContactResult],
    objects,
) -> None:
    for obj_idx, obj in enumerate(objects):
        x1, y1, x2, y2 = [int(v) for v in obj.bbox]
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
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    height, width = frame.shape[:2]
    x = max(0, min(x, width - text_w - 4))
    y = max(text_h + 4, min(y, height - 4))
    top_left = (x, y - text_h - 4)
    bottom_right = (x + text_w + 4, y + baseline + 2)
    cv2.rectangle(frame, top_left, bottom_right, bg_color, thickness=-1)
    cv2.putText(frame, text, (x + 2, y), font, font_scale, text_color, thickness)


if __name__ == "__main__":
    raise SystemExit(main())
