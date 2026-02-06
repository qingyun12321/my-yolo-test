# tools 目录说明

`tools/` 按用途分为三层：训练、数据处理、维护。

```text
tools/
├─ maintenance/
│  ├─ clean_logs.py
│  └─ clean_runs.py
├─ datasets/
│  ├─ hand/
│  │  └─ download_hand_keypoints.py
│  └─ segment/
│     └─ convert_coco_to_yolo_seg.py
└─ train/
   ├─ hand/
   │  └─ train_hand_keypoints.py
   └─ segment/
      └─ train_segment_objects.py
```

执行原则：

- 统一使用 `uv run python ...`；
- 所有脚本都支持自动定位仓库根目录（可从任意工作目录调用）。

常用命令：

```bash
uv run python tools/train/segment/train_segment_objects.py --help
uv run python tools/train/hand/train_hand_keypoints.py --help
uv run python tools/datasets/segment/convert_coco_to_yolo_seg.py --help
uv run python tools/maintenance/clean_logs.py --help
```
