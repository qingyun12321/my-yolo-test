# 分割训练流程（与 Ultralytics 官方一致）

本流程按官方推荐顺序组织：
1. COCO JSON 标注转换为 YOLO 格式（JSON2YOLO/Ultralytics converter）；
2. 配置数据集 YAML（`data=...`）；
3. 执行 `segment train`；
4. 通过 `cfg` 和 `key=value` 覆盖训练参数；
5. 推理时复用同一份 `names` 作为白名单来源。

## 官方参考

- Segmentation 训练（custom dataset）  
  https://docs.ultralytics.com/tasks/segment/#how-do-i-train-a-yolo26-segmentation-model-on-a-custom-dataset
- JSON2YOLO（官方转换器来源）  
  https://github.com/ultralytics/JSON2YOLO
- 配置与覆盖机制（cfg/default.yaml + arg=value）  
  https://docs.ultralytics.com/usage/cfg/

## 1) 标注转换（COCO JSON -> YOLO Seg）

官方现在推荐直接用 Ultralytics 内置转换器（原 JSON2YOLO 已并入）。

### 使用本项目脚本（封装官方 `convert_coco(use_segments=True)`）

```bash
uv run python tools/convert_coco_to_yolo_seg.py \
  --labels-dir datasets/coco-json/annotations \
  --save-dir datasets/yolo-seg \
  --cls91to80
```

### 与官方 Python 示例等价

```python
from ultralytics.data.converter import convert_coco
convert_coco(labels_dir="path/to/annotations", use_segments=True)
```

## 2) 准备 data.yaml

编辑 `training/configs/segment_data.yaml`：

- `path`: 数据集根目录
- `train` / `val` / `test`: 图像子目录
- `names`: 类别定义（推理白名单将复用这里）

可选：训练覆盖模板在 `training/configs/custom_train.yaml`。

## 3) 训练（官方命令 + 本项目等价脚本）

### 官方 CLI 形式

```bash
yolo segment train data=training/configs/segment_data.yaml model=models/yolo26n-seg.pt epochs=100 imgsz=640
```

### 本项目脚本（参数语义与官方一致）

```bash
uv run python tools/train_segment_objects.py \
  --data training/configs/segment_data.yaml \
  --model models/yolo26n-seg.pt \
  --epochs 100 \
  --imgsz 640 \
  --out models/object-seg-finetuned.pt
```

脚本会打印一条 `Official-equivalent CLI preview`，用于核对与官方命令完全对齐。

## 4) 使用 cfg 与覆盖参数（官方 cfg 机制）

官方机制是“先加载默认/自定义 cfg，再由 `arg=value` 覆盖”。  
本项目脚本对应参数：

- `--cfg path/to/custom_cfg.yaml` 对应 `cfg=...`
- `--override KEY=VALUE` 对应官方 `arg=value`（可重复）

示例：

```bash
uv run python tools/train_segment_objects.py \
  --data training/configs/segment_data.yaml \
  --model models/yolo26n-seg.pt \
  --cfg training/configs/custom_train.yaml \
  --override batch=8 \
  --override lr0=0.005 \
  --override cos_lr=True \
  --override mosaic=0.2
```

## 5) 推理白名单联动（训练/推理同源）

训练完成后，推理阶段直接读取同一份 `data.yaml` 的 `names`：

```bash
uv run python -m src.app \
  --det-model models/object-seg-finetuned.pt \
  --det-whitelist-config training/configs/segment_data.yaml \
  --det-whitelist-mode override
```

这样可以保证“训练类别定义”和“线上识别白名单”始终一致。
