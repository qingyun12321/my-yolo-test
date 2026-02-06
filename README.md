# my-yolo-test

动作识别 + 手物接触检测项目。  
核心目标是在真实场景下保持高鲁棒性：遮挡、抖动、重复检测、类别冲突都尽量稳。

## 1. 项目特点

- 基于 YOLO pose + YOLO seg + MediaPipe 的多模块融合；
- 引入手部 ROI 二次检测，提升手持小目标识别；
- ROI 检测路径采用批量推理，降低多 ROI 时的重复调用开销；
- 引入手关键点稳定、目标去重、跨类冲突抑制、目标时序稳定；
- 白名单可从训练配置 YAML 自动导入，训练/推理类别定义同源。

## 2. 依赖与运行方式（uv）

本项目由 `uv` 管理，推荐统一使用 `uv`：

```bash
uv sync
uv run python -m src.app --help
```

## 3. 快速开始

```bash
uv run python -m src.app \
  --pose-model models/yolo26n-pose.pt \
  --det-model models/yolo26n-seg.pt \
  --det-whitelist-config training/configs/segment_data.yaml \
  --det-whitelist-mode override \
  --debug
```

说明：

- 目标可视化默认优先显示分割 `mask` 轮廓；
- 如需仅显示 `bbox` 以换取更高预览帧率，可追加 `--no-draw-mask-edges`；
- 推理参数已对齐官方常用项，可直接通过 `--pred-*` 参数调节（如 `--pred-iou`、`--pred-max-det`）。

## 4. 目录速览

```text
my-yolo-test/
├─ src/                 # 运行时核心代码
├─ tools/               # 训练/数据处理/维护脚本
├─ training/            # 训练配置与简版训练说明
├─ docs/                # 全量说明书（推荐阅读）
├─ models/              # 模型权重
├─ datasets/            # 数据集
├─ runs/                # 训练输出
└─ output/              # 运行日志输出
```

## 5. 工具脚本

工具脚本目录与使用方式见：`tools/README.md`。

## 6. 文档入口（详细说明）

- 文档导航：`docs/README.md`
- 快速开始：`docs/01_项目总览与快速开始.md`
- 文件树与用途：`docs/02_项目结构与文件树.md`
- 管线与算法细节：`docs/03_运行时管线与核心算法.md`
- 运行参数全表：`docs/04_运行参数手册_app.md`
- 脚本参数全表：`docs/05_工具脚本参数手册.md`
- 训练流程：`docs/06_训练全流程指南.md`
- 数据集格式：`docs/07_数据集格式规范.md`
- 调参与排障：`docs/08_调参与排障手册.md`
