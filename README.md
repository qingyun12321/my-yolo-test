# my-yolo-test

动作识别 + 手与物体接触检测示例工程。  
当前版本重点支持：

- 手关键点稳定与 ROI 增强检测；
- 目标去重、跨类别冲突抑制；
- 目标时序稳定（短时遮挡保持 + 类别抖动抑制）；
- 从训练 YAML 自动导入白名单类别。

## 快速运行

```bash
uv run python -m src.app \
  --pose-model models/yolo26n-pose.pt \
  --det-model models/yolo26n-seg.pt \
  --det-whitelist-config training/configs/segment_data.yaml \
  --det-whitelist-mode override \
  --debug
```

## 关键参数

- `--debug`：统一打开调试信息（ROI、计数面板等）。
- `--det-whitelist-config`：从训练配置 YAML 加载白名单类别。
- `--obj-temporal`：开启目标时序稳定（建议保持开启）。
- `--hand-roi-*`：控制手部 ROI 扩展与稳定策略。

## 微调训练流程

完整流程见 `training/README.md`（已对齐 Ultralytics 官方 `segment train` + JSON2YOLO 转换流程）。
