# 运行参数手册（`src.app`）

本手册对应命令：

```bash
uv run python -m src.app --help
```

说明规则：

- “允许值”写的是**语法层**可接受值；
- “建议范围”写的是**业务层**更稳妥区间；
- 若代码内部有二次裁剪（clamp），会在“影响/备注”列注明。

---

## 1. 基础模型与输入参数

| 参数 | 默认值 | 允许值/范围 | 影响与备注 |
|---|---:|---|---|
| `--pose-model` | `models/yolo26n-pose.pt` | 有效模型路径或模型名 | 姿态模型来源，影响动作识别与人体框质量。 |
| `--det-model` | `models/yolo26n-seg.pt` | 有效模型路径或模型名 | 目标检测/分割模型来源，影响物体识别与接触判定。 |
| `--hand-model` | `models/hand_landmarker.task` | 有效路径 | MediaPipe 手关键点模型路径；不存在会自动下载。 |
| `--source` | `None` | 摄像头索引（如 `0`）或设备路径 | 视频输入源；为空时使用 `default_source()`。 |
| `--conf` | `0.2` | 浮点数，建议 `[0.05, 0.7]` | 姿态与检测共用置信度阈值；过低增误检，过高增漏检。 |
| `--imgsz` | `640` | 正整数，建议 `320~1280` | 推理输入尺寸，越大越准但更慢。 |
| `--device` | `None` | `cpu` / `0` / `0,1` 等 | 推理设备；`None` 交由底层自动选择。 |
| `--pred-iou` | `0.7` | 浮点；内部 clamp `[0,1]` | Ultralytics `predict` 的 NMS IoU 阈值。 |
| `--pred-max-det` | `300` | 整数；内部下限 `1` | 每帧最多保留检测数；减小可提升速度并抑制长尾误检。 |
| `--pred-agnostic-nms` / `--no-pred-agnostic-nms` | `False` | 布尔开关 | 类别无关 NMS；多类别重叠严重时可尝试开启。 |
| `--pred-half` / `--no-pred-half` | `False` | 布尔开关 | 半精度推理；通常仅在 CUDA 设备上有收益。 |
| `--pred-retina-masks` / `--no-pred-retina-masks` | `False` | 布尔开关 | 更高分辨率 mask，边界更细但更慢。 |
| `--pred-batch` | `1` | 整数；内部下限 `1` | `predict` 批处理大小；对多 ROI 批量推理有效。 |

---

## 2. 手检测与手稳定参数

| 参数 | 默认值 | 允许值/范围 | 影响与备注 |
|---|---:|---|---|
| `--hand-num` | `2` | 整数，建议 `1~4` | 手检测上限；真实单人场景通常 `2` 足够。 |
| `--hand-det-conf` | `0.5` | `[0,1]` | 手检测置信度阈值。 |
| `--hand-presence-conf` | `0.5` | `[0,1]` | 手存在置信度阈值。 |
| `--hand-track-conf` | `0.5` | `[0,1]` | 手追踪置信度阈值。 |
| `--hand-stabilize` / `--no-hand-stabilize` | `True` | 布尔开关 | 是否启用手关键点时序稳定器。建议开启。 |
| `--hand-smooth-alpha` | `0.55` | 语法浮点；内部 clamp 到 `[0,1]` | EMA 基础平滑系数；越小越稳但更迟滞。 |
| `--hand-smooth-adaptive` / `--no-hand-smooth-adaptive` | `True` | 布尔开关 | 是否随手速自适应 alpha。 |
| `--hand-smooth-fast-alpha` | `0.88` | 语法浮点；内部 clamp 到 `[0,1]` | 快速运动上限 alpha；越大越跟手。 |
| `--hand-smooth-motion-scale` | `0.12` | `>0`（内部下限 `1e-6`） | 运动敏感度；越小越容易触发“快速模式”。 |
| `--hand-hold-frames` | `2` | 整数 `>=0` | 手短时丢失保持帧数；越大越稳但可能残影。 |
| `--hand-side-merge-ratio` | `0.45` | 浮点，内部 `>=0` | 左右手去重阈值（中心距离比例）；越大越容易合并为单手。 |

---

## 3. 帧率、输出节流、动作与接触参数

| 参数 | 默认值 | 允许值/范围 | 影响与备注 |
|---|---:|---|---|
| `--fps` | `60.0` | 浮点，建议 `>=1` | 请求摄像头帧率；设备可能不完全遵守。 |
| `--interval` | `0.5` | 浮点；`<=0` 表示每帧输出 | 输出节流间隔（日志/stdout），不影响推理本身。 |
| `--keypoint-conf` | `0.2` | 浮点，建议 `[0,1]` | 人体关键点最低置信度阈值，影响动作与接触回退逻辑。 |
| `--contact-expand` | `30` | 整数，建议 `0~80` | 接触判定时 mask/bbox 膨胀像素。 |
| `--contact-dist` | `20.0` | 浮点，建议 `5~60` | 点到物体最大距离阈值；越大越宽松。 |
| `--contact-min-points` | `1` | 整数 `>=1` | 接触最少命中点数；越大越严格。 |

---

## 4. 类别过滤与白名单联动参数

| 参数 | 默认值 | 允许值/范围 | 影响与备注 |
|---|---:|---|---|
| `--det-include` | `None` | 逗号分隔字符串，如 `cup,bottle` | 仅保留指定类别；空表示不过滤。 |
| `--det-exclude` | `person,hand,...` | 逗号分隔字符串 | 排除类别名单；默认排除人/手相关。 |
| `--det-whitelist-config` | `None` | YAML 路径 | 从训练配置读取白名单。 |
| `--det-whitelist-field` | `names` | 点分路径字符串，如 `data.names` | 指定白名单字段路径。 |
| `--det-whitelist-mode` | `override` | `override` / `union` | `override` 用 YAML 替换命令行；`union` 做并集。 |

---

## 5. 目标后处理与时序稳定参数

| 参数 | 默认值 | 允许值/范围 | 影响与备注 |
|---|---:|---|---|
| `--obj-dedup` / `--no-obj-dedup` | `True` | 布尔开关 | 同类去重与冲突抑制总开关。 |
| `--obj-dedup-iou` | `0.45` | 建议 `[0,1]` | 同类去重 IoU 阈值。 |
| `--obj-dedup-center-ratio` | `0.35` | 浮点，内部 `>=0` | 去重备用条件：中心距离比例阈值。 |
| `--obj-conflict-suppress` / `--no-obj-conflict-suppress` | `True` | 布尔开关 | 跨类别嵌套冲突抑制开关。 |
| `--obj-conflict-overlap` | `0.75` | 建议 `[0,1]` | 冲突判定重叠比例阈值（交集/小框面积）。 |
| `--obj-conflict-area-ratio` | `1.8` | 浮点，内部 `>=1` | 冲突判定面积比阈值。 |
| `--obj-conflict-score-gap` | `0.15` | 浮点，内部 `>=0` | 分数差超过该值时直接保留高分。 |
| `--obj-temporal` / `--no-obj-temporal` | `True` | 布尔开关 | 目标时序稳定器开关（强烈建议开启）。 |
| `--obj-temporal-hold-frames` | `3` | 整数，内部 `>=0` | 目标轨迹内部丢失保留帧数（用于重关联）；可视化输出在无检测场景最多仅保留 1 帧，避免残影。 |
| `--obj-temporal-min-hits` | `1` | 整数，内部 `>=1` | 轨迹至少命中 N 次才输出。 |
| `--obj-temporal-iou` | `0.32` | 建议 `[0,1]` | 目标跨帧匹配 IoU 下限。 |
| `--obj-temporal-center-ratio` | `0.72` | 浮点，内部 `>=0` | 跨帧匹配中心距离比例阈值。 |
| `--obj-temporal-bbox-alpha` | `0.62` | 语法浮点；内部 clamp `[0,1]` | bbox/score EMA 系数；越大越跟手。 |
| `--obj-temporal-class-decay` | `0.9` | 语法浮点；内部 clamp `[0,1]` | 类别历史投票衰减。 |
| `--obj-temporal-score-decay` | `0.9` | 语法浮点；内部 clamp `[0,1]` | miss 阶段分数衰减，防止僵尸框。 |

---

## 6. ROI 检测参数（提升小物体识别核心）

| 参数 | 默认值 | 允许值/范围 | 影响与备注 |
|---|---:|---|---|
| `--hand-roi-det` / `--no-hand-roi-det` | `True` | 布尔开关 | 是否启用 ROI 二次检测。 |
| `--hand-roi-only` | `False` | 布尔开关 | 仅用 ROI 结果，不用全图结果。 |
| `--hand-roi-padding` | `0.35` | 语法浮点；内部 clamp `[0,1]` | 手框基础 padding。 |
| `--hand-roi-min-size` | `96` | 整数；内部下限 `32` | ROI 像素最小边。 |
| `--hand-roi-min-size-ratio` | `0.12` | 语法浮点；内部 clamp `[0,1]` | ROI 最小尺寸占短边比例。 |
| `--hand-roi-max-size-ratio` | `0.42` | 语法浮点；内部 clamp `[0.1,1.0]` | ROI 最大尺寸占短边比例，用于限制 ROI 过大。 |
| `--hand-roi-context-scale` | `1.7` | 浮点；内部下限 `1.0` | 前向/横向外扩基准比例。 |
| `--hand-roi-forward-shift` | `0.36` | 浮点；内部 `>=0` | 沿手前向的额外平移。 |
| `--hand-roi-inward-scale` | `1.18` | 浮点；内部下限 `1.0` | 向内补偿尺度。 |
| `--hand-roi-inward-shift` | `0.1` | 浮点；内部 `>=0` | 逆前向方向平移补偿。 |
| `--hand-roi-direction-smooth` | `0.35` | 语法浮点；内部 clamp `[0,1]` | ROI 方向平滑，降低外扩方向抖动。 |
| `--hand-roi-merge-iou` | `0.7` | 语法浮点；内部 clamp `[0,1]` | 单侧 ROI 合并 IoU 阈值。 |
| `--hand-roi-global-merge-iou` | `0.78` | 语法浮点；内部 clamp `[0,1]` | 跨左右 ROI 合并阈值。 |
| `--hand-roi-cross-side-merge-ratio` | `0.45` | 浮点；内部 `>=0` | 左右手 ROI 种子距离判重阈值。 |
| `--hand-roi-hold-frames` | `2` | 整数；内部 `>=0` | ROI 丢点保持帧数。 |
| `--hand-roi-shrink-floor` | `0.9` | 语法浮点；内部 clamp `[0,1]` | 防止 ROI 帧间骤缩。 |
| `--hand-roi-size-smooth` | `0.35` | 语法浮点；内部 clamp `[0,1]` | ROI 中心/大小平滑系数。 |
| `--roi-object-max-area-ratio` | `0.62` | 浮点；`<=0` 关闭 | ROI 检测目标占 ROI 面积过大时过滤，聚焦小物体。 |
| `--roi-partial-suppress` / `--no-roi-partial-suppress` | `True` | 布尔开关 | 合并前抑制 ROI 局部重复（全图已检出同一大物体）。 |
| `--roi-partial-profile` | `balanced` | `loose` / `balanced` / `strict` | ROI 与全图结果合并前的“局部重复抑制档位”。`loose` 保留更多 ROI 小目标，`strict` 更积极抑制 ROI 局部重复。 |
| `--unknown-roi-fallback` / `--no-unknown-roi-fallback` | `False` | 布尔开关 | ROI 无已知类别时，是否启用轮廓回退并输出 `unknown` mask（默认关闭）。 |
| `--unknown-profile` | `balanced` | `loose` / `balanced` / `strict` | unknown 回退档位。`loose` 召回更高但误报也可能增加，`strict` 更抑制背景边缘误报。 |

### 6.1 档位参数对应阈值（代码内置）

`--roi-partial-profile` 对应阈值：

| 档位 | overlap_threshold | area_ratio_threshold | score_margin | 调参倾向 |
|---|---:|---:|---:|---|
| `loose` | `0.90` | `0.48` | `0.04` | 更保留 ROI 结果，避免误杀 ROI 小目标 |
| `balanced` | `0.84` | `0.62` | `0.08` | 默认折中 |
| `strict` | `0.78` | `0.74` | `0.14` | 更积极丢弃 ROI 局部重复，优先全图结果 |

`--unknown-profile` 对应关键阈值：

| 档位 | min_area_ratio | max_area_ratio | max_hand_dist_ratio | max_hand_overlap_ratio | min_fill_ratio | min_solidity | max_aspect_ratio | border_margin | 调参倾向 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `loose` | `0.008` | `0.68` | `0.80` | `0.82` | `0.16` | `0.45` | `4.2` | `2` | 召回优先，允许更多 unknown |
| `balanced` | `0.012` | `0.55` | `0.58` | `0.68` | `0.22` | `0.55` | `3.2` | `3` | 默认折中 |
| `strict` | `0.018` | `0.45` | `0.48` | `0.55` | `0.30` | `0.65` | `2.6` | `5` | 误报优先，严格过滤背景轮廓 |

---

## 7. 调试、预览与日志参数

| 参数 | 默认值 | 允许值/范围 | 影响与备注 |
|---|---:|---|---|
| `--debug` / `--no-debug` | `False` | 布尔开关 | 一键开启所有调试叠加（含 FPS、计数、阶段耗时）。 |
| `--draw-mask-edges` / `--no-draw-mask-edges` | `True` | 布尔开关 | 是否绘制分割 mask 轮廓；关闭后仅绘制 bbox，预览更快。 |
| `--hand-roi-debug` | `False` | 布尔开关 | 单独开启 ROI 矩形可视化。 |
| `--no-preview` | `False` | 布尔开关 | 关闭窗口显示，仅做计算/日志。 |
| `--log-path` | `None` | 文件路径 | JSONL 日志自定义路径。 |
| `--no-log` | `False` | 布尔开关 | 禁用文件日志输出（stdout 仍保留）。 |

---

## 8. 参数调优优先级建议

### 8.1 先调这 6 个

1. `--conf`
2. `--contact-dist`
3. `--obj-dedup-iou`
4. `--obj-conflict-overlap`
5. `--hand-roi-context-scale`
6. `--hand-roi-forward-shift`

### 8.2 遮挡明显时再加这 4 个

1. `--obj-temporal-hold-frames`
2. `--obj-temporal-class-decay`
3. `--contact-min-points`
4. `--hand-roi-hold-frames`

### 8.3 画面延迟明显时优先调

1. `--imgsz`（先降）
2. `--hand-smooth-alpha`（适当增大）
3. `--hand-smooth-motion-scale`（适当减小）
4. `--obj-temporal-bbox-alpha`（适当增大）
5. `--pred-max-det`（适当降低，如 `300 -> 120`）
6. `--no-draw-mask-edges`（预览卡顿时先关闭 mask 轮廓绘制）
7. `--pred-batch`（多 ROI 时可尝试 `2~4`）

---

## 9. 参数冲突与注意事项

- `--hand-roi-only` 开启后，全图物体会被忽略，适合“只关心手附近物体”的场景；
- `--det-whitelist-mode override` 会覆盖 `--det-include`；
- `--interval <= 0` 时每帧输出日志，吞吐和磁盘压力会明显上升；
- `--no-preview` 常用于服务化或离线压测；
- `--debug` 会增加绘制开销，性能评估时建议关闭。
- 默认启用 `--draw-mask-edges`（分割轮廓可视化）；若只关心速度可加 `--no-draw-mask-edges`。
