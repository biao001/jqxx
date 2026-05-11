# 二、算法设计与技术选型

## 2.1 总体架构：多分支融合

```
┌─────────────┐
│  输入帧      │ frame (BGR, H×W×3)
└──────┬──────┘
       │
       ├──► [预处理层] 遮挡检测 (灰度方差 < θ → lens_covered)
       │
       ▼
┌─────────────────────────────────────────────────┐
│                 感知主干 (并行)                   │
├──────────────┬──────────────┬──────────────────┤
│ ① YOLOv8n    │ ② YOLOv8n-   │ ③ 专用 YOLOv8     │
│ COCO 预训练  │    pose       │ (seatbelt/smoke) │
│ → person,    │ → 17 keypts   │ → belt/no_belt, │
│   cell phone │  (骨架点)     │   cigarette      │
└──────┬───────┴──────┬───────┴────────┬─────────┘
       │              │                │
       ▼              ▼                ▼
┌─────────────────────────────────────────────────┐
│          ④ 规则 / 几何判定层                     │
│  · 手机靠头 (wrist↔nose 距离 < τ) → calling     │
│  · 手机在手 (phone∩wrist bbox) → phone_use      │
│  · 双手远离画面中下 → hands_off_wheel           │
│  · 肩-腰倾角 > 30° → abnormal_posture           │
│  · 无 person 检出 > 3s → no_driver              │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│          ⑤ 时序滑窗 (5 帧多数投票)              │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│ JSON 输出    │
└─────────────┘
```

## 2.2 子模块技术选型

### 2.2.1 目标检测主干：YOLOv8n（COCO 80 类预训练）

- **为什么 YOLO？** PDF 第 25 页"环境配置"已明确 conda + YOLOv8 方案；实时性 + 精度兼具。
- **为什么 n 版本？** nano 参数 3.2M，CPU 可跑；车载端侧部署友好。
- **公开权重**：Ultralytics 官方 `yolov8n.pt`，COCO 含类别：
  - `0: person`（驾驶员）
  - `67: cell phone`（手机，覆盖 B1/B2 通用情况）
- **推理接口**：`model(frame, conf=0.35, iou=0.5, verbose=False)`

### 2.2.2 姿态估计：YOLOv8n-pose

- **权重**：`yolov8n-pose.pt`，COCO-Pose 17 关键点
- **用途**：取 `nose(0)`、`left_wrist(9)`、`right_wrist(10)`、`left_shoulder(5)`、`right_shoulder(6)` 等
- **判定**（几何规则）：
  ```
  # 打电话判定
  calling = exists(cell_phone) && min(dist(phone.ctr, nose), 
                                       dist(phone.ctr, left_ear),
                                       dist(phone.ctr, right_ear)) < 0.15 * face_width
  
  # 双手离盘判定（方向盘假定在画面下 1/3 中心区）
  wheel_roi = [W*0.2, H*0.55, W*0.8, H*1.0]
  hands_off_wheel = !inside(lw, wheel_roi) && !inside(rw, wheel_roi)
  
  # 姿势异常
  shoulder_tilt = atan2(|lsh.y - rsh.y|, |lsh.x - rsh.x|)
  abnormal_posture = shoulder_tilt > 30°  OR  body_lean_x > 0.3*W
  ```

### 2.2.3 安全带与香烟：专用小型检测器

这两类在 COCO 中没有，方案：

**方案 A（推荐，零训练起步）**：
- 使用 HuggingFace / Roboflow 上的开源 YOLOv8 权重：
  - Seatbelt: [keremberke/yolov8n-seatbelt-detection](https://huggingface.co/keremberke/yolov8n-seatbelt-detection) 或 Roboflow Universe "seatbelt detection" 公开数据集导出
  - Smoking/Cigarette: Roboflow "cigarette-detection-aycnc" 等公开权重
- 若网络不可达：退化为"启发式"——香烟用手机检测替代（手部持物），安全带用 HSV 条带检测替代

**方案 B（若需自己训练）**：
- 数据集：
  - 安全带：[Seat Belt Detection Dataset](https://www.kaggle.com/datasets/) 或 SFD (State Farm Distracted Driver)
  - 抽烟：Roboflow "Cigarette Detection Computer Vision Project"
- 训练命令：`yolo detect train data=seatbelt.yaml model=yolov8n.pt epochs=50 imgsz=640`
- 见 `scripts/train.py`

### 2.2.4 遮挡 / 低光检测

- 简单而鲁棒：`laplacian_var < 10` → 镜头遮挡；`mean < 20` → 严重低光
- 无需模型

### 2.2.5 时序滑窗与告警去抖

- 滑窗长度 `K=5` 帧
- 规则：窗内同一行为出现 ≥3 次才触发；消失 ≥3 次才清除
- 数据结构：`collections.deque(maxlen=5)` per behavior
- 目的：抑制 YOLO 单帧闪动造成的告警震荡

## 2.3 数据集与训练策略

### 2.3.1 数据集来源（参照 PDF 第 20-23 页原则）

| 数据集 | 用途 | 说明 |
|--------|------|------|
| **State Farm Distracted Driver Detection** (Kaggle) | B1, B3, B6 | 10 类驾驶员姿态，含打电话/发短信/伸手拿东西/化妆/喝水 |
| **AUC Distracted Driver Dataset V2** | B1, B3 | 10 类，带姿态细分 |
| **Roboflow "seatbelt-detection"** | B4 | YOLO 格式，直接可训 |
| **Roboflow "cigarette-detection"** | B3 | YOLO 格式 |
| **DMD (Driver Monitoring Dataset)** | 多项 | 开源多模态，含 RGB+红外 |
| **合成数据 (Unity/Unreal)** | 长尾补充 | 按 PDF p23 建议 |

### 2.3.2 训练配置

```yaml
model: yolov8n.pt
imgsz: 640
epochs: 80
batch: 16
optimizer: AdamW
lr0: 0.001
augment:
  hsv_h: 0.015     # 模拟光照
  hsv_s: 0.7
  hsv_v: 0.4
  translate: 0.1
  scale: 0.5
  fliplr: 0.5
  mosaic: 1.0
```

### 2.3.3 评估指标

- mAP@0.5 (主指标)
- Precision / Recall
- 混淆矩阵（关注 phone_use ↔ calling 的区分）
- 实时 FPS

## 2.4 为什么这样选？（Trade-off）

| 备选方案 | 优点 | 缺点 | 决策 |
|---------|------|------|------|
| 纯分类 CNN（ResNet 10 类） | 训练简单 | 无 bbox，多行为并发难处理 | ✗ |
| Transformer (ViT / DETR) | 精度高 | 实时性差，部署复杂 | ✗ 不适合车载端侧 |
| YOLOv8 + 规则融合 | 实时、公开权重多、易扩展 | 需要手工规则 | ✓ |
| 3D CNN / SlowFast (视频) | 可识别时序动作 | 延迟大，算力高 | △ 二期可叠加 |
| MediaPipe Pose | 免训练 | 手机/香烟需另建 | △ 备选姿态 |

## 2.5 模块接口示例（给 5 号系统集成）

```python
from behavior_detector import BehaviorDetector

detector = BehaviorDetector(
    yolo_weights="models/yolov8n.pt",
    pose_weights="models/yolov8n-pose.pt",
    seatbelt_weights="models/seatbelt.pt",    # 可选
    smoking_weights="models/smoking.pt",      # 可选
    temporal_window=5,
    device="cpu",
)

# 单帧调用
result = detector.predict(frame, frame_id=0, timestamp=time.time())
# result 即 1.5 节 JSON

# 视频流
for frame in camera_stream():
    result = detector.predict(frame, ...)
    publish(result)
```
