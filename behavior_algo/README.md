# DMS · 行为识别算法 A

> 驾驶员危险行为实时识别 · YOLOv8 + Pose + 规则融合 + 时序平滑

## 🎯 模块职责

覆盖 DMS 项目的"行为动作"监测维度：

| 行为 | type | 预警 |
|------|------|------|
| 驾驶中使用手机 | `phone_use` | high |
| 驾驶中打电话 | `calling` | high |
| 驾驶中吸烟 | `smoking` | medium |
| 未系安全带 | `no_seatbelt` | high |
| 双手离开方向盘 | `hands_off_wheel` | high |
| 驾驶姿势异常 | `abnormal_posture` | low |
| 驾驶位无人 | `no_driver` | critical |
| 摄像头被遮挡 | `lens_covered` | medium |

## 📁 目录结构

```
behavior_algo_a/
├── behavior_detector.py      # 核心推理模块（对外接口）
├── gradio_demo.py            # Web Demo
├── README.md                 # 本文件
├── docs/
│   ├── 01_requirements.md    # 问题整理
│   └── 02_algorithm_design.md # 算法设计与技术选型
├── models/
│   ├── yolov8n.pt            # 通用检测（person / cell phone）
│   ├── yolov8n-pose.pt       # 17 关键点
│   ├── seatbelt.pt           # (可选) 安全带
│   └── smoking.pt            # (可选) 吸烟
├── scripts/
│   ├── download_weights.py   # 一键下载官方权重
│   └── train.py              # 自定义数据集训练
├── data/
│   ├── seatbelt_example.yaml # YOLO 数据集模板
│   └── smoking_example.yaml
├── samples/                  # 测试样例（放图片或视频）
└── outputs/                  # 推理结果输出
```

## 🚀 快速开始

### 1. 依赖

```bash
pip install ultralytics opencv-python torch torchvision gradio numpy pillow
```

### 🔥 最佳路径：训练 8 类统一模型（最大 FPS + 最佳精度）

这是**推荐**的工作流 — 一个 yolov8n 同时检出 8 类关键行为，单次推理代替多个分支，CPU 上 ~35ms / 帧：

```bash
# 1) 从 Roboflow 下载 3 个公开数据集 (约 500MB, 含 9000+ 张标注图)
export ROBOFLOW_API_KEY="<你的 key>"
python scripts/download_datasets.py  # 见下方 "数据集说明"

# 2) 合并为统一 8 类数据集 (几十秒)
python scripts/merge_datasets.py

# 3) 训练 (GPU 推荐; CPU 会很慢)
python scripts/train_unified.py --epochs 40 --device 0
# 或用 scripts/train_on_colab.ipynb 在免费 Colab T4 上跑 25-40 分钟

# 4) 运行实时客户端 (自动加载 models/unified.pt)
python live_client.py --style monitor
```

**统一模型 8 类别（由 Roboflow 3 个公开数据集合并而来）：**
| 类 | 含义 | 行为映射 |
|----|------|---------|
| hand_on_wheel | 手在方向盘（安全基线）| 抑制 hands_off_wheel 告警 |
| phone_use | 发短信 | phone_use |
| calling | 打电话 | calling |
| drinking | 喝水 | abnormal_posture |
| reach_behind | 伸手取物 | abnormal_posture |
| cigarette | 香烟 | smoking |
| no_seatbelt | 未系安全带 | no_seatbelt |
| seatbelt | 已系（基线）| 抑制 no_seatbelt 告警 |

### 2. 下载权重（仍支持独立分支的基础模型）

```bash
python scripts/download_weights.py
```

自动下载：
- `yolov8n.pt`（COCO，含 person + cell phone，6.2 MB）
- `yolov8n-pose.pt`（COCO-Pose 17 关键点，6.5 MB）
- `yolov8s.pt`（精度升级版，22 MB；手机识别不好时替换 yolov8n.pt）

**训练专用DMS模型（推荐长期用）** — **详细步骤见 [data/SEATBELT_DATASETS.md](data/SEATBELT_DATASETS.md)**

简版：20–40 分钟搞定（GPU），包含：
- 3 个推荐 Roboflow 数据集（3489 / 中等 / 2tech 三档）
- Roboflow API 一键路径 + 手动下载 zip 路径（两种都支持）
- 文件放置位置、`data.yaml` 改法、训练命令
- 类别名识别规则（包含 `no`/`without`/`unbelted` 自动判为未系）

```bash
pip install roboflow
python scripts/train_seatbelt.py --source roboflow \
    --rf-api-key <你的Key> \
    --rf-workspace seatbelttraining-7yh0f \
    --rf-project seatbelt-detection-lb1ec \
    --rf-version 3 --epochs 40 --device 0
# 训练完 best.pt 自动拷贝到 models/seatbelt.pt
```

### 3. 本地实时客户端（最低延迟，最高 FPS）

```bash
# 演示答辩用（监控样式可视化）
python live_client.py --style monitor

# 调试用（显示方向盘 ROI、行为列表、FPS）
python live_client.py --style debug        # 默认

# 手机识别不灵敏时，降低阈值 + 开高精度重检（已默认）
python live_client.py --phone-conf 0.15 --phone-recheck-imgsz 640

# 暗光车载场景可保持 CLAHE 增强（已默认）
# 若嫌慢可关 --no-low-light-enhance
```

快捷键：

| 键 | 作用 |
|---|---|
| `q` / `ESC` | 退出 |
| `s` | 截图保存到 `outputs/` |
| `r` | 重置时序滑窗 |
| `p` | 暂停/恢复推理 |
| `+` / `-` | 动态调整跳帧步长 |

调优参数（按实时性从低到高）：

```bash
# 默认（imgsz=384，每帧推理）— 速度精度平衡
python live_client.py

# 激进提速（小分辨率 + 跳帧）
python live_client.py --imgsz 256 --infer-every 2

# 有 GPU
python live_client.py --device 0 --imgsz 640

# 视频文件
python live_client.py --source samples/test.mp4 --save-video outputs/out.mp4

# 逐帧 JSON 日志
python live_client.py --log-json outputs/alerts.jsonl
```

#### 📷 摄像头权限（Windows）

首次运行若报错，按以下顺序检查：

1. **关闭占用摄像头的程序**：Teams / Zoom / 浏览器 / 旧的 Python 进程
2. **开启系统权限**：
   设置 → 隐私和安全性 → 相机
   - "允许访问此设备上的相机" = 开
   - "允许桌面应用访问你的相机" = 开
3. **重启摄像头驱动**：设备管理器 → 照相机 → 右键更新
4. 脚本会自动尝试 `DSHOW → MSMF → ANY` 三个后端和索引 `0/1/2`，全失败会打印详细说明

### 4. 作为库调用（给 后续系统集成）

```python
import time, cv2
from behavior_detector import BehaviorDetector

det = BehaviorDetector(
    yolo_weights="models/yolov8n.pt",
    pose_weights="models/yolov8n-pose.pt",
    seatbelt_weights="models/seatbelt.pt",   # 可选
    smoking_weights="models/smoking.pt",     # 可选
    device="cpu",
)

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok: break
    result = det.predict(frame, frame_id=0, timestamp=time.time())
    # result 是 JSON 字典，见 “输出规范”
    vis = det.visualize(frame, result)
    cv2.imshow("dms", vis); cv2.waitKey(1)
```

## 📥 输入规范

| 字段 | 类型 | 说明 |
|------|------|------|
| `frame` | `np.ndarray` | shape `(H,W,3)`, dtype `uint8`, **BGR** 通道序（OpenCV 默认） |
| `frame_id` | `int` | 单调递增帧号 |
| `timestamp` | `float` | Unix 时间戳（秒） |

- 建议分辨率 ≥ 640×480
- 建议帧率 15~30 FPS（CPU 上 YOLOv8n 约 15~25 FPS）

## 📤 输出规范

统一 JSON 结构，便于 5 号系统融合告警：

```json
{
  "frame_id": 1024,
  "timestamp": 1713772800.123,
  "latency_ms": 38.5,
  "behaviors": [
    {
      "type": "phone_use",
      "label_zh": "驾驶中使用手机",
      "confidence": 0.87,
      "bbox": [412, 160, 495, 240],
      "severity": "high",
      "duration_s": 2.4,
      "evidence": "画面内检出手机"
    }
  ],
  "alert_level": "high",
  "recommendation": "语音警告 + 仪表盘闪烁",
  "driver_present": true,
  "camera_ok": true
}
```

- `severity`: `low` / `medium` / `high` / `critical`
- `alert_level`: 当前帧最高严重度（`none` 表示正常）
- `recommendation`: 给出的动作建议字符串
- `bbox`: 可为空（姿态类判定无 bbox）

## 🧠 算法原理（简版）

```
frame ──► [遮挡检测]  ──► YOLOv8n ──► person / cell_phone
          │                │
          │                ├──► YOLOv8n-pose ──► 17 keypoints
          │                │                    │
          │                │                    ▼
          │                └──► 规则层: wrist↔phone / wrist↔wheel ROI
          │                                      │
          └──► seatbelt / smoking 专用 YOLO
                                                 │
                                                 ▼
                                            时序滑窗(5 帧)
                                                 │
                                                 ▼
                                              JSON
```

- **手机使用 vs 打电话**：通过手机中心 ↔ 耳/鼻距离相对面部宽度判定
- **双手离盘**：方向盘 ROI 默认为画面下半部中 70%，腕部关键点不在 ROI 内
- **姿势异常**：`atan2(|ΔyShoulder|, |ΔxShoulder|) > 30°`
- **时序去抖**：滑窗 5 帧 3 帧激活、3 帧失活，防止误报

详见 `docs/02_algorithm_design.md`。

## 🧪 训练自定义模型

```bash
# 1. 准备数据集（YOLO 格式），见 data/seatbelt_example.yaml
# 2. 训练
python scripts/train.py --data data/seatbelt_example.yaml \
                        --base yolov8n.pt --epochs 60 --imgsz 640

# 3. 训完把 best.pt 拷到 models/seatbelt.pt 即可被加载
cp runs/behavior/exp/weights/best.pt models/seatbelt.pt
```

### 推荐公开数据集
- State Farm Distracted Driver Detection (Kaggle) — 10 类分心姿态
- AUC Distracted Driver Dataset V2
- Roboflow Universe 上搜索 `seatbelt detection` / `cigarette detection`
- DMD (Driver Monitoring Dataset)

## ⚡ 性能指标（本机实测，CPU）

| 配置 | 中位延迟 | FPS |
|------|---------|-----|
| YOLOv8n + pose，imgsz=384（无 smoking） | **36 ms** | **28** |
| 上 + smoking (YOLOv11-M) 全图 | ~200 ms | 5 |
| 上 + smoking 仅驾驶员 crop | ~160 ms | 6 |
| +  `--infer-every 2`（等同跳帧） | ~80 ms* | ~12* |
| +  `--infer-every 3` | ~55 ms* | ~18* |

\* 显示端均摊 FPS；smoking 仅在推理帧跑。

**关于 YOLOv11-M smoking 模型的延迟：** [Enos-123/smoking-detection](https://huggingface.co/Enos-123/smoking-detection) 是 medium 尺寸（40 MB），CPU 上约 100–150 ms/次。三个可选提速方式：
1. `python live_client.py --infer-every 2` — 立即见效
2. `--device 0` 用 GPU — 延迟降至 ~10 ms
3. 自己训练 yolov8n 架构的香烟模型（~10× 快），见 `scripts/train.py`

Gradio 流 Tab 额外有 ~80–120 ms 的浏览器 JPEG + HTTP 开销，实感 FPS 约本地客户端的 1/3。

## 📘 接口约定（与其他分工模块）

| 交互方 | 方向 | 内容 |
|--------|------|------|
| 采集 → 本模块 | 输入 | BGR 视频帧 |
| 本模块 → 5 号 融合 | 输出 | JSON 告警（上文结构） |
| 本模块 ↔ 1 号 疲劳 | 共享 | `driver_present` 字段 |
| 本模块 ↔ 2 号 视线 | 共享 | 驾驶员 bbox |

## 🧩 TODO / 扩展

- [ ] 加入 3D CNN 或 SlowFast 做长时序动作识别（吃东西、转身等）
- [ ] 集成 MediaPipe Face 以融合面部动作（点火、接近手机的细粒度表情）
- [ ] ONNX 量化部署（车载 ARM CPU）
