# 行为识别算法 A · 设计说明文档

> 对标 DMS 项目 3 号分工。与疲劳检测（1 号）、视线追踪（2 号）并列组成完整 DMS 系统。
> 本文档聚焦**危险驾驶行为**的端侧实时识别。

---

## 一、项目定位

**一句话定义：** 基于摄像头视频流的本地危险驾驶行为识别模块。

| 维度 | 内容 |
|------|------|
| **项目聚焦** | 仅聚焦"行为动作"识别；不涉及疲劳、视线、表情模块 |
| **项目目标** | 对手机使用 / 打电话 / 吸烟 / 未系安全带 / 双手离盘等行为实时检测，输出事件 + 风险评分 |
| **关键检测对象** | phone_use / calling / smoking / no_seatbelt / hands_off_wheel / abnormal_posture / no_driver / lens_covered（8 类） |
| **核心特征指标** | YOLOv8 bbox 置信度、手机-头部距离、手腕位置、方向盘 ROI 命中、风险分数、持续时间 |
| **部署目标** | CPU 本地实时（≥ 15 FPS），GPU 可达 60+ FPS |

---

## 二、全流程技术路线

**六步链路：**

```
① 数据输入 → ② 数据合并 → ③ 模型训练 → ④ 实时推理 → ⑤ 规则评分 → ⑥ 本地输出
```

| 阶段 | 关键内容 |
|------|---------|
| **01 数据输入** | Roboflow 3 个公开数据集（distracted_driving / cigarette / seatbelt），共 9137 张标注图 + 摄像头实时流 |
| **02 数据合并** | 标签重映射 → 8 类统一数据集 `dms_unified`，自动生成 YOLO 格式 `data.yaml` |
| **03 模型训练** | YOLOv8n 基座 + AdamW + mosaic/mixup 增强，imgsz=640, epochs=40，产出 `unified.pt` |
| **04 实时推理** | 低光 CLAHE 增强 → 人体检测 → 驾驶员 crop → unified 多类检测 + pose 辅助判定 |
| **05 规则评分** | 手机-头距/手腕位置判 calling；方向盘 ROI 判 hands_off；时序滑窗去抖；加权互补风险分 |
| **06 本地输出** | 监控风格界面（ACTIVE ALERTS 面板 + 风险仪表条）+ 结构化 JSON + 日志记录 |

---

## 三、输入 / 输出设计

### 3.1 输入

| 来源 | 规格 | 用途 |
|------|------|------|
| **实时摄像头视频流** | `np.ndarray (H,W,3) uint8 BGR`，≥640×480，15–30 FPS | 在线推理 |
| **Roboflow 公开数据集** | 9137 张标注图（YOLO 格式） | 离线训练 `unified.pt` |
| **本地采集数据** | 光照/角度/戴眼镜等场景补充 | 微调与回归测试 |

### 3.2 分层输出

| 层 | 字段 | 说明 |
|----|------|------|
| **A 原始检测** | `bbox`, `class_id`, `class_name`, `confidence` | YOLOv8 网络直出 |
| **B 行为事件** | `type`, `label_zh`, `severity`, `duration_s`, `evidence` | 规则+时序滤波后的稳定事件 |
| **C 风险量化** | `risk_score` (0–100), `risk_tier` (safe/attention/warning/danger/critical) | 加权互补合成 |
| **D 展示输出** | 监控界面可视化、RISK 仪表条、ACTIVE ALERTS 面板、JSON 日志 | 给下游系统 / 用户 |

### 3.3 统一 JSON（示例）

```json
{
  "frame_id": 1024,
  "timestamp": 1713772800.123,
  "latency_ms": 38.5,
  "behaviors": [
    {
      "type": "calling",
      "label_zh": "驾驶中打电话",
      "confidence": 0.87,
      "bbox": [412, 160, 495, 240],
      "severity": "high",
      "duration_s": 2.4,
      "evidence": "calling: d_phone=32 d_wrist=58 face_w=72"
    }
  ],
  "alert_level": "high",
  "risk_score": 68.5,
  "risk_tier": "warning",
  "recommendation": "语音警告 + 仪表盘闪烁",
  "driver_present": true,
  "camera_ok": true
}
```

---

## 四、技术架构（四层设计）

### 4.1 数据层

- 3 个 Roboflow 公开数据集（workspace/project/version）
  - `yolov8-ei4l6/distracted-driving-yolov8/v6` — 5 类行为 1028 张
  - `yolov8-jymgm/cigarette-wkkgi/v5` — 香烟 90 张
  - `seatbelttraining-7yh0f/seatbelt-detection-lb1ec/v4` — 安全带 8019 张
- **合并策略**：类标签 ID 重映射 → 8 类 `dms_unified` 数据集
- **类别表**：
  | ID | 名 | 含义 |
  |----|----|------|
  | 0 | hand_on_wheel | 手在方向盘（安全基线） |
  | 1 | phone_use | 发短信/看屏 |
  | 2 | calling | 打电话 |
  | 3 | drinking | 喝水 |
  | 4 | reach_behind | 伸手取物 |
  | 5 | cigarette | 香烟 |
  | 6 | no_seatbelt | 未系安全带 |
  | 7 | seatbelt | 已系（安全基线） |

### 4.2 感知层

| 模型 | 作用 | 参数 |
|------|------|------|
| **YOLOv8n (unified.pt)** | 8 类行为统一检测 | 本项目训练，imgsz=640, ~6 MB |
| **YOLOv8n-pose** | 17 关键点（辅助判定 calling vs phone_use） | COCO 官方预训练 |
| **低光 CLAHE** | 暗光/红外场景预处理 | 无需训练 |

### 4.3 规则 / 时序层

```
规则 1  calling 判定（互斥）:
   A. d(phone, 头锚点) < 0.9 × face_w
   B. |phone.y − nose.y| < 1.0 × face_w    # 手机在面部高度
   C. d(wrist, 头锚点) < 1.2 × face_w      # 手腕抬到头部
   A ∧ (B ∨ C) ⇒ calling；否则 phone_use

规则 2  hands_off_wheel:
   wheel_roi = [W×0.15, H×0.5, W×0.85, H×1.0]
   左右手腕均不在 wheel_roi 内 ⇒ hands_off_wheel

规则 3  时序滑窗去抖（K=5）:
   deque(maxlen=5) per behavior type
   激活条件  窗内 ≥3 帧命中 ⇒ 稳定报告 + 开始计 duration_s
   失活条件  窗内 ≥3 帧未命中 ⇒ 清除

规则 4  unified 模型抑制:
   unified 检出 hand_on_wheel ⇒ 抑制 hands_off_wheel 告警
   unified 检出 seatbelt      ⇒ 抑制 no_seatbelt 告警
```

### 4.4 决策层（加权互补风险评分）

**公式**：

```
risk_score = 100 × (1 − Π(1 − w_i × c_i × dur_factor_i))

  w_i         行为基础权重 (见下表)
  c_i         置信度 (0~1)
  dur_factor  = min(1.6, 1 + 0.06 × min(dur_s, 10))
```

**基础权重**：

| 行为 | 权重 |
|------|------|
| no_driver | 1.00 |
| calling | 0.80 |
| phone_use | 0.75 |
| no_seatbelt | 0.70 |
| hands_off_wheel | 0.65 |
| smoking | 0.45 |
| lens_covered | 0.40 |
| abnormal_posture | 0.30 |

**风险等级映射**：
| 分数 | Tier | 预警动作 |
|------|------|---------|
| [0, 10) | safe | 无告警 |
| [10, 30) | attention | 屏幕柔和提示 |
| [30, 60) | warning | 仪表盘图标提示 |
| [60, 85) | danger | 语音警告 + 仪表盘闪烁 |
| [85, 100] | critical | 立即语音警告 + 方向盘震动 + 限速 |

**时间窗口预警规则**：
- 任一 `high` 级行为连续 > 2 s → 升级为 `danger` 语音告警
- 任一 `high` 级行为连续 > 5 s → 升级为 `critical` 强制干预
- 同一帧 ≥ 3 类并发 → 即刻 `critical`

### 4.5 关键能力总览

- ✅ **本地实时推理**：CPU 15–25 FPS，GPU 60+ FPS
- ✅ **可解释风险评分**：公式透明，逐行为贡献可追溯
- ✅ **多行为并发**：支持同帧多告警，不互相掩盖
- ✅ **时序去抖**：5 帧滑窗抑制单帧误报
- ✅ **支持后续扩展**：接入 3D CNN / SlowFast 做时序动作可无缝替换感知层

---

## 五、效果呈现

**监控样式 Live 界面**包含：

| 位置 | 组件 | 内容 |
|------|------|------|
| 顶部条 | DMS MONITOR 标题 + 时间戳 + FPS/延迟 | 实时状态 |
| 左侧状态灯 | DRIVER / CAMERA / SMOKE-DET | 模块健康度 |
| 画面 | bbox + 四角强调线 + 圆角类别标签 | 检出的行为 |
| 右侧 ACTIVE ALERTS 面板 | 多行为列表（最多 5 条）| 置信度 + duration |
| 底部风险条 | RISK 0–100 仪表 + tier 文字 | 量化风险 |
| 底部主标签 | 最严重行为大字（CALLING / SMOKING / …）| 主要预警 |
| 右下 REC 指示 | 录制闪烁 | 记录状态 |

**独立 Controls 面板**（与主窗口统一风格）：
- `conf` 通用置信度阈值
- `phone_conf` 手机专用阈值
- `skip frames` 跳帧步长（1–10）
- `imgsz` 推理分辨率
- `low_light` CLAHE 开关

---

## 六、工程交付清单

| 文件 | 用途 |
|------|------|
| `behavior_detector.py` | 核心推理模块（对外接口） |
| `live_client.py` | 本地 OpenCV 客户端（实时） |
| `control_panel.py` | 自绘控制面板 |
| `gradio_demo.py` | Web Demo（答辩/演示） |
| `scripts/download_datasets.py` | Roboflow 批量下载 |
| `scripts/merge_datasets.py` | 3 数据集合并成 8 类 |
| `scripts/train_unified.py` | 一键训练 YOLOv8n |
| `scripts/train_on_colab.ipynb` | Colab GPU 训练 |
| `models/unified.pt` | 训练产物 |
| `docs/01_requirements.md` | 问题整理 |
| `docs/02_algorithm_design.md` | 旧版算法设计（独立分支） |
| `docs/03_behavior_algorithm_design.md` | 本文档（unified 方案） |

---

## 七、性能指标（CPU 实测）

| 配置 | 单帧延迟 | FPS |
|------|---------|-----|
| yolov8n unified + pose, imgsz=384 | ~35 ms | **28** |
| + 跳帧 `--infer-every 2` | ~18 ms 均摊 | **50+** |
| + GPU (--device 0), imgsz=640 | ~12 ms | **80+** |

---

## 八、与其他模块的对接约定

| 交互方 | 方向 | 字段 |
|--------|------|------|
| 6 号 采集 → 本模块 | 输入 | BGR 视频帧 `frame, frame_id, timestamp` |
| 本模块 → 5 号 融合 | 输出 | `behaviors[]` + `risk_score` + `alert_level` |
| 1 号 疲劳 ↔ 本模块 | 共享 | `driver_present` |
| 2 号 视线 ↔ 本模块 | 共享 | 驾驶员 bbox |
