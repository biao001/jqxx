# 一、问题整理 (Requirements Analysis)

## 1.1 模块在 DMS 系统中的定位

- **生理状态**：疲劳、表情、视线（由 1 号"疲劳检测"、2 号"视觉注意力"负责）
- **行为动作**：本模块负责，聚焦"驾驶员手/身体相关的危险动作"

整合国际权威 DMS 规范（参考欧盟 GSR 2026/7 条款、中国 GB《智能网联汽车组合驾驶辅助系统安全要求》2027/1/1 实施标准，以及 PDF 第 26 页列出的 Smoking / Using Phone / No Driver / Lens Covered / Calling 指标），行为识别算法 A 的职责如下。

## 1.2 功能需求（Functional Requirements）

| 编号 | 行为类别 | 中文语义 | 预警等级 | 参考依据 |
|------|---------|---------|---------|----------|
| B1 | `phone_use` | 驾驶中使用手机（手持 / 低头看屏） | 高 | PDF p9, p22；欧盟 GSR |
| B2 | `calling` | 驾驶中打电话（手机贴耳） | 高 | PDF p26 |
| B3 | `smoking` | 驾驶中吸烟（手持香烟 / 低头点火） | 中 | PDF p9, p22 |
| B4 | `no_seatbelt` | 未系安全带 | 高 | PDF p9, p22；GB 强制标准 |
| B5 | `hands_off_wheel` | 双手长时间离开方向盘 | 高 | PDF p9, p22；L2+ 脱管风险 |
| B6 | `abnormal_posture` | 不规范驾驶姿势（身体过度倾斜） | 低 | PDF p9 |
| B7 | `no_driver` | 驾驶位无人 | 极高 | PDF p26 |
| B8 | `lens_covered` | 摄像头被遮挡 | 中 | PDF p26 |

## 1.3 非功能需求（Non-Functional Requirements）

| 指标 | 目标值 | 备注 |
|------|-------|------|
| 实时性 | ≥ 15 FPS @ 640×480 | CPU 可跑；GPU 下 ≥30 FPS |
| 误报率 FPR | ≤ 5% | 5 帧时序滑窗 + 多数投票抑制 |
| 漏报率 FNR | ≤ 10% | 针对 B1~B4 四个强制项 |
| 鲁棒性 | 白天/夜间/逆光/戴墨镜 | 数据增强 + 红外摄像头建议 |
| 隐私 | 本地推理，不上传原图 | 符合 PDF p7 隐私保护趋势 |

## 1.4 输入规范（Input Spec）

```python
# 数据结构
Input = {
    "frame":    np.ndarray,      # shape=(H,W,3), dtype=uint8, color=BGR
    "frame_id": int,             # 单调递增帧号
    "timestamp": float,          # Unix 时间戳（秒）
    "camera_id": str,            # 可选，多摄像头场景
}

# 物理规格
- 图像分辨率: 建议 ≥ 640×480，最小 320×240
- 色彩空间:   BGR (OpenCV 默认) 或 Gray (红外摄像头)
- 帧率:       15~30 FPS
- 来源:       USB 摄像头 / 车载 CAM / RTSP 流 / 本地视频文件 / 单帧图像
```

## 1.5 输出规范（Output Spec）

统一 JSON 结构，方便 5 号（系统集成）接入：

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
      "bbox": [x1, y1, x2, y2],
      "severity": "high",
      "duration_s": 2.4,
      "evidence": "检测到手机 + 手部靠近头部"
    }
  ],
  "alert_level": "high",
  "recommendation": "语音警告 + 方向盘震动",
  "driver_present": true,
  "camera_ok": true
}
```

`alert_level` 映射规则：`no_driver > phone_use/calling/no_seatbelt/hands_off_wheel > smoking/lens_covered > abnormal_posture`。

## 1.6 接口约定（与其他模块）

- **上游（数据采集 / 6 号）**：通过 `BehaviorDetector.predict(frame)` 调用；或订阅 ZMQ/ROS topic `/dms/frame`。
- **下游（决策融合 / 5 号）**：输出 JSON 到 `/dms/behavior_alert` topic，或作为函数返回值。
- **与 1 号疲劳模块**：共享 `driver_present` 字段，避免重复检测。
- **与 2 号视线模块**：共享 `bbox` 驾驶员区域，减少 ROI 提取开销。

## 1.7 交付清单

- [x] 需求分析（本文档）
- [x] 算法设计文档
- [x] `behavior_detector.py` 核心推理模块
- [x] `gradio_demo.py` 可视化 Demo
- [x] `train.py` 训练脚本（自定义数据集用）
- [x] 预训练权重下载脚本
- [x] README + 运行说明
