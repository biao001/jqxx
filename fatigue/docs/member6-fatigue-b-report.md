# 6号交付说明：疲劳检测算法 B

## 目标

6号负责实现一套学习型疲劳检测算法 B。该算法以连续视频帧为输入，输出与 5 号完全一致的四类结果：

- `Normal`
- `Yawning`
- `Looking Around`
- `Fatigued Driving`

## 方法概述

本方案不是从零设计全新网络，而是使用可落地的轻量时序模型：

1. 从驾驶视频中提取人脸相关特征
2. 将连续窗口特征送入 `LSTM/GRU`
3. 输出 3 个分数：
   - `yawn_score`
   - `look_away_score`
   - `fatigue_score`
4. 再映射成统一四类输出

## 输入

每个时间窗输入特征固定为：

- `ear`
- `mar`
- `head_yaw`
- `head_pitch`
- `head_roll`
- `drowsy_prob`

默认窗口长度：

- `16` 帧

## 训练数据

6号必须使用 2 号给出的统一数据集划分：

- `data/fatigue_b/labels/train.csv`
- `data/fatigue_b/labels/val.csv`
- `data/fatigue_b/labels/test.csv`

不能重新按帧随机划分，否则会产生数据泄漏。

## 模型结构

- 前处理：视频抽帧 + 人脸关键点 + 特征构造
- 主模型：`TemporalFatigueModel`
- 编码方式：`BiLSTM` 或 `GRU`
- 输出头：
  - `yawn_head`
  - `look_away_head`
  - `fatigue_head`

## 输出接口

```json
{
  "module": "fatigue",
  "model_name": "fatigue_b",
  "frame_id": 128,
  "timestamp": 5.12,
  "label": "Fatigued Driving",
  "confidence": 0.86,
  "indicators": {
    "yawn_score": 0.21,
    "look_away_score": 0.45,
    "fatigue_score": 0.86
  },
  "risk_level": "high"
}
```

## 与 5号 的区别

- 5号：规则法，可解释性更强
- 6号：学习法，更依赖训练数据和时序建模

两者输出一致，便于 7 号统一集成和做 A/B 对比。

## 6号最终交付

- 特征提取脚本
- 窗口构建脚本
- 训练脚本
- 评估脚本
- 推理脚本
- 最优模型权重
- 指标结果

## 答辩时可直接使用的话术

“我负责的是疲劳检测算法 B。考虑到疲劳驾驶具有明显的时间连续性，我们没有仅依赖单帧特征，而是将连续窗口特征输入时序模型进行建模。模型输出打哈欠、视线偏移和疲劳状态三个分数，再统一映射为 Normal、Yawning、Looking Around、Fatigued Driving 四类结果。这一方案与 5 号的规则法保持统一接口，便于后续系统集成和效果对比。”
