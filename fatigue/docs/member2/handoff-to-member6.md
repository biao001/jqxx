# 2号给6号的交接模板

## 已交付文件

- `data/fatigue_b/labels/train.csv`
- `data/fatigue_b/labels/val.csv`
- `data/fatigue_b/labels/test.csv`
- `src/fatigue_b/label_map.yaml`

## 约定

- 6号不能修改标签名
- 6号不能重新随机按帧切分数据
- 6号训练和评估必须沿用这三份 split 文件

## 关键字段说明

- `video_id`: 对应视频唯一标识
- `start_frame/end_frame`: 标注窗口范围
- `subject_id`: 用于防止数据泄漏
- `source_dataset`: 原始来源
- `is_yawning/is_look_away/is_fatigued`: 训练目标

## 建议给 6号 的输入特征

- `ear`
- `mar`
- `head_yaw`
- `head_pitch`
- `head_roll`
- `drowsy_prob`

## 需要 6号 输出的统一字段

```json
{
  "module": "fatigue",
  "model_name": "fatigue_b",
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
