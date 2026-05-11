# 2号交付说明：疲劳数据集统一规范

## 目标

2号负责把公开数据集和自采视频整理成一套统一疲劳数据集，供 5 号和 6 号共用。2号最终不负责模型训练，重点是数据标准化、标签统一、训练验证测试划分和数据统计。

## 建议数据来源

- `NTHU-DDD`
- `YawDD`
- `UTA-RLDD`
- 自采 demo 视频

## 对外统一标签

- `Normal`
- `Yawning`
- `Looking Around`
- `Fatigued Driving`

## 内部训练标签

- `is_yawning`
- `is_look_away`
- `is_fatigued`

## 统一标签映射

### NTHU-DDD

- `yawning -> is_yawning=1`
- `looking_aside -> is_look_away=1`
- `sleepy_eyes -> is_fatigued=1`
- `nodding -> is_fatigued=1`
- `normal -> all zero`

### YawDD

- `yawning -> is_yawning=1`
- `normal -> all zero`
- `talking_singing -> all zero`

### UTA-RLDD

- `alertness -> all zero`
- `low_vigilance -> is_fatigued=1`
- `drowsiness -> is_fatigued=1`

### self_recorded

- `yawning -> is_yawning=1`
- `looking_around -> is_look_away=1`
- `fatigued_driving -> is_fatigued=1`
- `normal -> all zero`

## 标注文件格式

```csv
video_id,start_frame,end_frame,subject_id,source_dataset,is_yawning,is_look_away,is_fatigued,split
```

## 划分原则

- 必须按 `subject_id` 划分
- 训练集和测试集不能出现同一个人
- 默认比例建议 `70/15/15`

## 2号最终交付

- `train.csv`
- `val.csv`
- `test.csv`
- 数据统计表
- 标签映射说明
- 给 6 号的交接说明
