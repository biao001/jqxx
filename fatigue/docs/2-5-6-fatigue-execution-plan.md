# 2、5、6号疲劳检测执行文档

## 目标

把 2号、5号、6号 的工作合并成一条可执行流程：

- 2号：下载、整理、标注疲劳检测数据集，输出统一的 `train.csv`、`val.csv`、`test.csv`。
- 5号：完成疲劳检测特征提取和基础规则判断。
- 6号：完成时序窗口构建、模型训练、评估、推理接口和最终交付。

最终交付给 7号 的内容：

- `outputs/fatigue_b/checkpoints/best_model.pt`
- `outputs/fatigue_b/reports/report.json`
- `outputs/fatigue_b/reports/confusion_matrix.csv`
- `outputs/fatigue_b/reports/demo_predictions.json`
- `src/fatigue_b/infer.py`
- 疲劳检测演示视频或截图

---

## 一、数据集选择和下载策略

### 1. 推荐数据集

本项目疲劳检测使用以下数据来源：

| 数据集 | 主要用途 | 对应标签 | 获取方式 |
| --- | --- | --- | --- |
| YawDD | 打哈欠检测 | `Yawning` | 公开页面可下载 |
| UTA-RLDD | 疲劳/低警觉检测 | `Fatigued Driving` | 官网或 Kaggle |
| NTHU-DDD | 疲劳、打哈欠、视线偏移 | `Yawning`、`Looking Around`、`Fatigued Driving` | 通常需要申请或从授权镜像获取 |
| 自采视频 | 系统演示补充 | 四类都可覆盖 | 自己录制 |

### 2. 下载注意事项

不要把未授权的大型数据集直接提交到仓库。建议只提交：

- 标签 CSV
- 数据说明文档
- 小型 demo 视频
- 脚本和模型结果

大型原始数据放在：

```text
data/fatigue_b/raw/
```

如果数据集来自 Kaggle、Google Drive、学校网盘或需要填写申请表，执行人需要先完成登录或授权，再把下载后的文件放入指定目录。

---

## 二、2号具体执行任务

### 任务 2.1：建立原始数据目录

创建目录：

```text
data/fatigue_b/raw/YawDD/
data/fatigue_b/raw/UTA-RLDD/
data/fatigue_b/raw/NTHU-DDD/
data/fatigue_b/raw/self_recorded/
```

建议视频命名格式：

```text
yawdd_s001_yawn_001.mp4
uta_s001_drowsy_001.mp4
nthu_s001_look_away_001.mp4
self_s001_normal_001.mp4
```

命名原则：

- 文件名不要有空格。
- 每个视频名必须能作为 `video_id`。
- `video_id` 后续要和特征 CSV 文件名一致。

### 任务 2.2：下载 YawDD

YawDD 用于打哈欠检测。

操作：

1. 打开 YawDD 官方数据页。
2. 下载 yawning、normal、talking/singing 相关视频。
3. 解压到：

```text
data/fatigue_b/raw/YawDD/
```

标注规则：

```text
yawning -> is_yawning=1,is_look_away=0,is_fatigued=0
normal -> is_yawning=0,is_look_away=0,is_fatigued=0
talking/singing -> is_yawning=0,is_look_away=0,is_fatigued=0
```

### 任务 2.3：下载 UTA-RLDD

UTA-RLDD 用于疲劳/低警觉检测。

操作：

1. 优先从 UTA-RLDD 官方页面获取。
2. 如果官网下载不方便，可以使用 Kaggle 镜像。
3. 解压到：

```text
data/fatigue_b/raw/UTA-RLDD/
```

标注规则：

```text
alertness -> is_yawning=0,is_look_away=0,is_fatigued=0
low_vigilance -> is_yawning=0,is_look_away=0,is_fatigued=1
drowsiness -> is_yawning=0,is_look_away=0,is_fatigued=1
```

### 任务 2.4：获取 NTHU-DDD

NTHU-DDD 用于补充视线偏移、打哈欠、困倦、点头等状态。

操作：

1. 如果能申请官方数据，优先使用官方授权数据。
2. 如果课程允许使用公开镜像，可以下载镜像版本。
3. 解压到：

```text
data/fatigue_b/raw/NTHU-DDD/
```

标注规则：

```text
yawning -> is_yawning=1,is_look_away=0,is_fatigued=0
looking_aside / looking_around -> is_yawning=0,is_look_away=1,is_fatigued=0
sleepy_eyes -> is_yawning=0,is_look_away=0,is_fatigued=1
nodding -> is_yawning=0,is_look_away=0,is_fatigued=1
normal -> is_yawning=0,is_look_away=0,is_fatigued=0
```

如果暂时拿不到 NTHU-DDD，先用自采视频补充 `Looking Around` 类别，保证流程能跑完。

### 任务 2.5：录制自采 demo 视频

至少录制以下四类短视频，每类 2 到 5 段，每段 5 到 20 秒：

```text
Normal
Yawning
Looking Around
Fatigued Driving
```

建议放入：

```text
data/fatigue_b/raw/self_recorded/
```

标注规则：

```text
Normal -> 0,0,0
Yawning -> 1,0,0
Looking Around -> 0,1,0
Fatigued Driving -> 0,0,1
```

### 任务 2.6：生成统一标注 CSV

最终必须生成：

```text
data/fatigue_b/labels/all.csv
data/fatigue_b/labels/train.csv
data/fatigue_b/labels/val.csv
data/fatigue_b/labels/test.csv
```

CSV 字段固定为：

```csv
video_id,start_frame,end_frame,subject_id,source_dataset,is_yawning,is_look_away,is_fatigued,split
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `video_id` | 视频唯一 ID，不带扩展名 |
| `start_frame` | 片段开始帧 |
| `end_frame` | 片段结束帧 |
| `subject_id` | 驾驶员或被试 ID |
| `source_dataset` | `YawDD`、`UTA-RLDD`、`NTHU-DDD`、`self_recorded` |
| `is_yawning` | 是否打哈欠 |
| `is_look_away` | 是否视线偏移 |
| `is_fatigued` | 是否疲劳驾驶 |
| `split` | `train`、`val`、`test` |

示例：

```csv
video_id,start_frame,end_frame,subject_id,source_dataset,is_yawning,is_look_away,is_fatigued,split
yawdd_s001_yawn_001,120,260,s001,YawDD,1,0,0,train
uta_s003_drowsy_001,80,360,s003,UTA-RLDD,0,0,1,val
nthu_s005_look_away_001,50,180,s005,NTHU-DDD,0,1,0,test
self_s007_normal_001,0,240,s007,self_recorded,0,0,0,train
```

划分要求：

- 必须按 `subject_id` 划分。
- 同一个 `subject_id` 不能同时出现在 train、val、test。
- 建议比例：`70% train / 15% val / 15% test`。
- 每个 split 里尽量都有四类样本。

2号完成标准：

- `all.csv`、`train.csv`、`val.csv`、`test.csv` 存在。
- 每个 CSV 字段完整。
- 每类至少有可用于 demo 的样本。
- 数据统计表已生成。

---

## 三、5号具体执行任务

5号负责疲劳特征提取和基础规则判断。

### 任务 5.1：安装依赖

执行：

```powershell
pip install -r requirements-fatigue-b.txt
```

### 任务 5.2：提取视频特征

对原始视频执行特征提取。

单个视频：

```powershell
python src/fatigue_b/extract_features.py --video data/fatigue_b/raw/self_recorded/self_s001_normal_001.mp4 --output-dir data/fatigue_b/processed/features
```

目录批量：

```powershell
python src/fatigue_b/extract_features.py --input-dir data/fatigue_b/raw/self_recorded --output-dir data/fatigue_b/processed/features
```

每个视频会生成一个特征 CSV：

```text
data/fatigue_b/processed/features/{video_id}.csv
```

特征字段：

```csv
frame_id,timestamp,ear,mar,head_yaw,head_pitch,head_roll,drowsy_prob,face_valid
```

### 任务 5.3：检查特征是否有效

检查每个特征 CSV：

- `face_valid` 不能长期为 0。
- `ear` 应该在闭眼时下降。
- `mar` 应该在打哈欠时升高。
- `head_yaw` 应该在左右看时变化明显。
- `drowsy_prob` 应该在闭眼、低头、困倦时升高。

### 任务 5.4：生成预览演示视频

没有训练模型前，可以用规则预览模式生成演示视频：

```powershell
python src/fatigue_b/demo_video.py --video data/fatigue_b/raw/self_recorded/self_s001_normal_001.mp4 --features-csv data/fatigue_b/processed/features/self_s001_normal_001.csv --output-video outputs/fatigue_b/demo/self_s001_normal_001_annotated.mp4 --output-json outputs/fatigue_b/demo/self_s001_normal_001_predictions.json --window-size 16
```

5号完成标准：

- 每个标注视频都有对应特征 CSV。
- 至少生成 1 到 3 个预览演示视频。
- 能解释 `EAR`、`MAR`、头部姿态和 `drowsy_prob` 的含义。

---

## 四、6号具体执行任务

6号负责窗口构建、模型训练、评估和最终推理。

### 任务 6.1：构建训练窗口

执行：

```powershell
python src/fatigue_b/build_windows.py --labels-csv data/fatigue_b/labels/train.csv --features-dir data/fatigue_b/processed/features --output-npz data/fatigue_b/processed/windows/train_windows.npz --output-metadata data/fatigue_b/processed/windows/train_metadata.csv --window-size 16
```

```powershell
python src/fatigue_b/build_windows.py --labels-csv data/fatigue_b/labels/val.csv --features-dir data/fatigue_b/processed/features --output-npz data/fatigue_b/processed/windows/val_windows.npz --output-metadata data/fatigue_b/processed/windows/val_metadata.csv --window-size 16
```

```powershell
python src/fatigue_b/build_windows.py --labels-csv data/fatigue_b/labels/test.csv --features-dir data/fatigue_b/processed/features --output-npz data/fatigue_b/processed/windows/test_windows.npz --output-metadata data/fatigue_b/processed/windows/test_metadata.csv --window-size 16
```

输出：

```text
data/fatigue_b/processed/windows/train_windows.npz
data/fatigue_b/processed/windows/val_windows.npz
data/fatigue_b/processed/windows/test_windows.npz
```

### 任务 6.2：训练疲劳检测模型

执行：

```powershell
python src/fatigue_b/train.py --train-npz data/fatigue_b/processed/windows/train_windows.npz --val-npz data/fatigue_b/processed/windows/val_windows.npz --output-dir outputs/fatigue_b/checkpoints --epochs 25 --batch-size 16 --rnn-type lstm
```

输出：

```text
outputs/fatigue_b/checkpoints/best_model.pt
outputs/fatigue_b/checkpoints/history.json
```

### 任务 6.3：评估模型

执行：

```powershell
python src/fatigue_b/eval.py --checkpoint outputs/fatigue_b/checkpoints/best_model.pt --test-npz data/fatigue_b/processed/windows/test_windows.npz --output-dir outputs/fatigue_b/reports --search-thresholds
```

输出：

```text
outputs/fatigue_b/reports/report.json
outputs/fatigue_b/reports/confusion_matrix.csv
```

### 任务 6.4：生成正式推理结果

选择一段 demo 视频对应的特征 CSV，执行：

```powershell
python src/fatigue_b/infer.py --checkpoint outputs/fatigue_b/checkpoints/best_model.pt --features-csv data/fatigue_b/processed/features/self_s001_normal_001.csv --output-json outputs/fatigue_b/reports/demo_predictions.json --window-size 16 --stride 8
```

输出 JSON 格式必须包含：

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

### 任务 6.5：用训练模型生成最终演示视频

执行：

```powershell
python src/fatigue_b/demo_video.py --video data/fatigue_b/raw/self_recorded/self_s001_normal_001.mp4 --features-csv data/fatigue_b/processed/features/self_s001_normal_001.csv --output-video outputs/fatigue_b/demo/final_fatigue_demo.mp4 --output-json outputs/fatigue_b/demo/final_fatigue_demo_predictions.json --window-size 16 --checkpoint outputs/fatigue_b/checkpoints/best_model.pt
```

6号完成标准：

- `best_model.pt` 存在。
- `history.json` 存在。
- `report.json` 存在。
- `confusion_matrix.csv` 存在。
- `demo_predictions.json` 存在。
- 能用 `infer.py` 被 7号 调用。

---

## 五、建议验收清单

### 2号验收

- [ ] 原始数据已放入 `data/fatigue_b/raw/`
- [ ] `all.csv` 已生成
- [ ] `train.csv` 已生成
- [ ] `val.csv` 已生成
- [ ] `test.csv` 已生成
- [ ] CSV 字段完整
- [ ] 同一个 `subject_id` 没有跨 split
- [ ] 四类标签都有样本
- [ ] 数据统计表已生成

### 5号验收

- [ ] 已安装依赖
- [ ] 已下载 MediaPipe `face_landmarker.task`
- [ ] 已生成每个视频的特征 CSV
- [ ] 特征 CSV 包含 `ear`、`mar`、`head_yaw`、`head_pitch`、`head_roll`、`drowsy_prob`
- [ ] 至少生成一个规则预览演示视频

### 6号验收

- [ ] 已生成 `train_windows.npz`
- [ ] 已生成 `val_windows.npz`
- [ ] 已生成 `test_windows.npz`
- [ ] 已训练生成 `best_model.pt`
- [ ] 已生成训练历史 `history.json`
- [ ] 已生成评估报告 `report.json`
- [ ] 已生成混淆矩阵 `confusion_matrix.csv`
- [ ] 已生成正式推理 JSON
- [ ] 已生成最终演示视频

---

## 六、当前项目状态对照

当前仓库已经有：

- `data/fatigue_b/labels/train.csv`
- `data/fatigue_b/labels/val.csv`
- `data/fatigue_b/labels/test.csv`
- `data/fatigue_b/processed/features/*.csv`
- `data/fatigue_b/processed/windows/*.npz`
- `src/fatigue_b/extract_features.py`
- `src/fatigue_b/build_windows.py`
- `src/fatigue_b/train.py`
- `src/fatigue_b/eval.py`
- `src/fatigue_b/infer.py`
- `outputs/fatigue_b/demo/dms测试视频_annotated.mp4`

当前仓库还缺：

- `outputs/fatigue_b/checkpoints/best_model.pt`
- `outputs/fatigue_b/checkpoints/history.json`
- `outputs/fatigue_b/reports/report.json`
- `outputs/fatigue_b/reports/confusion_matrix.csv`
- 正式训练模型生成的 `demo_predictions.json`

因此下一步优先级：

1. 扩充 2号 数据，至少保证每类有足够样本。
2. 重新提取所有新增视频特征。
3. 重新构建窗口。
4. 训练模型。
5. 评估模型。
6. 生成正式推理结果和演示视频。

---

## 七、风险和处理方式

### 风险 1：NTHU-DDD 无法直接下载

处理：

- 先申请官方授权。
- 如果课程允许，用公开镜像。
- 如果仍拿不到，先用自采视频补足 `Looking Around` 和 `Fatigued Driving`。

### 风险 2：数据量太小导致训练结果不稳定

处理：

- 每类至少准备 20 到 50 个片段。
- train、val、test 都要包含每个类别。
- 不要只用 11 条样本做正式结论。

### 风险 3：视频和 `video_id` 对不上

处理：

- 视频名、标签里的 `video_id`、特征 CSV 文件名必须一致。
- 例如：

```text
self_s001_normal_001.mp4
self_s001_normal_001.csv
video_id=self_s001_normal_001
```

### 风险 4：同一驾驶员泄漏到测试集

处理：

- 按 `subject_id` 划分。
- 同一个 `subject_id` 只能属于一个 split。

### 风险 5：MediaPipe 检测不到人脸

处理：

- 检查视频角度和光照。
- 优先使用正脸或驾驶员脸部清晰的视频。
- 对 `face_valid=0` 过多的视频重新选择片段。
