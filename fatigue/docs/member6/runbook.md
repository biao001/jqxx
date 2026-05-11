# 6号运行手册

## 1. 安装依赖

```bash
pip install -r requirements-fatigue-b.txt
```

## 2. 提取视频特征

```bash
python src/fatigue_b/extract_features.py ^
  --input-dir data/fatigue_b/raw/self_recorded/videos ^
  --output-dir data/fatigue_b/processed/features
```

## 3. 构建训练窗口

```bash
python src/fatigue_b/build_windows.py ^
  --labels-csv data/fatigue_b/labels/train.csv ^
  --features-dir data/fatigue_b/processed/features ^
  --output-npz data/fatigue_b/processed/windows/train_windows.npz ^
  --output-metadata data/fatigue_b/processed/windows/train_metadata.csv
```

```bash
python src/fatigue_b/build_windows.py ^
  --labels-csv data/fatigue_b/labels/val.csv ^
  --features-dir data/fatigue_b/processed/features ^
  --output-npz data/fatigue_b/processed/windows/val_windows.npz ^
  --output-metadata data/fatigue_b/processed/windows/val_metadata.csv
```

```bash
python src/fatigue_b/build_windows.py ^
  --labels-csv data/fatigue_b/labels/test.csv ^
  --features-dir data/fatigue_b/processed/features ^
  --output-npz data/fatigue_b/processed/windows/test_windows.npz ^
  --output-metadata data/fatigue_b/processed/windows/test_metadata.csv
```

## 4. 训练模型

```bash
python src/fatigue_b/train.py ^
  --train-npz data/fatigue_b/processed/windows/train_windows.npz ^
  --val-npz data/fatigue_b/processed/windows/val_windows.npz ^
  --output-dir outputs/fatigue_b/checkpoints ^
  --epochs 25 ^
  --batch-size 16
```

## 5. 评估模型

```bash
python src/fatigue_b/eval.py ^
  --checkpoint outputs/fatigue_b/checkpoints/best_model.pt ^
  --test-npz data/fatigue_b/processed/windows/test_windows.npz ^
  --output-dir outputs/fatigue_b/reports ^
  --search-thresholds
```

## 6. 输出推理结果

```bash
python src/fatigue_b/infer.py ^
  --checkpoint outputs/fatigue_b/checkpoints/best_model.pt ^
  --features-csv data/fatigue_b/processed/features/self_s19_clip01.csv ^
  --output-json outputs/fatigue_b/reports/demo_predictions.json ^
  --window-size 16 ^
  --stride 8
```

## 6.1 预览模式：单视频直接生成可视化结果

如果还没有训练好的 `best_model.pt`，可以先用预览模式直接生成带标签的视频：

```bash
python src/fatigue_b/extract_features.py ^
  --video data/fatigue_b/raw/internet_demo/yawn_drive.mp4 ^
  --output-dir data/fatigue_b/processed/features
```

```bash
python src/fatigue_b/demo_video.py ^
  --video data/fatigue_b/raw/internet_demo/yawn_drive.mp4 ^
  --features-csv data/fatigue_b/processed/features/yawn_drive.csv ^
  --output-video outputs/fatigue_b/demo/yawn_drive_annotated.mp4 ^
  --output-json outputs/fatigue_b/demo/yawn_drive_predictions.json ^
  --window-size 16
```

如果已经有训练好的 `best_model.pt`，可以追加：

```bash
python src/fatigue_b/demo_video.py ^
  --video data/fatigue_b/raw/internet_demo/yawn_drive.mp4 ^
  --features-csv data/fatigue_b/processed/features/yawn_drive.csv ^
  --output-video outputs/fatigue_b/demo/yawn_drive_annotated.mp4 ^
  --output-json outputs/fatigue_b/demo/yawn_drive_predictions.json ^
  --window-size 16 ^
  --checkpoint outputs/fatigue_b/checkpoints/best_model.pt

默认就是设备自动选择：

- 有 `CUDA` 就优先用 `GPU`
- 没有可用 `GPU` 就自动回退到 `CPU`

如果你要强制指定，也可以额外传：

```bash
--device cuda
```

或

```bash
--device cpu
```
```

## 7. 给 7号 的交付

- `outputs/fatigue_b/checkpoints/best_model.pt`
- `outputs/fatigue_b/reports/report.json`
- `outputs/fatigue_b/reports/confusion_matrix.csv`
- `outputs/fatigue_b/reports/demo_predictions.json`
- `src/fatigue_b/infer.py`
