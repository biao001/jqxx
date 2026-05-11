"""
train.py — 训练自定义 YOLOv8 检测器 (针对安全带 / 抽烟)

使用场景: 当官方 COCO 权重不包含目标类别（如 seatbelt、cigarette）
          时，用本脚本在自建/公开数据集上微调。

公开数据集推荐：
  1. Seatbelt:
     - Roboflow Universe "seatbelt-detection"
       https://universe.roboflow.com/ (搜索 seatbelt)
     - Kaggle "Seat Belt Detection"
  2. Cigarette / Smoking:
     - Roboflow "cigarette-detection-aycnc"
     - "Smoking Person Detection" (Kaggle)
  3. Distracted Driver (综合 10 类):
     - State Farm Distracted Driver Detection (Kaggle)
     - AUC Distracted Driver Dataset V2

数据集约定 (YOLO 格式):
  dataset/
    images/
      train/*.jpg
      val/*.jpg
    labels/
      train/*.txt   # 每行: class cx cy w h (归一化)
      val/*.txt
    dataset.yaml

示例 dataset.yaml (seatbelt):
  path: D:/Desktop/DMS/behavior_algo_a/data/seatbelt
  train: images/train
  val: images/val
  nc: 2
  names: ['belt', 'no_belt']

运行示例:
  python scripts/train.py --data data/seatbelt/dataset.yaml \
                          --base yolov8n.pt --epochs 60 --imgsz 640 \
                          --project runs/seatbelt --name exp1

训练完成后:
  best.pt 在 runs/seatbelt/exp1/weights/best.pt
  拷贝到 models/ 目录，Detector 会自动加载
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset.yaml 路径")
    ap.add_argument("--base", default="yolov8n.pt",
                    help="基础权重（从 COCO 迁移）")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0",
                    help="GPU id 或 cpu；多卡用 0,1")
    ap.add_argument("--project", default="runs/behavior")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--patience", type=int, default=20,
                    help="早停 patience")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml 不存在: {data_yaml}")

    print(f"[train] 基础权重: {args.base}")
    print(f"[train] 数据集:   {data_yaml}")
    print(f"[train] 轮数:     {args.epochs}")

    model = YOLO(args.base)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        # 数据增强（模拟驾驶室复杂光照）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # 优化器
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # 损失权重
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )

    print(f"[train] 完成。best 权重: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
