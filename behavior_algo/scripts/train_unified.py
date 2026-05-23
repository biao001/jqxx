"""
train_unified.py — 一键训练 8 类 DMS 危险行为检测器

前置: 运行过 scripts/merge_datasets.py 生成 data/dms_unified/

用法:
  # GPU (推荐)
  python scripts/train_unified.py --device 0 --epochs 40

  # CPU (慢，仅冒烟测试)
  python scripts/train_unified.py --device cpu --epochs 5 --imgsz 320

  # 断点续训
  python scripts/train_unified.py --resume

  # 更大模型 (精度 ↑ 速度 ↓)
  python scripts/train_unified.py --base yolov8s.pt --epochs 60

训练完成自动拷贝 best.pt → models/unified.pt
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dms_unified" / "data.yaml"
MODELS = ROOT / "models"
RUNS = ROOT / "runs" / "unified"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA),
                    help="数据集 data.yaml 路径(默认 dms_unified；重训用 data/reckless_mapped/data.yaml)")
    ap.add_argument("--base", default="yolov8n.pt",
                    help="基础权重 (yolov8n/s/m.pt)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0",
                    help="GPU id 如 0 / 0,1 / cpu")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=2,
                    help="DataLoader workers; 8 太多会撑爆 /dev/shm 导致 OOM kill")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"[!] {data_path} 不存在，请先准备数据集 "
            f"(reckless 重训: python scripts/build_reckless_mapped.py)")

    MODELS.mkdir(exist_ok=True)

    print(f"[train] base={args.base}")
    print(f"[train] data={data_path}")
    print(f"[train] epochs={args.epochs}  imgsz={args.imgsz}  device={args.device}")

    model = YOLO(args.base)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(RUNS),
        name=args.name,
        resume=args.resume,
        patience=args.patience,
        # 驾驶室光照增强
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0, mixup=0.1, copy_paste=0.1,
        optimizer="AdamW", lr0=0.001, lrf=0.01,
        weight_decay=0.0005, warmup_epochs=3.0,
        # 损失权重 - 小物体（手机/香烟）提权
        box=7.5, cls=0.5, dfl=1.5,
    )

    best = RUNS / args.name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"找不到 best.pt: {best}")
    target = MODELS / "unified.pt"
    shutil.copy(best, target)
    print(f"\n[ok] best.pt → {target}")
    print(f"[use] python live_client.py --unified {target} --style monitor")


if __name__ == "__main__":
    main()
