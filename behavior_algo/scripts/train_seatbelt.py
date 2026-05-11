"""
train_seatbelt.py — 一键训练安全带检测模型

三种数据来源（按推荐度排序）:

  ① Roboflow Universe (最简单, 需免费 API key)
     https://universe.roboflow.com/?q=seatbelt
     注册后 workspace → Settings → API → 复制 API Key

     用法:
       python scripts/train_seatbelt.py --source roboflow \
           --rf-api-key YOUR_KEY \
           --rf-workspace 2tech \
           --rf-project seat-belt-detection-udcfg \
           --rf-version 1 \
           --epochs 40

  ② 本地已有 YOLO 格式数据集
     按 data/seatbelt_example.yaml 约定路径整理后:
       python scripts/train_seatbelt.py --source local \
           --data data/seatbelt_example.yaml --epochs 50

  ③ Kaggle 数据集 (手动下载后用 local 模式)
     https://www.kaggle.com/datasets/prajjwalkumarpanzade/smoking-and-drinking-dataset-for-yolo
     https://www.kaggle.com/datasets/vitaminc/cigarette-smoker-detection
     解压到 data/seatbelt/ 后运行 local 模式

训练完成后, best.pt 会自动拷贝到 models/seatbelt.pt，检测器下次启动即可加载。
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs" / "seatbelt"


def prepare_roboflow(api_key, workspace, project, version):
    """从 Roboflow Universe 下载 YOLOv8 格式数据集"""
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("[!] 请先安装: pip install roboflow")

    print(f"[rf] 连接 Roboflow workspace={workspace}/{project} v{version}")
    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project).version(version).download(
        "yolov8", location=str(DATA_DIR / "seatbelt_rf"))
    yaml_path = Path(ds.location) / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Roboflow 下载后未找到 data.yaml: {yaml_path}")
    print(f"[rf] 数据集已下载到 {yaml_path.parent}")
    return str(yaml_path)


def train(data_yaml, base, epochs, imgsz, batch, device, name):
    print(f"[train] base={base}  data={data_yaml}  epochs={epochs}")
    model = YOLO(base)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(RUNS_DIR),
        name=name,
        patience=15,
        # 驾驶室光照增强
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0, mixup=0.1,
        optimizer="AdamW", lr0=0.001, weight_decay=0.0005,
        warmup_epochs=3.0,
    )
    best = RUNS_DIR / name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"训练产物不存在: {best}")
    target = MODELS_DIR / "seatbelt.pt"
    shutil.copy(best, target)
    print(f"[train] 完成。已拷贝 {best} → {target}")
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["roboflow", "local"],
                    default="roboflow")
    # Roboflow
    ap.add_argument("--rf-api-key", default=None)
    ap.add_argument("--rf-workspace", default="2tech")
    ap.add_argument("--rf-project",   default="seat-belt-detection-udcfg")
    ap.add_argument("--rf-version",   type=int, default=1)
    # Local
    ap.add_argument("--data", default="data/seatbelt_example.yaml")
    # Common
    ap.add_argument("--base",   default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz",  type=int, default=640)
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--device", default="0", help="0 或 cpu")
    ap.add_argument("--name",   default="exp1")
    args = ap.parse_args()

    if args.source == "roboflow":
        if not args.rf_api_key:
            raise SystemExit(
                "[!] Roboflow 模式需 --rf-api-key。\n"
                "   免费注册 https://roboflow.com/ → Settings → API\n"
                "   或改用 --source local 配合本地数据集。")
        data_yaml = prepare_roboflow(args.rf_api_key, args.rf_workspace,
                                     args.rf_project, args.rf_version)
    else:
        data_yaml = str((ROOT / args.data).resolve())
        if not Path(data_yaml).exists():
            raise SystemExit(f"[!] data.yaml 不存在: {data_yaml}")

    train(data_yaml, args.base, args.epochs, args.imgsz, args.batch,
          args.device, args.name)


if __name__ == "__main__":
    main()
