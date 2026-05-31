"""
train_unified.py — 一键训练 7 类 DMS 危险行为检测器

数据集: datasets/reckless_mapped/(由 scripts/build_reckless_mapped.py 生成)

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
import json
import shutil
import time
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "datasets" / "reckless_mapped" / "data.yaml"
MODELS = ROOT / "models"
RUNS = ROOT / "runs" / "unified"


def build_data_yaml(base_yaml: str, include_incremental: bool, all_splits: bool) -> str:
    """按需把数据划分改写成临时 data.yaml：
    - include_incremental: train 追加 unassigned/images(增量未并入数据)。
    - all_splits(演示过拟合): train 追加 valid/test，并把 val 设回 train(在训练数据上"验证"，
      指标虚高、看着很准——仅用于现场演示展示实时性，不代表泛化能力)。
    不改动则原样返回。"""
    if not (include_incremental or all_splits):
        return base_yaml
    import tempfile
    import yaml

    data = yaml.safe_load(Path(base_yaml).read_text())
    ds_path = Path(data.get("path") or Path(base_yaml).parent)
    train_dirs = ["train/images"]
    if include_incremental:
        # 只纳入「有非空标注」的增量图：unassigned 多为未标注暂存帧，YOLO 会把无标注图
        # 当背景负样本，反而压制 phone/smoking 等目标类(适得其反)。用显式图片清单 .txt 传入。
        u_img = ds_path / "unassigned" / "images"
        u_lbl = ds_path / "unassigned" / "labels"
        labeled: list[str] = []
        n_total = 0
        if u_img.exists():
            for img in sorted(u_img.iterdir()):
                if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                n_total += 1
                lbl = u_lbl / (img.stem + ".txt")
                if lbl.exists() and lbl.read_text().strip():  # 仅非空标注
                    labeled.append(str(img.resolve()))
        if labeled:
            lst = Path(tempfile.gettempdir()) / f"dms_incremental_{ds_path.name}.txt"
            lst.write_text("\n".join(labeled))
            train_dirs.append(str(lst))  # ultralytics 支持以 .txt 图片清单作为训练源
            print(f"[train] 增量纳入 {len(labeled)}/{n_total} 张已标注图(未标注的已跳过，避免当背景压制目标类)")
        else:
            print(f"[train][warn] unassigned 有 {n_total} 张图但无非空标注 → 跳过纳入"
                  f"(避免把未标注图当背景)。请先标注或用「并入训练集」。")
    if all_splits:
        for sub in ("valid/images", "test/images"):
            if (ds_path / sub).exists():
                train_dirs.append(sub)
        data["val"] = "train/images"  # 演示：在训练集上验证 → 过拟合(指标虚高)
    data["train"] = train_dirs
    data["path"] = str(ds_path)  # 锚定数据集根，确保相对 train/val 不相对 /tmp 解析
    out = Path(tempfile.gettempdir()) / f"dms_train_{ds_path.name}.yaml"
    out.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    print(f"[train] 改写数据划分 → {out}  train={train_dirs}  val={data.get('val')}")
    return str(out)


def dump_per_class(trainer):
    """每个 fit-epoch 结束把「各类别 mAP」写到 run 目录的 per_class.json，供前端实时展示。

    后端 status() 读取它 → 训练时就能看到每个标签(phone/smoking…)的 mAP，而不只是总 mAP50。
    """
    try:
        m = trainer.validator.metrics  # DetMetrics
        names = getattr(m, "names", {}) or {}
        per: dict[str, dict] = {}
        for i, ci in enumerate(m.ap_class_index):
            p, r, ap50, ap = m.box.class_result(i)
            per[str(names.get(int(ci), int(ci)))] = {
                "map50": round(float(ap50), 4), "map": round(float(ap), 4),
                "p": round(float(p), 4), "r": round(float(r), 4),
            }
        out = {
            "epoch": int(getattr(trainer, "epoch", 0)) + 1,
            "map50": round(float(m.box.map50), 4),
            "map": round(float(m.box.map), 4),
            "per_class": per,
        }
        (Path(trainer.save_dir) / "per_class.json").write_text(json.dumps(out, ensure_ascii=False))
    except Exception as exc:  # 不能因为可视化失败影响训练
        print(f"[train][warn] per_class 写出失败: {exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA),
                    help="数据集 data.yaml 路径(默认 datasets/reckless_mapped)")
    ap.add_argument("--base", default="yolov8n.pt",
                    help="基础权重 (yolov8n/s/m.pt)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0",
                    help="GPU id 如 0 / 0,1 / cpu")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--out", default="unified.pt",
                    help="训练完成后 best.pt 拷贝到 models/<out>(多模型项目用)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--freeze", type=int, default=0,
                    help="冻结前 N 层(0=不冻结)；增量微调防遗忘/过拟合时可冻 backbone(yolov8 用 10)")
    ap.add_argument("--lr0", type=float, default=0.001,
                    help="初始学习率；从已训权重续训建议更低(1e-4~5e-4)")
    ap.add_argument("--all-splits", action="store_true",
                    help="演示模式：valid/test 也并入训练并以 train 当验证集(过拟合、指标虚高，仅现场演示用)")
    ap.add_argument("--include-incremental", action="store_true",
                    help="把 unassigned(增量/未并入训练)也纳入本次训练")
    ap.add_argument("--workers", type=int, default=2,
                    help="DataLoader workers; 8 太多会撑爆 /dev/shm 导致 OOM kill")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"[!] {data_path} 不存在，请先准备数据集 "
            f"(reckless 重训: python scripts/build_reckless_mapped.py)")

    # 演示过拟合 / 纳入增量：按需改写成临时 data.yaml
    effective_data = build_data_yaml(str(data_path), args.include_incremental, args.all_splits)

    MODELS.mkdir(exist_ok=True)

    print(f"[train] base={args.base}")
    print(f"[train] data={effective_data}")
    print(f"[train] epochs={args.epochs}  imgsz={args.imgsz}  device={args.device}  "
          f"freeze={args.freeze}  lr0={args.lr0}  all_splits={args.all_splits}")

    model = YOLO(args.base)
    model.add_callback("on_fit_epoch_end", dump_per_class)  # 逐 epoch 写各类别 mAP
    model.train(
        data=str(effective_data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(RUNS),
        name=args.name,
        resume=args.resume,
        patience=args.patience,
        freeze=args.freeze,  # 0=不冻结；冻 backbone(如 10)防遗忘/过拟合
        # 驾驶室光照增强
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0, mixup=0.1, copy_paste=0.1,
        optimizer="AdamW", lr0=args.lr0, lrf=0.01,
        weight_decay=0.0005, warmup_epochs=3.0,
        # 损失权重 - 小物体（手机/香烟）提权
        box=7.5, cls=0.5, dfl=1.5,
    )

    best = RUNS / args.name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"找不到 best.pt: {best}")
    target = MODELS / Path(args.out).name
    shutil.copy(best, target)
    print(f"\n[ok] best.pt → {target}")

    # 训练溯源：把「基于哪个数据集/权重/参数训出来的 + 最终各类别 mAP」写到 <model>.meta.json，
    # 随模型长期保留(模型管理列表会展示)，避免「不知道这一轮模型是用什么数据训的」。
    meta = {
        "dataset": Path(args.data).parent.name,
        "base": args.base,
        "epochs": args.epochs, "imgsz": args.imgsz,
        "freeze": args.freeze, "lr0": args.lr0,
        "all_splits": args.all_splits, "include_incremental": args.include_incremental,
        "finished_at": int(time.time()),
    }
    pc = RUNS / args.name / "per_class.json"
    if pc.exists():
        try:
            meta.update(json.loads(pc.read_text()))  # 并入最终 epoch 的 map50/map/per_class
        except Exception:
            pass
    (MODELS / (Path(args.out).name + ".meta.json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[ok] 训练溯源 → {MODELS / (Path(args.out).name + '.meta.json')}")
    print(f"[use] python live_client.py --unified {target} --style monitor")


if __name__ == "__main__":
    main()
