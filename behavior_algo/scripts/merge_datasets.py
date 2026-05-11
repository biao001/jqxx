"""
merge_datasets.py — 把 3 个 Roboflow 数据集合并成 1 个 8 类 YOLOv8 数据集

源：
  data/rf_datasets/distracted_driving  (5 类)
  data/rf_datasets/cigarette           (1 类)
  data/rf_datasets/seatbelt            (2 类)

目标类别表（8 类，覆盖 DMS 核心危险行为）：
  0: hand_on_wheel       (安全基线，反向推 hands_off_wheel)
  1: phone_use           (对应 Texting)
  2: calling             (对应 Calling)
  3: drinking            (distraction 扩展)
  4: reach_behind        (distraction 扩展)
  5: cigarette           (smoking)
  6: no_seatbelt
  7: seatbelt            (belt ok 基线)

输出目录: data/dms_unified/
  ├─ train/ images + labels
  ├─ valid/ images + labels
  └─ data.yaml
"""
import os
import shutil
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "rf_datasets"
DST = ROOT / "data" / "dms_unified"

# 新统一类别表
UNIFIED_NAMES = [
    "hand_on_wheel",   # 0
    "phone_use",       # 1 (texting)
    "calling",         # 2
    "drinking",        # 3
    "reach_behind",    # 4
    "cigarette",       # 5
    "no_seatbelt",     # 6
    "seatbelt",        # 7
]

# 各源数据集 → 统一 ID 的映射
#   按源数据集 data.yaml 里的 names 顺序（即其原始 class id）
MAPPING = {
    "distracted_driving": {
        0: 0,   # D0-Hand-on-Wheel → hand_on_wheel
        1: 1,   # D1-Texting       → phone_use
        2: 2,   # D2-Calling       → calling
        3: 3,   # D3-Drinking      → drinking
        4: 4,   # D4-Reach-Behind  → reach_behind
    },
    "cigarette": {
        0: 5,   # Cigarettesbutts → cigarette
    },
    "seatbelt": {
        0: 6,   # no-seatbelt → no_seatbelt
        1: 7,   # seatbelt    → seatbelt
    },
}


def remap_label_file(src_path: Path, dst_path: Path, mapping: dict):
    """读一个 YOLO txt 标签, 按 mapping 重映射类 id, 写到 dst_path"""
    lines_out = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            old_id = int(parts[0])
            if old_id not in mapping:
                # 不在 mapping 的类直接丢弃（不该出现，但防御）
                continue
            new_id = mapping[old_id]
            lines_out.append(
                f"{new_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))


def copy_split(src_ds_dir: Path, mapping: dict, prefix: str):
    """对单个数据集的 train/valid 两个 split 做处理"""
    counts = {"train": 0, "valid": 0}
    for split in ("train", "valid"):
        img_src = src_ds_dir / split / "images"
        lbl_src = src_ds_dir / split / "labels"
        if not img_src.exists():
            print(f"  [skip] {split} 不存在")
            continue
        img_dst = DST / split / "images"
        lbl_dst = DST / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img_path in img_src.iterdir():
            if not img_path.is_file():
                continue
            # 加前缀避免跨数据集重名
            new_name = f"{prefix}_{img_path.name}"
            shutil.copy(img_path, img_dst / new_name)

            # 标签改名（.txt 扩展名）
            lbl_name = img_path.stem + ".txt"
            lbl_path = lbl_src / lbl_name
            if lbl_path.exists():
                new_lbl_name = f"{prefix}_{lbl_name}"
                remap_label_file(lbl_path, lbl_dst / new_lbl_name, mapping)
            counts[split] += 1
    print(f"  {counts['train']} train  /  {counts['valid']} valid")


def main():
    if DST.exists():
        print(f"[!] 目标目录已存在 {DST}，先清空")
        shutil.rmtree(DST)

    DST.mkdir(parents=True, exist_ok=True)

    for sub, mapping in MAPPING.items():
        src_dir = SRC / sub
        if not src_dir.exists():
            print(f"[skip] {src_dir} 不存在")
            continue
        print(f"[merge] {sub}  mapping={mapping}")
        copy_split(src_dir, mapping, prefix=sub)

    # 写 data.yaml
    yaml_path = DST / "data.yaml"
    cfg = {
        "path": str(DST.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "nc": len(UNIFIED_NAMES),
        "names": UNIFIED_NAMES,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"\n[ok] 合并完成 → {DST}")
    print(f"    data.yaml: {yaml_path}")
    print(f"    类别: {UNIFIED_NAMES}")

    # 统计
    n_train_img = len(list((DST / "train" / "images").glob("*.*")))
    n_val_img   = len(list((DST / "valid" / "images").glob("*.*")))
    print(f"    训练集: {n_train_img} 张")
    print(f"    验证集: {n_val_img} 张")


if __name__ == "__main__":
    main()
