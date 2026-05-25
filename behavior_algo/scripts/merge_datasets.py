"""
merge_datasets.py — 把多个 Roboflow 数据集按"类别名"映射合并成 1 个 8 类 YOLOv8 数据集

按名字而非 id 匹配，所以源数据集即使重新排列类别也不会错位。
未在 NAME_MAPPING 中列出的类直接丢弃。

源: datasets/rf_datasets/<subdir>/{train,valid,test}/{images,labels}
目标: datasets/dms_unified/{train,valid}/{images,labels} + data.yaml

unified 类别表 (8 类):
  0 hand_on_wheel
  1 phone_use      (texting / using phone screen)
  2 calling        (talking on the phone)
  3 drinking
  4 reach_behind
  5 cigarette      (smoking)
  6 no_seatbelt
  7 seatbelt
"""
import random
import re
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT.parent / "datasets" / "rf_datasets"
DST = ROOT.parent / "datasets" / "dms_unified"


UNIFIED_NAMES = [
    "phone_use",        # 0
    "smoking",          # 1
    "drinking",         # 2
    "eating",           # 3
    "hand_on_wheel",    # 4 (安全基线)
    "hands_off_wheel",  # 5 (告警)
    "no_seatbelt",      # 6
    "seatbelt",         # 7 (安全基线)
]
UNIFIED_INDEX = {n: i for i, n in enumerate(UNIFIED_NAMES)}


def _norm(name: str) -> str:
    """统一类名格式：去 Roboflow 'D0-' / 'c0 - ' 这种前缀；空格/连字符/下划线归一；小写"""
    s = name.strip().lower()
    s = re.sub(r"^[a-z]?\d+[\s\-_]*", "", s)
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# 每个数据集子目录 → 源类名(归一化) → 目标 unified 类名
NAME_MAPPING = {
    "reckless": {
        # ['Cigarette','Eating','microsleep','Sleepy','HandsOnWheel',
        #  'HandsNotOnWheel','Phone','Drinking','Seatbelt']
        "phone":            "phone_use",
        "cigarette":        "smoking",
        "drinking":         "drinking",
        "eating":           "eating",
        "handsonwheel":     "hand_on_wheel",
        "handsnotonwheel":  "hands_off_wheel",
        "seatbelt":         "seatbelt",
        # microsleep / sleepy 不映射 → 丢弃（疲劳归 fatigue 模块）
    },
    "seatbelt": {
        # ['no-seatbelt', 'seatbelt']
        "no seatbelt":  "no_seatbelt",
        "seatbelt":     "seatbelt",
    },
    # 下面两个仅用于补 phone_use 样本（reckless 的 Phone 标注太少）。
    # 只映射手机类，其他类全部丢弃，避免再引入多源标注冲突。
    "abnormal": {
        "phone":        "phone_use",
    },
    "dms_sny": {
        "mobile use":   "phone_use",
    },
}

# 每个数据集子目录最多取多少张训练图（None=全取）。用于抑制 seatbelt 这类
# 体量过大的来源，避免类别失衡导致模型过度学习某一标签。
MAX_TRAIN_PER_DATASET = {
    "seatbelt": 2500,
}


def load_source_classes(ds_dir: Path):
    yml = ds_dir / "data.yaml"
    if not yml.exists():
        return []
    cfg = yaml.safe_load(yml.read_text(encoding="utf-8"))
    names = cfg.get("names", [])
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return list(names)


def build_id_map(src_classes, name_map):
    """返回 src_class_id → unified_class_id；未列出的类不进结果"""
    out = {}
    for src_id, cls_name in enumerate(src_classes):
        key = _norm(cls_name)
        target = name_map.get(key)
        if target is None:
            continue
        if target not in UNIFIED_INDEX:
            raise ValueError(f"目标类不在 UNIFIED_NAMES: {target}")
        out[src_id] = UNIFIED_INDEX[target]
    return out


def remap_lines(src_path: Path, id_map):
    """读 label，按 id_map 重映射，返回有效行列表（不含丢弃类）"""
    lines = []
    if not src_path.exists():
        return lines
    for line in src_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        old_id = int(parts[0])
        if old_id not in id_map:
            continue
        lines.append(f"{id_map[old_id]} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
    return lines


def copy_split(ds_dir: Path, id_map, prefix: str, max_train=None):
    counts = {"train": 0, "valid": 0}
    for split in ("train", "valid"):
        img_src = ds_dir / split / "images"
        lbl_src = ds_dir / split / "labels"
        if not img_src.exists():
            continue
        img_dst = DST / split / "images"
        lbl_dst = DST / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        imgs = sorted(p for p in img_src.iterdir() if p.is_file())
        # 仅对 train 做上限抽样（valid 全保留，保证评估稳定）
        if split == "train" and max_train and len(imgs) > max_train:
            random.Random(42).shuffle(imgs)
            imgs = imgs[:max_train]

        for img_path in imgs:
            lines = remap_lines(lbl_src / (img_path.stem + ".txt"), id_map)
            if not lines:
                continue  # 跳过映射后无目标类的空图，避免大量背景图稀释
            new_stem = f"{prefix}_{img_path.stem}"
            shutil.copy(img_path, img_dst / f"{prefix}_{img_path.name}")
            (lbl_dst / f"{new_stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            counts[split] += 1
    print(f"  {counts['train']} train  /  {counts['valid']} valid"
          + (f"  (train 上限 {max_train})" if max_train else ""))


def main():
    if DST.exists():
        print(f"[!] 目标 {DST} 已存在，清空")
        shutil.rmtree(DST)
    DST.mkdir(parents=True, exist_ok=True)

    for sub, name_map in NAME_MAPPING.items():
        ds_dir = SRC / sub
        if not ds_dir.exists():
            print(f"[skip] {ds_dir} 不存在")
            continue
        src_classes = load_source_classes(ds_dir)
        id_map = build_id_map(src_classes, name_map)
        print(f"[merge] {sub}")
        print(f"        src classes ({len(src_classes)}): {src_classes}")
        print(f"        id_map: {id_map}  → {[UNIFIED_NAMES[v] for v in id_map.values()]}")
        copy_split(ds_dir, id_map, prefix=sub,
                   max_train=MAX_TRAIN_PER_DATASET.get(sub))

    yaml_path = DST / "data.yaml"
    cfg = {
        "path":  str(DST.resolve()),
        "train": "train/images",
        "val":   "valid/images",
        "nc":    len(UNIFIED_NAMES),
        "names": UNIFIED_NAMES,
    }
    yaml_path.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True),
                          encoding="utf-8")

    print(f"\n[ok] 合并完成 → {DST}")
    print(f"     训练集: {len(list((DST/'train'/'images').glob('*.*')))} 张")
    print(f"     验证集: {len(list((DST/'valid'/'images').glob('*.*')))} 张")


if __name__ == "__main__":
    main()
