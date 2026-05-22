"""
balance_dataset.py — 对 data/dms_unified/train 按类下采样，缓解类别不均衡

只下采样"多数类"，且只删【纯多数类、不含任何受保护少数类】的图，
避免误删稀缺类样本。valid 不动（保证评估稳定）。

用法:
  python scripts/balance_dataset.py            # 用默认配额
  python scripts/balance_dataset.py --dry-run  # 只看会删多少，不动文件
"""
import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "dms_unified" / "train"
TRASH = ROOT / "data" / "dms_unified" / "_balance_trash"

NAMES = ["phone_use", "smoking", "drinking", "eating",
         "hand_on_wheel", "hands_off_wheel", "no_seatbelt", "seatbelt"]

# 多数类的 train 框数上限（超出则下采样）
MAX_BOXES = {
    "drinking": 1200,
    "seatbelt": 1200,
}
# 受保护的稀缺类：含这些类的图绝不删
PROTECT = {"phone_use", "hand_on_wheel", "no_seatbelt", "hands_off_wheel",
           "smoking", "eating"}


def img_classes(lbl: Path):
    cls = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            if line.strip():
                cls.append(int(line.split()[0]))
    return cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    img_dir = TRAIN / "images"
    lbl_dir = TRAIN / "labels"
    protect_ids = {NAMES.index(n) for n in PROTECT}
    rng = random.Random(42)

    # 收集每张图的类别
    items = []
    box_count = Counter()
    for img in sorted(img_dir.iterdir()):
        if not img.is_file():
            continue
        cls = img_classes(lbl_dir / (img.stem + ".txt"))
        for c in cls:
            box_count[c] += 1
        items.append((img, set(cls), cls))

    print("=== 下采样前 ===")
    for i, n in enumerate(NAMES):
        print(f"  {n:<16} {box_count[i]}")

    to_delete = []
    for cname, cap in MAX_BOXES.items():
        cid = NAMES.index(cname)
        # 候选：含该类、且不含任何受保护类的"可删图"
        cand = [it for it in items
                if cid in it[1] and not (it[1] & protect_ids) and it not in to_delete]
        rng.shuffle(cand)
        cur = box_count[cid]
        for it in cand:
            if cur <= cap:
                break
            # 删这张图，扣掉它贡献的该类框
            n_this = it[2].count(cid)
            cur -= n_this
            box_count[cid] -= n_this
            to_delete.append(it)

    print(f"\n将删除 {len(to_delete)} 张训练图"
          + ("（dry-run，不动文件）" if args.dry_run else ""))

    if not args.dry_run:
        for img, _, _ in to_delete:
            for src in (img, lbl_dir / (img.stem + ".txt")):
                if src.exists():
                    dst = TRASH / src.parent.name / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))

    # 重新统计
    final = Counter()
    for img in img_dir.iterdir():
        if img.is_file():
            for c in img_classes(lbl_dir / (img.stem + ".txt")):
                final[c] += 1
    total = sum(final.values()) or 1
    print("\n=== 下采样后 ===")
    for i, n in enumerate(NAMES):
        print(f"  {n:<16} {final[i]:>5}  ({100*final[i]/total:4.1f}%)")
    print(f"  剩余训练图: {len(list(img_dir.glob('*.*')))}")


if __name__ == "__main__":
    main()
