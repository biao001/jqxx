"""
build_reckless_mapped.py — 以单一来源 reckless 为主干，构建产品语义数据集

设计原则(规避旧 merge 的质量塌方):
  * 主干只用 reckless(单一来源、标注一致，托管模型 cigarette 可达 0.84~0.90)
  * 去掉空标注(背景)图 —— 旧数据 55% 是背景，稀释训练
  * phone_use 零污染补充：reckless 手机样本少(174)，从 abnormal / dms_sny
    只挑「唯一目标行为是手机」的图、且仅保留手机标注。这样既不漏标其它行为
    (会教坏模型)，也不把这两个低质数据集的 cigarette/seatbelt 等混进来。

reckless 9 类 → 产品 7 类(no_seatbelt 由检测器逻辑推断，不训):
  Cigarette->smoking  Drinking->drinking  Eating->eating
  HandsNotOnWheel->hands_off_wheel  HandsOnWheel->hand_on_wheel
  Phone->phone_use  Seatbelt->seatbelt   Sleepy/microsleep->丢弃

图片用符号链接，不复制。输出 datasets/reckless_mapped/。
"""
from pathlib import Path
from collections import Counter
import yaml

ROOT = Path(__file__).resolve().parent.parent
RF = ROOT.parent / "datasets" / "rf_datasets"
DST = ROOT.parent / "datasets" / "reckless_mapped"

NAMES = ["phone_use", "smoking", "drinking", "eating",
         "hand_on_wheel", "hands_off_wheel", "seatbelt"]
NAME2ID = {n: i for i, n in enumerate(NAMES)}
PHONE_ID = NAME2ID["phone_use"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# 标注框面积(占整图比例)超过此值视为错标(几乎整张图)，构建时丢弃
MAX_BOX_AREA = 0.85

# 主干 reckless: 原 id -> 目标类名(不在表中的丢弃)
RECKLESS_MAP = {
    0: "smoking", 1: "drinking", 2: "eating", 3: "hands_off_wheel",
    4: "hand_on_wheel", 5: "phone_use", 6: "seatbelt",
}

# 补充集：源类名(小写) -> 我们的「目标行为」类名。仅这些算目标行为；
# 其它(Distracted/SafeDriving/Drowsy 等)是非目标，判定共现时忽略。
SUPP_TARGET = {
    "cigarette": "smoking", "smoking": "smoking",
    "drinking": "drinking", "eating": "eating",
    "phone": "phone_use", "mobile use": "phone_use",
    "seatbelt": "seatbelt",
}


def _link_and_write(img: Path, lines, dst_img: Path, dst_lbl: Path):
    link = dst_img / img.name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(img.resolve())
    (dst_lbl / (img.stem + ".txt")).write_text("\n".join(lines))


def build_reckless(cls_cnt):
    """主干：全类映射，丢弃空标注图。"""
    src = RF / "reckless"
    stats = {}
    for split in ["train", "valid", "test"]:
        si, sl = src / split / "images", src / split / "labels"
        if not si.exists():
            continue
        di, dl = DST / split / "images", DST / split / "labels"
        di.mkdir(parents=True, exist_ok=True)
        dl.mkdir(parents=True, exist_ok=True)
        kept = dropped = 0
        for img in si.iterdir():
            if img.suffix.lower() not in IMG_EXT:
                continue
            lbl = sl / (img.stem + ".txt")
            out = []
            if lbl.exists():
                for ln in lbl.read_text().splitlines():
                    p = ln.split()
                    if len(p) < 5:
                        continue
                    old = int(p[0])
                    if old not in RECKLESS_MAP:
                        continue
                    if float(p[3]) * float(p[4]) > MAX_BOX_AREA:  # 丢弃几乎整张图的错标
                        continue
                    nid = NAME2ID[RECKLESS_MAP[old]]
                    out.append(" ".join([str(nid)] + p[1:]))
                    cls_cnt[split][nid] += 1
            if not out:        # 空标注图 → 丢弃
                dropped += 1
                continue
            _link_and_write(img, out, di, dl)
            kept += 1
        stats[split] = (kept, dropped)
    return stats


def supplement_phone(ds_name, cls_cnt):
    """从补充集只收『唯一目标行为是手机』的图，仅保留手机标注。"""
    src = RF / ds_name
    names = [n.lower() for n in yaml.safe_load(open(src / "data.yaml"))["names"]]
    added = {}
    for split in ["train", "valid", "test"]:
        si, sl = src / split / "images", src / split / "labels"
        if not si.exists():
            continue
        di, dl = DST / split / "images", DST / split / "labels"
        di.mkdir(parents=True, exist_ok=True)
        dl.mkdir(parents=True, exist_ok=True)
        n = 0
        for img in si.iterdir():
            if img.suffix.lower() not in IMG_EXT:
                continue
            lbl = sl / (img.stem + ".txt")
            if not lbl.exists():
                continue
            phone_lines, targets = [], set()
            for ln in lbl.read_text().splitlines():
                p = ln.split()
                if len(p) < 5:
                    continue
                src_name = names[int(p[0])]
                tgt = SUPP_TARGET.get(src_name)
                if tgt is None:
                    continue                      # 非目标行为(Distracted 等)忽略
                targets.add(tgt)
                if tgt == "phone_use" and float(p[3]) * float(p[4]) <= MAX_BOX_AREA:
                    phone_lines.append(" ".join([str(PHONE_ID)] + p[1:]))
            # 仅当唯一目标行为是手机时收录(避免漏标其它行为)
            if targets == {"phone_use"} and phone_lines:
                # 加数据集前缀，避免与 reckless 文件名冲突
                link = di / f"{ds_name}_{img.name}"
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(img.resolve())
                (dl / f"{ds_name}_{img.stem}.txt").write_text("\n".join(phone_lines))
                cls_cnt[split][PHONE_ID] += len(phone_lines)
                n += 1
        added[split] = n
    return added


def main():
    if not (RF / "reckless").exists():
        raise SystemExit(f"[!] 缺少 {RF/'reckless'}")
    DST.mkdir(parents=True, exist_ok=True)
    cls_cnt = {s: Counter() for s in ["train", "valid", "test"]}

    rk = build_reckless(cls_cnt)
    supp = {ds: supplement_phone(ds, cls_cnt) for ds in ["abnormal", "dms_sny"]
            if (RF / ds).exists()}

    yaml_text = (
        f"path: {DST}\ntrain: train/images\nval: valid/images\n"
        + ("test: test/images\n" if (DST / "test" / "images").exists() else "")
        + f"nc: {len(NAMES)}\nnames: {NAMES}\n"
    )
    (DST / "data.yaml").write_text(yaml_text)

    print(f"[ok] 输出 {DST}\n[ok] 类别({len(NAMES)}): {NAMES}\n")
    print("reckless 主干(已丢弃空标注图):")
    for s, (k, d) in rk.items():
        print(f"  [{s}] 保留 {k}  丢弃空图 {d}")
    print("phone 补充(仅唯一目标=手机的图):")
    for ds, a in supp.items():
        print(f"  {ds}: {a}")
    print("\n最终各 split 类别实例数:")
    for s in ["train", "valid", "test"]:
        if not cls_cnt[s]:
            continue
        print(f"  [{s}] " + "  ".join(
            f"{NAMES[i]}={cls_cnt[s].get(i,0)}" for i in range(len(NAMES))))
    print("\n[next] python scripts/train_unified.py --data datasets/reckless_mapped/data.yaml --device 0 --epochs 50")


if __name__ == "__main__":
    main()
