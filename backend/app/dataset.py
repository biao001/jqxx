"""数据集监测与上传：扫描 YOLO 数据集的类别分布与标注质量，支持上传自定义数据集。

用于模型持续优化时筛查训练数据(类别是否均衡、标注框是否异常)与管理自定义数据集。
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import yaml

# 框面积超过此比例视为"几乎整张图"的错标(与构建脚本 MAX_BOX_AREA 一致)
HUGE_AREA = 0.85
TINY_AREA = 0.003


def _read_names(ds_dir: Path) -> list[str]:
    y = ds_dir / "data.yaml"
    if y.exists():
        try:
            data = yaml.safe_load(y.read_text())
            names = data.get("names")
            if isinstance(names, dict):
                return [names[k] for k in sorted(names)]
            if isinstance(names, list):
                return names
        except Exception:
            pass
    return []


def scan_dataset(ds_dir: Path) -> dict[str, Any]:
    """扫描一个 YOLO 数据集目录，返回类别分布、各 split 数量、标注框尺寸质量审计。"""
    if not ds_dir.exists():
        return {"exists": False}
    names = _read_names(ds_dir)
    splits: dict[str, Any] = {}
    # 框面积分桶(占整图比例)：极小(<0.3%) / 正常 / 偏大(>50%，疑似错标)
    size_buckets = {"tiny": 0, "normal": 0, "huge": 0}
    total_cls: dict[int, int] = {}

    for split in BROWSE_SPLITS:
        lbl_dir = ds_dir / split / "labels"
        img_dir = ds_dir / split / "images"
        if not lbl_dir.exists():
            continue
        cls_count: dict[int, int] = {}
        n_img = sum(1 for _ in img_dir.glob("*")) if img_dir.exists() else 0
        n_lbl = 0
        n_ann = 0  # 含至少一个框的标注(非空)
        for f in lbl_dir.glob("*.txt"):
            n_lbl += 1
            has_box = False
            for line in f.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                has_box = True
                cid = int(p[0])
                cls_count[cid] = cls_count.get(cid, 0) + 1
                total_cls[cid] = total_cls.get(cid, 0) + 1
                area = float(p[3]) * float(p[4])
                if area < TINY_AREA:
                    size_buckets["tiny"] += 1
                elif area > HUGE_AREA:
                    size_buckets["huge"] += 1
                else:
                    size_buckets["normal"] += 1
            if has_box:
                n_ann += 1
        splits[split] = {
            "images": n_img,
            "labels": n_lbl,
            "annotated": n_ann,
            "classes": {(names[i] if i < len(names) else str(i)): c for i, c in sorted(cls_count.items())},
        }

    classes = [
        {"name": names[i] if i < len(names) else str(i), "count": c}
        for i, c in sorted(total_cls.items(), key=lambda kv: -kv[1])
    ]
    return {
        "exists": True,
        "names": names,
        "num_classes": len(names),
        "splits": splits,
        "classes": classes,
        "bbox_quality": size_buckets,
        "total_instances": sum(total_cls.values()),
    }


# 训练用的三个 split(写入 data.yaml、训练读取)
VALID_SPLITS = ("train", "valid", "test")
# "未划分"暂存区:新上传/切帧的图片先进这里，不参与训练，待自动划分后分配到上面三个
UNASSIGNED = "unassigned"
# 可浏览/可做文件操作的 split(含暂存区)
BROWSE_SPLITS = (UNASSIGNED, "train", "valid", "test")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def safe_split_name(split: str, name: str) -> bool:
    if split not in BROWSE_SPLITS:
        return False
    return bool(name) and "/" not in name and "\\" not in name and ".." not in name


def _scan_split_items(ds_dir: Path, sp: str, cls: int | None, issue: str | None) -> list[dict[str, Any]]:
    lbl_dir = ds_dir / sp / "labels"
    out: list[dict[str, Any]] = []
    if not lbl_dir.exists():
        return out
    for f in sorted(lbl_dir.glob("*.txt")):
        boxes: list[dict[str, Any]] = []
        cls_set: set[int] = set()
        tiny = huge = False
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cid = int(p[0])
            cx, cy, w, h = (float(x) for x in p[1:5])
            boxes.append({"cls": cid, "cx": cx, "cy": cy, "w": w, "h": h})
            cls_set.add(cid)
            area = w * h
            if area < TINY_AREA:
                tiny = True
            elif area > HUGE_AREA:
                huge = True
        if cls is not None and cls not in cls_set:
            continue
        if issue == "tiny" and not tiny:
            continue
        if issue == "huge" and not huge:
            continue
        if issue == "empty" and boxes:
            continue
        out.append({"name": f.stem, "split": sp, "boxes": boxes, "classes": sorted(cls_set), "tiny": tiny, "huge": huge})
    return out


def list_images(ds_dir: Path, split: str, cls: int | None, issue: str | None, page: int, page_size: int) -> dict[str, Any]:
    """逐张列出标注(归一化框)，支持按类别/质量筛选与分页。split='all' 时跨所有 split 聚合(每项带 split)。"""
    names = _read_names(ds_dir)
    splits = list(BROWSE_SPLITS) if split == "all" else [split]
    items: list[dict[str, Any]] = []
    for sp in splits:
        items.extend(_scan_split_items(ds_dir, sp, cls, issue))
    total = len(items)
    start = page * page_size
    return {"total": total, "page": page, "page_size": page_size, "names": names, "items": items[start : start + page_size]}


def image_path(ds_dir: Path, split: str, name: str) -> Path | None:
    img_dir = ds_dir / split / "images"
    for ext in IMG_EXT:
        p = img_dir / (name + ext)
        if p.exists():
            return p
    p = img_dir / name
    return p if p.exists() else None


def write_labels(ds_dir: Path, split: str, name: str, boxes: list[dict[str, Any]]) -> int:
    lbl = ds_dir / split / "labels" / (name + ".txt")
    lines = [
        f"{int(b['cls'])} {float(b['cx']):.6f} {float(b['cy']):.6f} {float(b['w']):.6f} {float(b['h']):.6f}"
        for b in boxes
        if all(k in b for k in ("cls", "cx", "cy", "w", "h"))
    ]
    lbl.write_text("\n".join(lines))
    return len(lines)


def delete_image(ds_dir: Path, split: str, name: str) -> bool:
    lbl = ds_dir / split / "labels" / (name + ".txt")
    img = image_path(ds_dir, split, name)
    removed = False
    if lbl.exists():
        lbl.unlink()
        removed = True
    if img and (img.exists() or img.is_symlink()):
        img.unlink()
        removed = True
    return removed


def _next_seq(img_dir: Path, prefix: str) -> int:
    """扫描已有 <prefix>_NNNNN.* 文件，返回下一个序号(从已用最大值+1，避开删除留下的空洞冲突)。"""
    mx = 0
    if img_dir.exists():
        for p in img_dir.iterdir():
            if p.stem.startswith(prefix):
                tail = p.stem[len(prefix):]
                if tail.isdigit():
                    mx = max(mx, int(tail))
    return mx + 1


def move_image(ds_dir: Path, split: str, name: str, to_split: str) -> bool:
    """把图片及其标注从一个 split 移动到另一个 split(用于重分训练/验证/测试集)。"""
    if to_split not in BROWSE_SPLITS or split == to_split:
        return False
    src_img = image_path(ds_dir, split, name)
    if src_img is None:
        return False
    dst_img_dir = ds_dir / to_split / "images"
    dst_lbl_dir = ds_dir / to_split / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    src_img.replace(dst_img_dir / src_img.name)
    src_lbl = ds_dir / split / "labels" / (name + ".txt")
    if src_lbl.exists():
        src_lbl.replace(dst_lbl_dir / (name + ".txt"))
    return True


def redistribute_splits(ds_dir: Path, val_frac: float, test_frac: float) -> dict[str, Any]:
    """把所有图片(连标注)随机重新划分到 train/valid/test。val_frac/test_frac 为比例(0~1)。"""
    import random

    val_frac = max(0.0, min(0.9, val_frac))
    test_frac = max(0.0, min(0.9, test_frac))
    if val_frac + test_frac >= 1.0:
        raise ValueError("valid + test 比例必须小于 1")
    all_imgs: list[tuple[str, str]] = []  # (current_split, name)
    for s in BROWSE_SPLITS:  # 含未划分暂存区，一并分配到 train/valid/test
        d = ds_dir / s / "images"
        if d.exists():
            for p in d.iterdir():
                if p.suffix.lower() in IMG_EXT:
                    all_imgs.append((s, p.stem))
    n = len(all_imgs)
    random.shuffle(all_imgs)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    targets = ["valid"] * n_val + ["test"] * n_test + ["train"] * (n - n_val - n_test)
    moved = 0
    for (cur, name), tgt in zip(all_imgs, targets):
        if cur != tgt and move_image(ds_dir, cur, name, tgt):
            moved += 1
    counts = {s: sum(1 for _ in (ds_dir / s / "images").iterdir()) if (ds_dir / s / "images").exists() else 0 for s in VALID_SPLITS}
    return {"total": n, "moved": moved, "splits": counts}


def add_single_image(ds_dir: Path, split: str, filename: str, content: bytes) -> str:
    """上传单张图片到指定 split，创建空标注(待画框)。

    统一命名为 <数据集名>_<split>_<5位序号>，如 realtime_demo_train_00001，
    便于排序、追溯来源、避免随机名难以管理。返回名称(stem)。
    """
    img_dir = ds_dir / split / "images"
    lbl_dir = ds_dir / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(filename or "").suffix.lower()
    if ext not in IMG_EXT:
        ext = ".jpg"
    prefix = f"{ds_dir.name}_{split}_"
    name = f"{prefix}{_next_seq(img_dir, prefix):05d}"
    (img_dir / (name + ext)).write_bytes(content)
    (lbl_dir / (name + ".txt")).write_text("")
    return name


def save_uploaded_dataset(zip_bytes: bytes, dest_dir: Path) -> dict[str, Any]:
    """解压上传的数据集 zip 到 dest_dir，统计图片/标注数。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    imgs = lbls = 0
    img_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            # 防目录穿越
            if name.startswith("/") or ".." in Path(name).parts:
                continue
            suffix = Path(name).suffix.lower()
            if suffix in img_ext:
                imgs += 1
            elif suffix == ".txt":
                lbls += 1
            elif suffix not in (".yaml", ".yml"):
                continue
            target = dest_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
    return {"images": imgs, "labels": lbls, "dir": dest_dir.name}
