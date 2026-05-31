"""数据集监测与上传：扫描 YOLO 数据集的类别分布与标注质量，支持上传自定义数据集。

用于模型持续优化时筛查训练数据(类别是否均衡、标注框是否异常)与管理自定义数据集。
"""
from __future__ import annotations

import io
import json
import math
import random
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
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
        {"cls": i, "name": names[i] if i < len(names) else str(i), "count": c}
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
    # 目标已存在同名文件 → 改用目标 split 的下一个序号，避免 replace() 静默覆盖造成数据丢失
    dst_name = name
    if (dst_img_dir / src_img.name).exists():
        pre = f"{ds_dir.name}_{to_split}_"
        dst_name = f"{pre}{_next_seq(dst_img_dir, pre):05d}"
    src_img.replace(dst_img_dir / (dst_name + src_img.suffix))
    src_lbl = ds_dir / split / "labels" / (name + ".txt")
    if src_lbl.exists():
        src_lbl.replace(dst_lbl_dir / (dst_name + ".txt"))
    return True


def merge_labeled(src_dir: Path, dst_dir: Path) -> dict[str, Any]:
    """把 src 数据集中「有非空标注」的图(连标注)复制并入 dst 的同名 split(train→train…)。

    - 只并入有非空标注的图：未标注图会被 YOLO 当背景负样本压制目标类，故跳过。
    - copy 不 move：源数据集保持完整，可继续独立使用。
    - 同名已存在则跳过：可重复执行不重复并入(幂等)；并入文件名带源数据集前缀，便于日后撤销(删除即可)。
    用于「把增量/演示采集的数据并进现役训练集」做防遗忘的混合微调。
    """
    import shutil

    added = {s: 0 for s in VALID_SPLITS}
    skipped_unlabeled = 0
    for sp in VALID_SPLITS:
        s_img = src_dir / sp / "images"
        s_lbl = src_dir / sp / "labels"
        if not s_img.exists():
            continue
        d_img = dst_dir / sp / "images"
        d_lbl = dst_dir / sp / "labels"
        d_img.mkdir(parents=True, exist_ok=True)
        d_lbl.mkdir(parents=True, exist_ok=True)
        for img in sorted(s_img.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            lbl = s_lbl / (img.stem + ".txt")
            if not (lbl.exists() and lbl.read_text().strip()):
                skipped_unlabeled += 1
                continue
            dst_img = d_img / img.name
            if dst_img.exists():
                continue  # 已并入，跳过(幂等)
            shutil.copy2(img, dst_img)
            shutil.copy2(lbl, d_lbl / (img.stem + ".txt"))
            added[sp] += 1
    return {"added": added, "total": sum(added.values()), "skipped_unlabeled": skipped_unlabeled}


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


# ---------- 感知去重(dHash) ----------
# 入库前用感知哈希拦掉与已有图「几乎一样」的图(视频相邻帧/实时连拍最常见)，
# 避免重复上传+重复标注浪费人力。精确同一文件也会被一并判重。

DHASH_CACHE = ".dhash.json"   # 数据集根下的指纹缓存(键=<split>/<文件名>)，避免每次全量重算
DUP_HAMMING_DIST = 5          # 64 位 dHash 汉明距离阈值：<= 即判近重复(越小越严)


def image_dhash(content: bytes) -> int | None:
    """64 位 dHash 感知哈希：缩成 9×8 灰度，比较每行相邻像素。解码失败返回 None。"""
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    diff = small[:, 1:] > small[:, :-1]   # 8×8 bool
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(v)
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class DedupIndex:
    """数据集级感知去重索引：维护跨 split 的 dHash 指纹，判定新图是否与已入库图近重复。

    用 .dhash.json 缓存避免每次全量重算；构建时按磁盘实际文件增量补算、剔除已删项。
    批量入库时新指纹先记到内存，最后 flush() 落盘。线程内单次请求使用，不做并发保护。
    """

    def __init__(self, ds_dir: Path, max_dist: int = DUP_HAMMING_DIST):
        self.ds_dir = ds_dir
        self.max_dist = max(0, int(max_dist))
        self._by_key: dict[str, int] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        cache: dict[str, int] = {}
        cache_path = self.ds_dir / DHASH_CACHE
        if cache_path.exists():
            try:
                cache = {k: int(v) for k, v in json.loads(cache_path.read_text()).items()}
            except Exception:
                cache = {}
        current: dict[str, int] = {}
        for sp in BROWSE_SPLITS:
            img_dir = self.ds_dir / sp / "images"
            if not img_dir.exists():
                continue
            for p in img_dir.iterdir():
                if p.suffix.lower() not in IMG_EXT:
                    continue
                key = f"{sp}/{p.name}"
                h = cache.get(key)
                if h is None:
                    self._dirty = True  # 缓存缺失 → 需重算并回写
                    try:
                        h = image_dhash(p.read_bytes())
                    except Exception:
                        h = None
                    if h is None:
                        continue
                current[key] = h
        if len(current) != len(cache):
            self._dirty = True  # 有图被删/键不一致 → 回写收敛
        self._by_key = current

    def is_dup(self, h: int | None) -> bool:
        if h is None:
            return False
        return any(_hamming(h, e) <= self.max_dist for e in self._by_key.values())

    def register(self, split: str, filename: str, h: int | None) -> None:
        if h is not None:
            self._by_key[f"{split}/{filename}"] = h
            self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        try:
            (self.ds_dir / DHASH_CACHE).write_text(
                json.dumps({k: str(v) for k, v in self._by_key.items()})
            )
            self._dirty = False
        except Exception:
            pass


def add_single_image(ds_dir: Path, split: str, filename: str, content: bytes,
                     dedup: "DedupIndex | None" = None) -> str | None:
    """上传单张图片到指定 split，创建空标注(待画框)。

    统一命名为 <数据集名>_<split>_<5位序号>，如 realtime_demo_train_00001，
    便于排序、追溯来源、避免随机名难以管理。返回名称(stem)。
    dedup 给定时：先算感知哈希，若与库内近重复则跳过(不写文件，返回 None)；
    否则写入并把新指纹登记进索引(批量结束后由调用方 flush() 落盘)。
    """
    h = None
    if dedup is not None:
        h = image_dhash(content)
        if dedup.is_dup(h):
            return None
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
    if dedup is not None:
        dedup.register(split, name + ext, h)
    return name


# ---------- 离线数据增强 ----------
# 对已标注图生成旋转/翻转/噪声/亮度/模糊变体并正确变换 bbox，写回 train 扩充训练集。
# 解决基础样本过少(如 realtime_demo 仅 137 框)的问题——在线增强多样性受限于原始基数，
# 离线增强把扩充后的样本固化进数据集。增强样本统一命名带 _aug_ 前缀，便于一键清除。

AUG_PREFIX_TAG = "aug"  # 文件名标记：<dataset>_aug_<NNNNN>，用于识别/清除增强样本
AUG_OPS = ("rotate", "flip", "noise", "brightness", "blur")


def _read_yolo_boxes(lbl_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            c = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            continue
        boxes.append((c, cx, cy, w, h))
    return boxes


def _rotate_box(box, angle_deg: float, W: int, H: int):
    """绕图像中心旋转后，重算 bbox 的轴对齐外接框(归一化)。出框过多则丢弃返回 None。"""
    c, cx, cy, w, h = box
    px, py = cx * W, cy * H
    bw, bh = w * W, h * H
    corners = [(px - bw / 2, py - bh / 2), (px + bw / 2, py - bh / 2),
               (px + bw / 2, py + bh / 2), (px - bw / 2, py + bh / 2)]
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    ox, oy = W / 2, H / 2
    xs, ys = [], []
    for x, y in corners:
        dx, dy = x - ox, y - oy
        rx = ox + dx * cos_a - dy * sin_a
        ry = oy + dx * sin_a + dy * cos_a
        xs.append(rx); ys.append(ry)
    nx1, nx2 = max(0.0, min(xs)), min(float(W), max(xs))
    ny1, ny2 = max(0.0, min(ys)), min(float(H), max(ys))
    if nx2 - nx1 < 2 or ny2 - ny1 < 2:
        return None
    ncx = (nx1 + nx2) / 2 / W
    ncy = (ny1 + ny2) / 2 / H
    nw = (nx2 - nx1) / W
    nh = (ny2 - ny1) / H
    return (c, ncx, ncy, nw, nh)


def _motion_blur(img: np.ndarray, ksize: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((ksize, ksize), dtype=np.float32)
    k[ksize // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (ksize, ksize))
    s = k.sum()
    if s > 0:
        k /= s
    return cv2.filter2D(img, -1, k)


# 各操作强度的默认上限（前端可逐项调节）
AUG_INTENSITY_DEFAULTS = {
    "max_deg": 15.0,     # 旋转最大角度 ±
    "noise_sigma": 18.0, # 高斯噪声最大 σ
    "brightness": 30.0,  # 亮度偏移最大 |β|(像素)，对比度按比例联动
    "blur_ksize": 9,     # 运动模糊最大核(取 3..ksize 的奇数)
}


def _norm_intensity(intensity: dict[str, Any] | None) -> dict[str, float]:
    it = dict(AUG_INTENSITY_DEFAULTS)
    if intensity:
        for k in it:
            if k in intensity and intensity[k] is not None:
                try:
                    it[k] = float(intensity[k])
                except (TypeError, ValueError):
                    pass
    it["max_deg"] = max(0.0, min(45.0, it["max_deg"]))
    it["noise_sigma"] = max(0.0, min(60.0, it["noise_sigma"]))
    it["brightness"] = max(0.0, min(80.0, it["brightness"]))
    it["blur_ksize"] = int(max(3, min(21, it["blur_ksize"])))
    return it


def _apply_aug(img: np.ndarray, boxes, ops: list[str], intensity: dict[str, float], rng: random.Random):
    """对一张图随机施加选中的增强组合，返回 (增强图, 变换后 boxes)。intensity 控制各操作强度。"""
    H, W = img.shape[:2]
    out = img
    new_boxes = list(boxes)

    max_deg = intensity["max_deg"]
    if "rotate" in ops and max_deg > 0:
        angle = rng.uniform(-max_deg, max_deg)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        out = cv2.warpAffine(out, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)
        rotated = [_rotate_box(b, angle, W, H) for b in new_boxes]
        new_boxes = [b for b in rotated if b is not None]

    if "flip" in ops and rng.random() < 0.5:
        out = cv2.flip(out, 1)
        new_boxes = [(c, 1.0 - cx, cy, w, h) for (c, cx, cy, w, h) in new_boxes]

    if "brightness" in ops:
        b = intensity["brightness"]
        alpha = rng.uniform(1.0 - b / 120.0, 1.0 + b / 120.0)  # 对比度随亮度幅度联动
        beta = rng.uniform(-b, b)                              # 亮度偏移
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if "blur" in ops and rng.random() < 0.6:
        kmax = int(intensity["blur_ksize"])
        odds = [k for k in range(3, kmax + 1, 2)] or [3]
        ksize = rng.choice(odds)
        out = _motion_blur(out, ksize, rng.uniform(0, 180))

    if "noise" in ops:
        smax = intensity["noise_sigma"]
        if smax > 0:
            sigma = rng.uniform(smax * 0.4, smax)
            gauss = np.random.normal(0, sigma, out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

    return out, new_boxes


def _collect_sources(ds_dir: Path, source_split: str, classes: set[int] | None):
    """收集 source_split 中含有效框的原图(排除增强样本)。

    classes 非空时只保留"含至少一个目标类框"的图。返回 [(img_path, boxes), ...]。
    """
    src_img_dir = ds_dir / source_split / "images"
    src_lbl_dir = ds_dir / source_split / "labels"
    items: list[tuple[Path, list]] = []
    if not src_img_dir.exists():
        return items
    for img_path in sorted(src_img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXT:
            continue
        if f"_{AUG_PREFIX_TAG}_" in img_path.stem:
            continue  # 不对增强样本再增强，避免雪崩
        boxes = _read_yolo_boxes(src_lbl_dir / (img_path.stem + ".txt"))
        if not boxes:
            continue
        if classes is not None and not any(b[0] in classes for b in boxes):
            continue
        items.append((img_path, boxes))
    return items


def _train_class_counts(ds_dir: Path) -> dict[int, int]:
    """统计 train 当前各类实例(框)数，用于均衡到目标。"""
    counts: dict[int, int] = {}
    lbl_dir = ds_dir / "train" / "labels"
    if lbl_dir.exists():
        for f in lbl_dir.glob("*.txt"):
            for b in _read_yolo_boxes(f):
                counts[b[0]] = counts.get(b[0], 0) + 1
    return counts


def augment_dataset(ds_dir: Path, ops: list[str], factor: int = 2,
                    max_deg: float = 15.0, source_split: str = "train",
                    classes: list[int] | None = None,
                    target_per_class: int | None = None,
                    intensity: dict[str, Any] | None = None,
                    seed: int | None = None) -> dict[str, Any]:
    """离线增强，样本固定写入 train（valid/test 保持纯净，避免泄漏导致评估虚高）。

    两种模式：
    - 固定份数(默认)：对每张符合条件的原图生成 factor 份。
    - 均衡到目标(target_per_class>0)：对每个目标类，从含该类的原图反复生成，
      直到该类在 train 的实例数达到 target_per_class（自动纠正类别不均衡）。
    classes 非空时：只用"含目标类"的原图作增强源（均衡模式只补这些类）。
    intensity 逐操作强度见 AUG_INTENSITY_DEFAULTS。命名 <dataset>_aug_<NNNNN>。
    """
    ops = [o for o in ops if o in AUG_OPS]
    if not ops:
        raise ValueError("未选择任何有效增强操作")
    if source_split not in BROWSE_SPLITS:
        raise ValueError("非法 source_split")
    factor = max(1, min(10, int(factor)))
    it = _norm_intensity(intensity)
    cls_set = {int(c) for c in classes} if classes else None
    # 兼容旧入参：max_deg 显式传入且 intensity 未带 max_deg 时沿用
    if intensity is None or "max_deg" not in (intensity or {}):
        it["max_deg"] = max(0.0, min(45.0, float(max_deg)))

    items = _collect_sources(ds_dir, source_split, cls_set)
    out_img_dir = ds_dir / "train" / "images"
    out_lbl_dir = ds_dir / "train" / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    prefix = f"{ds_dir.name}_{AUG_PREFIX_TAG}_"
    seq = _next_seq(out_img_dir, prefix)

    def _write_one(img, boxes) -> list | None:
        nonlocal seq
        aug_img, aug_boxes = _apply_aug(img, boxes, ops, it, rng)
        if not aug_boxes:
            return None
        name = f"{prefix}{seq:05d}"
        seq += 1
        ok, buf = cv2.imencode(".jpg", aug_img)
        if not ok:
            return None
        buf.tofile(str(out_img_dir / (name + ".jpg")))
        lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for (c, cx, cy, w, h) in aug_boxes]
        (out_lbl_dir / (name + ".txt")).write_text("\n".join(lines))
        return aug_boxes

    generated = 0

    if target_per_class and int(target_per_class) > 0:
        # ---- 均衡到目标 ----
        target = int(target_per_class)
        counts = _train_class_counts(ds_dir)
        target_classes = cls_set if cls_set is not None else set(counts.keys())
        by_class: dict[int, list] = {}
        for path, boxes in items:
            present = {b[0] for b in boxes}
            for c in present:
                if c in target_classes:
                    by_class.setdefault(c, []).append((path, boxes))
        img_cache: dict[str, np.ndarray] = {}
        per_class_added: dict[int, int] = {}
        SAFETY = target * 50 + 200  # 防失控上限
        for c in sorted(target_classes):
            pool = by_class.get(c, [])
            if not pool:
                continue
            tries = 0
            while counts.get(c, 0) < target and tries < SAFETY:
                tries += 1
                path, boxes = pool[rng.randrange(len(pool))]
                key = str(path)
                if key not in img_cache:
                    img_cache[key] = cv2.imdecode(np.fromfile(key, dtype=np.uint8), cv2.IMREAD_COLOR)
                img = img_cache[key]
                if img is None:
                    break
                aug_boxes = _write_one(img, boxes)
                if aug_boxes is None:
                    continue
                generated += 1
                per_class_added[c] = per_class_added.get(c, 0) + 1
                for b in aug_boxes:  # 一张图的所有类都计数
                    counts[b[0]] = counts.get(b[0], 0) + 1
        return {"generated": generated, "sources": len(items), "ops": ops,
                "mode": "balance", "target_per_class": target,
                "per_class_added": {str(k): v for k, v in per_class_added.items()},
                "intensity": it}

    # ---- 固定份数 ----
    for path, boxes in items:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        for _ in range(factor):
            if _write_one(img, boxes) is not None:
                generated += 1
    return {"generated": generated, "sources": len(items), "ops": ops,
            "mode": "fixed", "factor": factor, "intensity": it}


def preview_augment(ds_dir: Path, ops: list[str], max_deg: float = 15.0,
                    source_split: str = "train", classes: list[int] | None = None,
                    intensity: dict[str, Any] | None = None,
                    count: int = 4, seed: int | None = None) -> dict[str, Any]:
    """生成 count 张增强预览(带框绘制)，返回 base64 dataURL 列表，不写入数据集。"""
    import base64
    ops = [o for o in ops if o in AUG_OPS]
    if not ops:
        raise ValueError("未选择任何有效增强操作")
    it = _norm_intensity(intensity)
    if intensity is None or "max_deg" not in (intensity or {}):
        it["max_deg"] = max(0.0, min(45.0, float(max_deg)))
    cls_set = {int(c) for c in classes} if classes else None
    items = _collect_sources(ds_dir, source_split, cls_set)
    if not items:
        return {"previews": [], "note": "没有符合条件的有标注原图"}
    rng = random.Random(seed)
    count = max(1, min(8, int(count)))
    previews: list[dict[str, Any]] = []
    names = _read_names(ds_dir)
    for _ in range(count):
        path, boxes = items[rng.randrange(len(items))]
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        aug_img, aug_boxes = _apply_aug(img, boxes, ops, it, rng)
        H, W = aug_img.shape[:2]
        vis = aug_img.copy()
        for (c, cx, cy, w, h) in aug_boxes:
            x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
            x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 120), 2)
            lbl = names[c] if c < len(names) else str(c)
            cv2.putText(vis, lbl, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 120), 1, cv2.LINE_AA)
        if W > 480:
            s = 480 / W
            vis = cv2.resize(vis, (480, int(H * s)))
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        previews.append({"src": path.stem, "boxes": len(aug_boxes), "image": f"data:image/jpeg;base64,{b64}"})
    return {"previews": previews, "ops": ops, "intensity": it}


def clear_augmented(ds_dir: Path) -> dict[str, Any]:
    """删除所有 split 中由离线增强生成的样本(<dataset>_aug_*)。"""
    removed = 0
    tag = f"_{AUG_PREFIX_TAG}_"
    for s in BROWSE_SPLITS:
        img_dir = ds_dir / s / "images"
        lbl_dir = ds_dir / s / "labels"
        if not img_dir.exists():
            continue
        for p in list(img_dir.iterdir()):
            if tag in p.stem and p.suffix.lower() in IMG_EXT:
                p.unlink(missing_ok=True)
                (lbl_dir / (p.stem + ".txt")).unlink(missing_ok=True)
                removed += 1
    return {"removed": removed}


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
