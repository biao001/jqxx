"""
review_dataset.py — Gradio 数据集筛选页

对合并后的 data/dms_unified 逐张翻看（带 GT 标注框 + 类别名），
人工剔除"污染"样本。删除 = 把 图片+标签 移到 dms_unified/_trash/，可恢复。
筛完直接用 dms_unified 重新训练即可（无需重新 merge）。

启动:
  python review_dataset.py
  # 局域网: GRADIO_SERVER_NAME=0.0.0.0 python review_dataset.py
浏览器: http://127.0.0.1:7861
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "dms_unified"
TRASH = DATA / "_trash"

NAMES = ["phone_use", "smoking", "drinking", "eating", "no_seatbelt", "seatbelt"]
COLORS = [
    (255, 80, 80), (80, 80, 255), (80, 200, 255), (80, 255, 120),
    (255, 80, 255), (120, 255, 255),
]


def list_items(split_filter="all", class_filter="all"):
    """返回 [(split, img_path, label_path, classes_set)]，按筛选条件"""
    items = []
    splits = ["train", "valid"] if split_filter == "all" else [split_filter]
    for split in splits:
        img_dir = DATA / split / "images"
        lbl_dir = DATA / split / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if not img.is_file():
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            classes = set()
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    if line.strip():
                        classes.add(int(line.split()[0]))
            if class_filter != "all":
                cid = NAMES.index(class_filter)
                if cid not in classes:
                    continue
            items.append((split, img, lbl, classes))
    return items


def render(split, img_path: Path, lbl_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return np.zeros((360, 480, 3), dtype=np.uint8)
    H, W = img.shape[:2]
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
            x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
            color = COLORS[cid % len(COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            name = NAMES[cid] if cid < len(NAMES) else str(cid)
            cv2.putText(img, name, (x1, max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_state(split_filter, class_filter):
    items = list_items(split_filter, class_filter)
    return {"items": items, "idx": 0, "deleted": []}


def view(state):
    items = state["items"]
    if not items:
        return None, "（没有匹配的图片）", state
    idx = state["idx"] % len(items)
    state["idx"] = idx
    split, img_path, lbl_path, classes = items[idx]
    rgb = render(split, img_path, lbl_path)
    cls_str = ", ".join(NAMES[c] for c in sorted(classes)) or "（无标注）"
    info = (f"**{idx+1} / {len(items)}**  ·  split=`{split}`  ·  类别: {cls_str}\n\n"
            f"`{img_path.name}`\n\n已删除本轮: {len(state['deleted'])} 张")
    return rgb, info, state


def nav(state, step):
    if state["items"]:
        state["idx"] = (state["idx"] + step) % len(state["items"])
    return view(state)


def delete_current(state):
    items = state["items"]
    if not items:
        return None, "（空）", state
    idx = state["idx"] % len(items)
    split, img_path, lbl_path, classes = items[idx]
    # 移到 _trash
    for src in (img_path, lbl_path):
        if src.exists():
            dst = TRASH / split / src.parent.name / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    state["deleted"].append((split, img_path.name))
    items.pop(idx)
    if idx >= len(items):
        state["idx"] = max(0, len(items) - 1)
    return view(state)


def undo_last(state):
    if not state["deleted"]:
        return view(state)
    split, name = state["deleted"].pop()
    stem = Path(name).stem
    for sub in ("images", "labels"):
        src = TRASH / split / sub / (name if sub == "images" else stem + ".txt")
        if src.exists():
            dst = DATA / split / sub / src.name
            shutil.move(str(src), str(dst))
    # 重新载入当前筛选（简单起见重建 items）
    return view(state)


def reload_state(split_filter, class_filter, state):
    new = build_state(split_filter, class_filter)
    new["deleted"] = state.get("deleted", [])
    return view(new)


def jump(state, target):
    if state["items"]:
        try:
            t = int(target) - 1
            state["idx"] = max(0, min(t, len(state["items"]) - 1))
        except (ValueError, TypeError):
            pass
    return view(state)


with gr.Blocks(title="DMS 数据集筛选") as demo:
    gr.Markdown("# DMS 数据集筛选页\n逐张审查 `dms_unified`，删除 = 移到 `_trash/`（可撤销）。筛完直接重训。")

    with gr.Row():
        split_dd = gr.Dropdown(["all", "train", "valid"], value="all", label="split")
        class_dd = gr.Dropdown(["all"] + NAMES, value="all", label="只看类别")
        reload_btn = gr.Button("🔄 应用筛选 / 重新载入")

    img_out = gr.Image(label="标注预览 (GT bbox)", height=480)
    info_md = gr.Markdown()

    with gr.Row():
        prev_btn = gr.Button("← 上一张")
        next_btn = gr.Button("下一张 →")
        del_btn = gr.Button("🗑 删除这张", variant="stop")
        undo_btn = gr.Button("↩ 撤销上次删除")
    with gr.Row():
        jump_in = gr.Textbox(label="跳到第几张", scale=3)
        jump_btn = gr.Button("跳转", scale=1)

    st = gr.State(build_state("all", "all"))

    demo.load(view, st, [img_out, info_md, st])
    reload_btn.click(reload_state, [split_dd, class_dd, st], [img_out, info_md, st])
    prev_btn.click(lambda s: nav(s, -1), st, [img_out, info_md, st])
    next_btn.click(lambda s: nav(s, +1), st, [img_out, info_md, st])
    del_btn.click(delete_current, st, [img_out, info_md, st])
    undo_btn.click(undo_last, st, [img_out, info_md, st])
    jump_btn.click(jump, [st, jump_in], [img_out, info_md, st])


if __name__ == "__main__":
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7861"))
    demo.launch(server_name=server_name, server_port=server_port,
                inbrowser=False)
