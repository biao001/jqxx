"""
control_panel.py — 自绘监控风格控制面板

代替 cv2.createTrackbar 的丑陋原生滑块。外观与主 monitor 窗口一致：
深色底 + 青绿色高亮。支持鼠标点击拖动、悬停高亮。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


# 色板（与 _visualize_monitor 保持一致）
DARK   = (22, 22, 22)
PANEL  = (38, 38, 42)
CYAN   = (200, 255, 80)       # BGR - 亮青绿
GREEN  = (80, 255, 80)
ORANGE = (0, 165, 255)
WHITE  = (240, 240, 240)
DIM    = (140, 140, 140)


@dataclass
class Slider:
    name: str                        # 显示名
    getter: Callable[[], float]      # 读当前值
    setter: Callable[[float], None]  # 写入新值
    vmin: float
    vmax: float
    step: float = 1.0
    fmt: str = "{:.0f}"              # 数值格式

    # 运行时布局（由 panel 赋值）
    y0: int = 0
    y1: int = 0
    bar_x0: int = 0
    bar_x1: int = 0


class ControlPanel:
    WIN = "DMS Controls"
    W = 420
    PAD = 18
    ROW_H = 44
    HEAD_H = 44
    FOOT_H = 28

    def __init__(self, sliders: List[Slider]):
        self.sliders = sliders
        self.H = self.HEAD_H + len(sliders) * self.ROW_H + self.FOOT_H
        self.canvas_base = None
        self._dragging_idx: Optional[int] = None
        self._hover_idx: Optional[int] = None

        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN, self.W, self.H)
        cv2.setMouseCallback(self.WIN, self._on_mouse)

        self._layout()

    # ---------- 布局 ----------

    def _layout(self):
        bar_x0 = self.PAD + 120
        bar_x1 = self.W - self.PAD - 60
        for i, sl in enumerate(self.sliders):
            sl.y0 = self.HEAD_H + i * self.ROW_H
            sl.y1 = sl.y0 + self.ROW_H
            sl.bar_x0 = bar_x0
            sl.bar_x1 = bar_x1

    # ---------- 鼠标回调 ----------

    def _on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, sl in enumerate(self.sliders):
                if sl.y0 <= y <= sl.y1 and sl.bar_x0 - 6 <= x <= sl.bar_x1 + 6:
                    self._dragging_idx = i
                    self._apply_mouse(sl, x)
                    return
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._dragging_idx is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
                sl = self.sliders[self._dragging_idx]
                self._apply_mouse(sl, x)
                return
            # hover 高亮
            self._hover_idx = None
            for i, sl in enumerate(self.sliders):
                if sl.y0 <= y <= sl.y1:
                    self._hover_idx = i
                    break
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging_idx = None

    def _apply_mouse(self, sl: Slider, x: int):
        rng = sl.bar_x1 - sl.bar_x0
        ratio = max(0.0, min(1.0, (x - sl.bar_x0) / rng))
        val = sl.vmin + ratio * (sl.vmax - sl.vmin)
        if sl.step:
            val = round(val / sl.step) * sl.step
        sl.setter(max(sl.vmin, min(sl.vmax, val)))

    # ---------- 绘制 ----------

    def _draw_rounded(self, img, p0, p1, color, thick=1, radius=6, fill=None):
        x1, y1 = p0; x2, y2 = p1
        if fill is not None:
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), fill, -1)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), fill, -1)
            for cx, cy in [(x1+radius,y1+radius), (x2-radius,y1+radius),
                           (x1+radius,y2-radius), (x2-radius,y2-radius)]:
                cv2.circle(img, (cx, cy), radius, fill, -1)
        if thick > 0:
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thick)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thick)
            for cx, cy in [(x1+radius,y1+radius), (x2-radius,y1+radius),
                           (x1+radius,y2-radius), (x2-radius,y2-radius)]:
                cv2.circle(img, (cx, cy), radius, color, thick, cv2.LINE_AA)

    def render(self, extra_info: str = ""):
        img = np.full((self.H, self.W, 3), DARK, dtype=np.uint8)

        # 标题栏（与 monitor 头部呼应）
        cv2.rectangle(img, (0, 0), (self.W, self.HEAD_H - 6), PANEL, -1)
        cv2.putText(img, "DMS", (self.PAD, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2, cv2.LINE_AA)
        cv2.putText(img, "CONTROLS", (self.PAD + 50, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        cv2.line(img, (0, self.HEAD_H - 6), (self.W, self.HEAD_H - 6),
                 CYAN, 1, cv2.LINE_AA)

        # 每个 slider
        for i, sl in enumerate(self.sliders):
            cy = sl.y0 + self.ROW_H // 2
            # 名字
            color_name = CYAN if i == self._hover_idx else WHITE
            cv2.putText(img, sl.name, (self.PAD, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_name, 1, cv2.LINE_AA)
            # 滑条轨道
            rail_y = cy
            cv2.line(img, (sl.bar_x0, rail_y), (sl.bar_x1, rail_y),
                     DIM, 2, cv2.LINE_AA)
            # 当前进度
            v = sl.getter()
            if sl.vmax == sl.vmin:
                ratio = 0
            else:
                ratio = (v - sl.vmin) / (sl.vmax - sl.vmin)
            ratio = max(0.0, min(1.0, ratio))
            knob_x = sl.bar_x0 + int(ratio * (sl.bar_x1 - sl.bar_x0))
            # 已填充部分
            cv2.line(img, (sl.bar_x0, rail_y), (knob_x, rail_y),
                     CYAN, 3, cv2.LINE_AA)
            # 旋钮圆点（带 halo）
            halo = img.copy()
            cv2.circle(halo, (knob_x, rail_y), 10, CYAN, -1)
            img = cv2.addWeighted(halo, 0.2, img, 0.8, 0)
            cv2.circle(img, (knob_x, rail_y), 5, CYAN, -1, cv2.LINE_AA)
            cv2.circle(img, (knob_x, rail_y), 5, DARK, 1, cv2.LINE_AA)
            # 当前值
            val_text = sl.fmt.format(v)
            cv2.putText(img, val_text,
                        (sl.bar_x1 + 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)

        # 底部提示
        cv2.line(img, (0, self.H - self.FOOT_H),
                 (self.W, self.H - self.FOOT_H), DIM, 1)
        foot = extra_info or "drag slider | keys: q=quit r=reset p=pause"
        cv2.putText(img, foot, (self.PAD, self.H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, DIM, 1, cv2.LINE_AA)

        cv2.imshow(self.WIN, img)
        return img
