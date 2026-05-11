"""
_svg_helpers.py — SVG 绘图主题 + 复用构件

与 PPT / README 风格统一的蓝紫渐变 + 圆角卡片。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple


# ---------- 主题 ----------

BLUE        = "#4A7BF5"
BLUE_DARK   = "#2F5ACC"
BLUE_LIGHT  = "#CFDEFB"
PURPLE      = "#6B4FD9"
PURPLE_DARK = "#4A38B0"
PURPLE_LIGHT = "#D9CEF8"
CYAN        = "#22B8CF"
GREEN       = "#22C55E"
ORANGE      = "#F59E0B"
RED         = "#EF4444"
BG          = "#F5F7FF"
CARD        = "#FFFFFF"
BORDER      = "#E5E7FF"
TEXT_DARK   = "#1F2937"
TEXT_MID    = "#4B5563"
TEXT_LIGHT  = "#6B7280"
GRID        = "#E8EAF4"

FONT_STACK = "'Microsoft YaHei', 'SF Pro SC', 'PingFang SC', 'Helvetica Neue', Arial, sans-serif"
MONO_STACK = "'JetBrains Mono', Consolas, 'SF Mono', monospace"


def esc(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))


# ---------- SVG 组件 ----------

def svg_open(w: int, h: int, title: str = "") -> str:
    """SVG 头 + 公共 defs（渐变/阴影/箭头）"""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT_STACK}">
  <title>{esc(title)}</title>
  <defs>
    <linearGradient id="gradBlue" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#6AA1FF"/>
      <stop offset="100%" stop-color="{BLUE_DARK}"/>
    </linearGradient>
    <linearGradient id="gradPurple" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#9883F3"/>
      <stop offset="100%" stop-color="{PURPLE_DARK}"/>
    </linearGradient>
    <linearGradient id="gradBlueFill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#EEF2FF"/>
      <stop offset="100%" stop-color="#DDE3FF"/>
    </linearGradient>
    <linearGradient id="gradPurpleFill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#F3EEFF"/>
      <stop offset="100%" stop-color="#E4DAFD"/>
    </linearGradient>
    <linearGradient id="gradDark" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#1A2040"/>
      <stop offset="100%" stop-color="#0B1028"/>
    </linearGradient>
    <filter id="soft" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#6B4FD9" flood-opacity="0.08"/>
    </filter>
    <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="2" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <marker id="arrowBlue" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="{BLUE}"/>
    </marker>
    <marker id="arrowPurple" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="{PURPLE}"/>
    </marker>
    <marker id="arrowGrey" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#B5BFEB"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="{w}" height="{h}" fill="{BG}"/>
'''


def svg_close() -> str:
    return "</svg>"


def rect(x, y, w, h, fill=CARD, stroke=BORDER, sw=1, rx=10,
         shadow=False, opacity=1.0) -> str:
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}"'
    s += f' fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
    if opacity < 1.0:
        s += f' fill-opacity="{opacity}"'
    if shadow:
        s += ' filter="url(#soft)"'
    s += '/>'
    return s


def circle(cx, cy, r, fill=BLUE, stroke=None, sw=1, opacity=1.0) -> str:
    s = f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"'
    if stroke:
        s += f' stroke="{stroke}" stroke-width="{sw}"'
    if opacity < 1.0:
        s += f' fill-opacity="{opacity}"'
    s += '/>'
    return s


def text(x, y, content, size=13, color=TEXT_DARK, weight=400,
         anchor="start", family=None, dy=0, italic=False) -> str:
    fam = family or FONT_STACK
    style = f'font-size:{size}px;fill:{color};font-weight:{weight};'
    style += f'font-family:{fam};'
    if italic:
        style += 'font-style:italic;'
    return (f'<text x="{x}" y="{y + dy}" text-anchor="{anchor}" '
            f'style="{style}">{esc(content)}</text>')


def title_block(x, y, text_main: str, subtitle: str = "") -> str:
    out = []
    # 蓝色竖线
    out.append(f'<rect x="{x}" y="{y}" width="6" height="30" fill="{BLUE}" rx="2"/>')
    out.append(text(x + 14, y + 24, text_main, size=22, weight=700, color=TEXT_DARK))
    if subtitle:
        out.append(text(x + 14, y + 46, subtitle, size=12, color=TEXT_LIGHT))
    # 横线
    out.append(f'<line x1="{x + 14}" y1="{y + 56}" x2="{x + 860}" y2="{y + 56}" '
               f'stroke="#D5D8E6" stroke-width="1"/>')
    out.append(circle(x + 862, y + 56, 4, fill=BLUE))
    return "\n".join(out)


def arrow_h(x1, y1, x2, y2, color="grey", width=2) -> str:
    marker = {"blue": "arrowBlue", "purple": "arrowPurple",
              "grey": "arrowGrey"}[color]
    stroke = {"blue": BLUE, "purple": PURPLE, "grey": "#B5BFEB"}[color]
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{width}" '
            f'marker-end="url(#{marker})" stroke-linecap="round"/>')


def arrow_bent(x1, y1, x2, y2, color="grey", width=2) -> str:
    """L 形折线箭头"""
    marker = {"blue": "arrowBlue", "purple": "arrowPurple",
              "grey": "arrowGrey"}[color]
    stroke = {"blue": BLUE, "purple": PURPLE, "grey": "#B5BFEB"}[color]
    mid_x = (x1 + x2) / 2
    return (f'<path d="M {x1} {y1} L {mid_x} {y1} L {mid_x} {y2} L {x2} {y2}" '
            f'stroke="{stroke}" stroke-width="{width}" fill="none" '
            f'marker-end="url(#{marker})" stroke-linecap="round"/>')


def icon_badge(cx, cy, r, char, fill="url(#gradBlue)",
               text_color="#FFFFFF", size=None) -> str:
    """圆形图标 + 文字符号"""
    size = size or int(r * 1.0)
    out = [f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"/>']
    out.append(text(cx, cy + int(size * 0.35), char, size=size,
                    color=text_color, weight=700,
                    anchor="middle", family=MONO_STACK))
    return "\n".join(out)


def number_badge(x, y, w, h, num, fill=BLUE_DARK) -> str:
    """01/02 小编号块"""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h//2}" ry="{h//2}" fill="{fill}"/>']
    out.append(text(x + w/2, y + h*0.72, num, size=int(h*0.6),
                    color="#FFFFFF", weight=700, anchor="middle",
                    family=MONO_STACK))
    return "\n".join(out)


def chip(x, y, w, h, label, fill=None, text_color=None, size=12) -> str:
    """圆角标签"""
    fill = fill or BLUE_LIGHT
    tc = text_color or BLUE_DARK
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2}" ry="{h/2}" fill="{fill}"/>']
    out.append(text(x + w/2, y + h*0.68, label, size=size,
                    color=tc, weight=600, anchor="middle"))
    return "\n".join(out)


def card(x, y, w, h, title, body_lines: List[str],
         num: str = None, icon_char: str = None,
         icon_fill: str = "url(#gradBlue)",
         body_size: int = 11, title_size: int = 16,
         title_color=TEXT_DARK, accent=PURPLE) -> str:
    """标准圆角卡片：顶部图标 + 编号 + 标题 + 下划线 + 条目"""
    out = [rect(x, y, w, h, fill=CARD, stroke=BORDER, sw=1, rx=10, shadow=True)]
    inner_y = y + 16
    if icon_char:
        out.append(icon_badge(x + w/2, y, 22, icon_char, fill=icon_fill))
        inner_y = y + 30
    if num:
        out.append(number_badge(x + 14, inner_y, 36, 20, num, fill=BLUE_DARK))
        inner_y += 28
    # 标题
    out.append(text(x + 14, inner_y + title_size, title,
                    size=title_size, weight=700, color=title_color))
    # 紫色短横
    out.append(f'<rect x="{x + 14}" y="{inner_y + title_size + 8}" width="32" height="3" '
               f'fill="{accent}" rx="1.5"/>')
    # 条目
    line_y = inner_y + title_size + 26
    for item in body_lines:
        out.append(circle(x + 20, line_y - 4, 3, fill=accent))
        out.append(text(x + 28, line_y, item, size=body_size, color=TEXT_MID))
        line_y += body_size + 9
    return "\n".join(out)


def formula_box(x, y, w, h, formula: str, color_bg="#EEF2FF",
                 color_border=BLUE, text_color=BLUE_DARK,
                 size=14) -> str:
    out = [rect(x, y, w, h, fill=color_bg, stroke=color_border, sw=1.5, rx=8)]
    out.append(text(x + w/2, y + h*0.62, formula, size=size,
                    color=text_color, weight=700, anchor="middle",
                    family=MONO_STACK))
    return "\n".join(out)


def code_box(x, y, w, h, lines: List[str], size=11,
             bg="#0F172A", text_color="#CBD5E1",
             keyword_color="#A5B4FC", string_color="#FCA5A5",
             comment_color="#64748B") -> str:
    """伪代码框 + 基础高亮"""
    out = [rect(x, y, w, h, fill=bg, stroke="#1E293B", sw=1, rx=6)]
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="22" rx="6" fill="#1E293B"/>')
    # 红黄绿点
    for i, col in enumerate(["#FF5F57", "#FFBD2E", "#27C93F"]):
        out.append(circle(x + 12 + i * 14, y + 11, 5, fill=col))
    # 内容
    ly = y + 40
    for line in lines:
        col = text_color
        if line.strip().startswith("#") or line.strip().startswith("//"):
            col = comment_color
        out.append(text(x + 12, ly, line, size=size, color=col,
                        family=MONO_STACK))
        ly += size + 6
    return "\n".join(out)


def camera_deco(x: int, y: int) -> str:
    """右上装饰：像素阵 + 镜头"""
    out = []
    for r in range(3):
        for c in range(4):
            px = x + c * 10
            py = y + r * 10
            shade = [PURPLE, BLUE, PURPLE_LIGHT][(r + c) % 3]
            out.append(f'<rect x="{px}" y="{py}" width="7" height="7" fill="{shade}"/>')
    cx = x + 75; cy = y + 20
    out.append(circle(cx, cy, 22, fill="#EEEFF8"))
    out.append(circle(cx, cy, 17, fill="#2D387A"))
    out.append(circle(cx, cy, 12, fill="#0F1640"))
    out.append(circle(cx + 3, cy - 3, 4, fill=BLUE))
    return "\n".join(out)


def bottom_chips(cx: int, y: int, chips_data: List[Tuple[str, str]],
                 total_w: int = 860) -> str:
    """底部胶囊状态栏"""
    out = [rect(cx - total_w/2, y, total_w, 44,
                fill="#FBFCFF", stroke="#D0D6F2", sw=1, rx=22)]
    n = len(chips_data)
    seg = total_w / n
    for i, (icn, lbl) in enumerate(chips_data):
        ix = cx - total_w/2 + seg * i + seg/2
        fill = "url(#gradBlue)" if i % 2 == 0 else "url(#gradPurple)"
        out.append(icon_badge(ix - 60, y + 22, 14, icn, fill=fill, size=13))
        out.append(text(ix - 42, y + 28, lbl, size=14, weight=600,
                        color=TEXT_DARK))
        if i < n - 1:
            x_div = cx - total_w/2 + seg * (i + 1)
            out.append(f'<line x1="{x_div}" y1="{y + 10}" x2="{x_div}" y2="{y + 34}" '
                       f'stroke="#D5DAEE" stroke-width="1"/>')
    return "\n".join(out)


def write_svg(path: str, content: str) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[ok] {path}")
