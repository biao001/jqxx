"""
build_diagrams.py — 生成 5 张精美 SVG 流程/原理图

输出到 docs/diagrams/
  01_yolo_architecture.svg   YOLOv8 网络原理
  02_inference_pipeline.svg  端到端推理流程
  03_io_schema.svg           输入输出数据结构
  04_optimizations.svg       算法优化对比
  05_risk_scoring.svg        时间窗口 + 加权评分
"""
from _svg_helpers import (
    svg_open, svg_close, rect, circle, text, title_block,
    arrow_h, arrow_bent, icon_badge, number_badge, chip, card,
    formula_box, code_box, camera_deco, bottom_chips, write_svg,
    BLUE, BLUE_DARK, BLUE_LIGHT, PURPLE, PURPLE_DARK, PURPLE_LIGHT,
    CYAN, GREEN, ORANGE, RED, BG, CARD, BORDER,
    TEXT_DARK, TEXT_MID, TEXT_LIGHT, GRID,
    FONT_STACK, MONO_STACK, esc,
)


# ==========================================================
# 01  YOLOv8 网络架构原理
# ==========================================================

def build_yolo_architecture():
    W, H = 1600, 900
    parts = [svg_open(W, H, "YOLOv8 Architecture")]
    parts.append(title_block(60, 40,
        "YOLOv8 网络原理",
        "Backbone（CSP）→ Neck（PAN-FPN）→ Anchor-Free Head · 单次前向输出多尺度检测"))
    parts.append(camera_deco(1420, 40))

    # --- 输入 ---
    ix, iy = 70, 160
    parts.append(rect(ix, iy, 170, 220, fill=CARD, stroke=BORDER, shadow=True))
    parts.append(icon_badge(ix + 85, iy + 10, 22, "▶", fill="url(#gradBlue)"))
    parts.append(text(ix + 85, iy + 70, "输入图像", size=15, weight=700,
                      anchor="middle", color=TEXT_DARK))
    # 占位图
    parts.append(rect(ix + 25, iy + 85, 120, 90, fill="#1E2445", rx=6))
    # 人形
    parts.append(circle(ix + 85, iy + 115, 12, fill="#6B7BC8"))
    parts.append(rect(ix + 65, iy + 130, 40, 35, fill="#6B7BC8", rx=4))
    parts.append(text(ix + 85, iy + 195, "640×640×3",
                      size=11, color=TEXT_LIGHT, anchor="middle",
                      family=MONO_STACK))
    parts.append(text(ix + 85, iy + 215, "BGR uint8",
                      size=11, color=TEXT_LIGHT, anchor="middle",
                      family=MONO_STACK))
    parts.append(arrow_h(245, iy + 110, 285, iy + 110, "blue"))

    # --- Backbone ---
    bx, by = 290, 160
    parts.append(rect(bx, by, 290, 550, fill="url(#gradBlueFill)",
                      stroke=BLUE, sw=1.2, rx=12, shadow=True))
    parts.append(number_badge(bx + 14, by + 14, 38, 20, "01"))
    parts.append(text(bx + 14, by + 60, "Backbone",
                      size=18, weight=700, color=BLUE_DARK))
    parts.append(text(bx + 14, by + 80, "CSPDarknet · C2f 结构",
                      size=11, color=TEXT_MID))

    # Backbone stages
    stages = [
        ("Stem Conv", "3×3 s=2", "320×320", "#BAD1FF"),
        ("C2f + DW", "Stage 1", "160×160", "#A5C4FF"),
        ("C2f + DW", "Stage 2", "80×80",   "#8FB6FF"),
        ("C2f + DW", "Stage 3", "40×40",   "#7AA9FF"),
        ("SPPF",     "金字塔池化", "20×20", "#5A8EF5"),
    ]
    sy = by + 100
    for name, sub, size_txt, col in stages:
        parts.append(rect(bx + 16, sy, 258, 56, fill=col, stroke="none", rx=8))
        parts.append(text(bx + 30, sy + 22, name, size=13, weight=700, color="#0F1B3C"))
        parts.append(text(bx + 30, sy + 40, sub, size=10, color="#2E3A6C"))
        parts.append(text(bx + 258, sy + 36, size_txt, size=11,
                          color="#0F1B3C", weight=600, anchor="end",
                          family=MONO_STACK))
        sy += 68

    # Backbone 输出三尺度
    parts.append(arrow_h(580, by + 210, 620, by + 210, "blue"))
    parts.append(arrow_h(580, by + 340, 620, by + 340, "blue"))
    parts.append(arrow_h(580, by + 470, 620, by + 470, "blue"))

    # --- Neck (PAN-FPN) ---
    nx, ny = 625, 160
    parts.append(rect(nx, ny, 290, 550, fill="url(#gradPurpleFill)",
                      stroke=PURPLE, sw=1.2, rx=12, shadow=True))
    parts.append(number_badge(nx + 14, ny + 14, 38, 20, "02"))
    parts.append(text(nx + 14, ny + 60, "Neck (PAN-FPN)",
                      size=18, weight=700, color=PURPLE_DARK))
    parts.append(text(nx + 14, ny + 80, "多尺度特征融合 · 上下采样双路径",
                      size=11, color=TEXT_MID))

    # FPN/PAN 可视化
    feats = [("P3", "80×80", "小目标"),
             ("P4", "40×40", "中目标"),
             ("P5", "20×20", "大目标")]
    for i, (k, sz, desc) in enumerate(feats):
        fy = ny + 140 + i * 150
        # Up path
        parts.append(rect(nx + 20, fy, 110, 100, fill=PURPLE_LIGHT, rx=8))
        parts.append(text(nx + 75, fy + 28, k, size=17, weight=700,
                          anchor="middle", color=PURPLE_DARK,
                          family=MONO_STACK))
        parts.append(text(nx + 75, fy + 52, sz, size=11, anchor="middle",
                          color="#3D2B7A", family=MONO_STACK))
        parts.append(text(nx + 75, fy + 80, desc, size=11, anchor="middle",
                          color=TEXT_MID))
        # Down path
        parts.append(rect(nx + 160, fy, 110, 100, fill="#D9CEF8", rx=8))
        parts.append(text(nx + 215, fy + 28, f"N{3+i}", size=17, weight=700,
                          anchor="middle", color=PURPLE_DARK,
                          family=MONO_STACK))
        parts.append(text(nx + 215, fy + 52, sz, size=11, anchor="middle",
                          color="#3D2B7A", family=MONO_STACK))
        parts.append(text(nx + 215, fy + 80, "融合输出", size=11, anchor="middle",
                          color=TEXT_MID))
        # FPN 连接
        parts.append(arrow_h(nx + 130, fy + 50, nx + 160, fy + 50, "purple", 1.5))

    # FPN 内部 up/down 箭头
    parts.append(f'<path d="M {nx+75} {ny+240} L {nx+75} {ny+440}" '
                 f'stroke="{PURPLE}" stroke-width="1.5" fill="none" '
                 f'stroke-dasharray="4,4"/>')
    parts.append(text(nx + 55, ny + 340, "↑", size=22,
                      color=PURPLE, weight=700))
    parts.append(text(nx + 80, ny + 345, "上采样", size=10,
                      color=PURPLE_DARK, anchor="start"))

    parts.append(f'<path d="M {nx+215} {ny+440} L {nx+215} {ny+240}" '
                 f'stroke="{PURPLE}" stroke-width="1.5" fill="none" '
                 f'stroke-dasharray="4,4"/>')
    parts.append(text(nx + 235, ny + 340, "↓", size=22,
                      color=PURPLE, weight=700))
    parts.append(text(nx + 260, ny + 345, "下采样", size=10,
                      color=PURPLE_DARK, anchor="start"))

    parts.append(arrow_h(920, by + 210, 960, by + 210, "purple"))
    parts.append(arrow_h(920, by + 340, 960, by + 340, "purple"))
    parts.append(arrow_h(920, by + 470, 960, by + 470, "purple"))

    # --- Head ---
    hx, hy = 965, 160
    parts.append(rect(hx, hy, 290, 550, fill=CARD, stroke=BLUE, sw=1.2,
                      rx=12, shadow=True))
    parts.append(number_badge(hx + 14, hy + 14, 38, 20, "03"))
    parts.append(text(hx + 14, hy + 60, "Detect Head (anchor-free)",
                      size=17, weight=700, color=BLUE_DARK))
    parts.append(text(hx + 14, hy + 80, "解耦输出：分类 + 回归 + DFL",
                      size=11, color=TEXT_MID))

    # 三个 head 分支
    for i, (k, sz) in enumerate([("Head@P3/N3", "80×80"),
                                   ("Head@P4/N4", "40×40"),
                                   ("Head@P5/N5", "20×20")]):
        fy = hy + 120 + i * 150
        parts.append(rect(hx + 20, fy, 250, 120, fill="#F4F7FF",
                          stroke=BLUE_LIGHT, rx=8))
        parts.append(text(hx + 30, fy + 20, k, size=12, weight=700,
                          color=BLUE_DARK, family=MONO_STACK))
        parts.append(text(hx + 260, fy + 20, sz, size=11,
                          color=TEXT_LIGHT, anchor="end",
                          family=MONO_STACK))
        # cls + reg 分支
        parts.append(rect(hx + 30, fy + 35, 110, 30, fill=BLUE_LIGHT, rx=4))
        parts.append(text(hx + 85, fy + 55, "cls (8 类)", size=11,
                          weight=600, color=BLUE_DARK, anchor="middle",
                          family=MONO_STACK))
        parts.append(rect(hx + 150, fy + 35, 110, 30, fill=PURPLE_LIGHT, rx=4))
        parts.append(text(hx + 205, fy + 55, "reg (4)", size=11,
                          weight=600, color=PURPLE_DARK, anchor="middle",
                          family=MONO_STACK))
        parts.append(text(hx + 30, fy + 85, "• anchor-free 中心点",
                          size=10, color=TEXT_MID))
        parts.append(text(hx + 30, fy + 102, "• DFL 分布式回归",
                          size=10, color=TEXT_MID))

    # --- 输出 ---
    parts.append(arrow_h(1260, 435, 1300, 435, "blue"))
    ox, oy = 1305, 220
    parts.append(rect(ox, oy, 260, 420, fill=CARD, stroke=BORDER,
                      rx=12, shadow=True))
    parts.append(text(ox + 130, oy + 30, "检测输出", size=17, weight=700,
                      color=TEXT_DARK, anchor="middle"))
    parts.append(f'<rect x="{ox+14}" y="{oy+40}" width="32" height="3" fill="{PURPLE}" rx="1.5"/>')

    # 示例 JSON
    det_items = [
        ("cls_id",    "int", "类别 ID 0-7"),
        ("class",     "str", "phone_use / calling..."),
        ("confidence", "float", "0.0–1.0"),
        ("bbox",      "xyxy", "[x1, y1, x2, y2]"),
        ("stride",    "int",  "8 / 16 / 32"),
    ]
    dy = oy + 70
    for name, t, desc in det_items:
        parts.append(rect(ox + 14, dy, 232, 54, fill="#F9FAFF",
                          stroke=BORDER, rx=6))
        parts.append(text(ox + 26, dy + 22, name, size=13,
                          weight=700, color=BLUE_DARK, family=MONO_STACK))
        parts.append(chip(ox + 180, dy + 8, 60, 18, t,
                          fill=PURPLE_LIGHT, text_color=PURPLE_DARK, size=10))
        parts.append(text(ox + 26, dy + 42, desc, size=11, color=TEXT_MID))
        dy += 62

    # 右下角小说明
    parts.append(rect(ox + 10, oy + 390, 240, 40, fill=BLUE_LIGHT, rx=6))
    parts.append(text(ox + 130, oy + 414, "NMS → 最终 bbox 列表",
                      size=12, weight=700, color=BLUE_DARK,
                      anchor="middle", family=MONO_STACK))

    # --- 底部关键点说明 ---
    notes = [
        ("Anchor-Free", "不再使用预设框，直接预测中心点 + 宽高偏移"),
        ("DFL Loss",    "Distribution Focal Loss，对边界分布建模"),
        ("C2f 模块",    "Cross Stage Partial 改进版，特征复用效率高"),
        ("TAL Assign",  "Task-Aligned Learning 动态标签分配"),
    ]
    ny2 = 760
    for i, (k, d) in enumerate(notes):
        x0 = 70 + i * 380
        parts.append(rect(x0, ny2, 360, 90, fill=CARD, stroke=BORDER,
                          rx=8, shadow=True))
        parts.append(circle(x0 + 28, ny2 + 28, 15, fill="url(#gradPurple)"))
        parts.append(text(x0 + 28, ny2 + 33, str(i+1), size=14,
                          color="#FFFFFF", weight=700, anchor="middle",
                          family=MONO_STACK))
        parts.append(text(x0 + 52, ny2 + 32, k, size=14, weight=700,
                          color=TEXT_DARK))
        parts.append(text(x0 + 16, ny2 + 62, d, size=11, color=TEXT_MID))

    parts.append(svg_close())
    write_svg("docs/diagrams/01_yolo_architecture.svg", "\n".join(parts))


# ==========================================================
# 02  端到端推理流程
# ==========================================================

def build_inference_pipeline():
    W, H = 1600, 900
    parts = [svg_open(W, H, "Inference Pipeline")]
    parts.append(title_block(60, 40, "端到端推理流程",
        "从单帧输入到结构化 JSON 输出的全链路（本地 OpenCV / Gradio 共用）"))
    parts.append(camera_deco(1420, 40))

    # 主流水线 9 步
    steps = [
        ("01", "帧采集", "Camera Stream\n异步线程 BGR", BLUE, "▶"),
        ("02", "镜头检查", "Laplacian 方差\nmean 阈值判定", PURPLE, "◉"),
        ("03", "低光增强", "CLAHE on Y\n只当 mean<80", BLUE, "☀"),
        ("04", "YOLOv8 检测", "person / phone\n+ 自训 unified.pt", PURPLE, "▣"),
        ("05", "驾驶员 Crop", "选最大 person\n+10% padding", BLUE, "◈"),
        ("06", "Pose 关键点", "17 点 COCO\nnose/ear/wrist", PURPLE, "☍"),
        ("07", "规则判定", "phone ↔ 头部\nwheel ROI", BLUE, "⚖"),
        ("08", "时序滑窗", "K=5 投票\n3/3 去抖", PURPLE, "⏱"),
        ("09", "风险评分", "加权互补\n0-100 + tier", BLUE, "★"),
    ]

    start_x, start_y = 60, 160
    box_w, box_h = 160, 150
    gap_x = 7
    for i, (num, title_t, body, col, icn) in enumerate(steps):
        x = start_x + i * (box_w + gap_x)
        y = start_y
        parts.append(rect(x, y, box_w, box_h, fill=CARD, stroke=BORDER,
                          rx=10, shadow=True))
        grad = "url(#gradBlue)" if col == BLUE else "url(#gradPurple)"
        parts.append(icon_badge(x + box_w/2, y, 22, icn, fill=grad))
        parts.append(number_badge(x + 12, y + 22, 34, 18, num))
        parts.append(text(x + box_w/2, y + 68, title_t, size=14,
                          weight=700, color=TEXT_DARK, anchor="middle"))
        for j, bl in enumerate(body.split("\n")):
            parts.append(text(x + box_w/2, y + 95 + j*16, bl,
                              size=10, color=TEXT_MID, anchor="middle",
                              family=MONO_STACK))
        if i < len(steps) - 1:
            ax1 = x + box_w + 1
            ax2 = x + box_w + gap_x - 1
            parts.append(arrow_h(ax1, y + box_h/2 + 10, ax2,
                                 y + box_h/2 + 10, "grey", 2))

    # ---- 旁支模块 ----
    # 上路 : 训练好的权重注入
    yt = 360
    parts.append(rect(700, yt, 300, 90, fill="url(#gradPurpleFill)",
                      stroke=PURPLE, rx=10, shadow=True))
    parts.append(icon_badge(735, yt + 30, 18, "⚙", fill="url(#gradPurple)"))
    parts.append(text(760, yt + 28, "models/unified.pt", size=13,
                      weight=700, color=PURPLE_DARK, family=MONO_STACK))
    parts.append(text(760, yt + 50, "8 类 YOLOv8n · 6 MB",
                      size=11, color=TEXT_MID))
    parts.append(text(760, yt + 70, "由 train_unified.py 产出",
                      size=11, color=TEXT_MID))
    # 箭头指向 step 04 / step 06 底部
    parts.append(f'<path d="M 700 {yt+30} L 630 {yt+30} L 630 {start_y+box_h+5} '
                 f'L 600 {start_y+box_h+5}" '
                 f'stroke="{PURPLE}" stroke-width="1.5" fill="none" '
                 f'stroke-dasharray="6,4" marker-end="url(#arrowPurple)"/>')

    # 下路 : 时序滑窗机制详解
    yw = 490
    parts.append(rect(70, yw, 960, 160, fill=CARD, stroke=BORDER,
                      rx=10, shadow=True))
    parts.append(text(90, yw + 28, "⟳ 时序滑窗（K=5 激活3/失活3）",
                      size=15, weight=700, color=TEXT_DARK))
    parts.append(text(90, yw + 48, "抑制单帧抖动 · 同时计算 duration_s",
                      size=11, color=TEXT_MID))
    # 画 10 个格子模拟时序
    gx = 400; gy = yw + 30
    states = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1]  # 示例
    labels = ["t-9", "t-8", "t-7", "t-6", "t-5", "t-4", "t-3", "t-2", "t-1", "t"]
    activated_from = 4
    for i, (s, lbl) in enumerate(zip(states, labels)):
        cx = gx + i * 52
        fill = PURPLE if s else "#E5E7FF"
        if i >= activated_from:
            stroke = GREEN
            sw = 2
        else:
            stroke = BORDER
            sw = 1
        parts.append(rect(cx, gy, 44, 44, fill=fill, stroke=stroke, sw=sw, rx=6))
        ch = "●" if s else "○"
        parts.append(text(cx + 22, gy + 28, ch, size=20,
                          color="#FFFFFF" if s else TEXT_LIGHT,
                          anchor="middle", weight=700))
        parts.append(text(cx + 22, gy + 60, lbl, size=10,
                          color=TEXT_LIGHT, anchor="middle",
                          family=MONO_STACK))
    # 激活时刻标注
    ax = gx + activated_from * 52 + 22
    parts.append(f'<line x1="{ax}" y1="{gy-10}" x2="{ax}" y2="{gy+48}" '
                 f'stroke="{GREEN}" stroke-width="2" stroke-dasharray="3,3"/>')
    parts.append(text(ax, gy - 16, "↑ 激活 (3/5)",
                      size=11, color=GREEN, weight=700, anchor="middle"))

    # 右上小说明
    parts.append(text(90, yw + 90, "• 窗内命中 ≥3 → 稳定报告，开始计 duration",
                      size=11, color=TEXT_MID))
    parts.append(text(90, yw + 110, "• 窗内未命中 ≥3 → 清除；抖动/瞬时检出被滤掉",
                      size=11, color=TEXT_MID))
    parts.append(text(90, yw + 130, "• 结果 JSON 含 duration_s，用于风险评分加成",
                      size=11, color=TEXT_MID))

    # 下方输出 JSON 面板
    yj = 680
    parts.append(rect(70, yj, 960, 190, fill="#0F172A", rx=10))
    parts.append(f'<rect x="70" y="{yj}" width="960" height="22" rx="10" fill="#1E293B"/>')
    for i, c in enumerate(["#FF5F57", "#FFBD2E", "#27C93F"]):
        parts.append(circle(84 + i*14, yj + 11, 5, fill=c))
    parts.append(text(180, yj + 15, "BehaviorDetector.predict(frame) → JSON",
                      size=11, color="#CBD5E1", family=MONO_STACK))
    json_lines = [
        '{',
        '  "frame_id": 1024,',
        '  "timestamp": 1713772800.123,',
        '  "latency_ms": 38.5,',
        '  "behaviors": [',
        '    {"type":"calling","confidence":0.87,"bbox":[412,160,495,240],',
        '     "severity":"high","duration_s":2.4,"evidence":"..."}',
        '  ],',
        '  "alert_level":"high", "risk_score":68.5, "risk_tier":"warning",',
        '  "driver_present":true, "camera_ok":true',
        '}',
    ]
    for i, ln in enumerate(json_lines):
        col = "#CBD5E1"
        if '"' in ln and ":" in ln:
            col = "#E2E8F0"
        parts.append(text(84, yj + 50 + i * 15, ln, size=11,
                          color=col, family=MONO_STACK))

    # 右侧 : 输出去向
    parts.append(rect(1060, yj, 470, 190, fill=CARD, stroke=BORDER,
                      rx=10, shadow=True))
    parts.append(text(1080, yj + 30, "输出消费者", size=15, weight=700,
                      color=TEXT_DARK))
    parts.append(f'<rect x="1080" y="{yj+38}" width="32" height="3" fill="{PURPLE}" rx="1.5"/>')
    downstream = [
        ("◱", "监控 UI", "live_client.py 可视化绘制"),
        ("⊞", "5 号系统集成", "ZMQ / ROS topic 分发"),
        ("☰", "日志存档", "JSONL 追加 + 截图"),
        ("♫", "告警反馈", "语音合成 + 仪表盘震动"),
    ]
    for i, (ic, k, v) in enumerate(downstream):
        dy = yj + 50 + i * 32
        parts.append(icon_badge(1095, dy + 9, 11, ic,
            fill="url(#gradBlue)" if i%2==0 else "url(#gradPurple)", size=11))
        parts.append(text(1115, dy + 14, k, size=12, weight=700,
                          color=TEXT_DARK))
        parts.append(text(1215, dy + 14, v, size=11, color=TEXT_MID))

    # 左下 FPS 小卡
    parts.append(rect(1040, yt - 30, 180, 130, fill="url(#gradBlueFill)",
                      stroke=BLUE, rx=10, shadow=True))
    parts.append(text(1130, yt - 8, "性能", size=13, weight=700,
                      color=BLUE_DARK, anchor="middle"))
    for i, (k, v) in enumerate([("CPU (n=384)", "28 FPS"),
                                  ("CPU + skip 2", "50 FPS"),
                                  ("GPU", "80+ FPS")]):
        dy = yt + i * 26
        parts.append(text(1055, dy + 16, k, size=11,
                          color=TEXT_MID, family=MONO_STACK))
        parts.append(text(1205, dy + 16, v, size=12, weight=700,
                          color=BLUE_DARK, anchor="end", family=MONO_STACK))

    # 性能箭头指向 step 5/9
    parts.append(f'<path d="M 1040 {yt+40} L 1000 {yt+40}" stroke="{BLUE}" '
                 f'stroke-width="1.5" fill="none" stroke-dasharray="5,4" '
                 f'marker-end="url(#arrowBlue)"/>')

    parts.append(svg_close())
    write_svg("docs/diagrams/02_inference_pipeline.svg", "\n".join(parts))


# ==========================================================
# 03  输入 / 输出数据结构设计
# ==========================================================

def build_io_schema():
    W, H = 1600, 900
    parts = [svg_open(W, H, "I/O Schema")]
    parts.append(title_block(60, 40, "输入 / 输出数据结构设计",
        "面向 5 号系统集成的标准契约 · 向后兼容扩展"))
    parts.append(camera_deco(1420, 40))

    # ==== 左：输入 ====
    lx, ly = 60, 160
    parts.append(rect(lx, ly, 460, 670, fill="#FBFCFF",
                      stroke=BLUE_LIGHT, sw=1.5, rx=12, shadow=True))
    parts.append(icon_badge(lx + 30, ly + 20, 16, "↓", fill="url(#gradBlue)"))
    parts.append(text(lx + 55, ly + 30, "输入 INPUT", size=18,
                      weight=700, color=TEXT_DARK))
    parts.append(f'<rect x="{lx+20}" y="{ly+50}" width="40" height="3" fill="{BLUE}" rx="1.5"/>')

    # 帧字段
    y_ = ly + 80
    parts.append(text(lx + 24, y_, "Frame Payload", size=13,
                      weight=700, color=BLUE_DARK))
    in_fields = [
        ("frame",       "np.ndarray", "(H,W,3) uint8 BGR", "必填"),
        ("frame_id",    "int",        "单调递增", "必填"),
        ("timestamp",   "float",      "Unix 秒", "必填"),
        ("camera_id",   "str",        "多摄像头场景",  "可选"),
    ]
    y_ += 12
    for name, t, desc, req in in_fields:
        y_ += 52
        parts.append(rect(lx + 18, y_ - 30, 420, 46, fill=CARD,
                          stroke=BORDER, rx=6))
        parts.append(text(lx + 32, y_ - 12, name, size=13, weight=700,
                          color=BLUE_DARK, family=MONO_STACK))
        type_col = PURPLE_LIGHT
        parts.append(chip(lx + 170, y_ - 24, 80, 18, t,
                          fill=type_col, text_color=PURPLE_DARK, size=10))
        req_fill = GREEN if req == "必填" else "#CBD5E1"
        req_txt = "#FFFFFF" if req == "必填" else TEXT_MID
        parts.append(chip(lx + 260, y_ - 24, 42, 18, req,
                          fill=req_fill, text_color=req_txt, size=10))
        parts.append(text(lx + 32, y_ + 4, desc, size=11, color=TEXT_MID))

    # 分辨率规范
    y_ += 50
    parts.append(rect(lx + 18, y_, 420, 170, fill="#EEF2FF",
                      stroke=BLUE_LIGHT, rx=8))
    parts.append(text(lx + 30, y_ + 22, "物理规格", size=13, weight=700,
                      color=BLUE_DARK))
    specs = [
        ("分辨率", "建议 ≥ 640×480，最小 320×240"),
        ("色彩", "BGR (OpenCV 默认) 或 Gray（红外）"),
        ("帧率",  "15 ~ 30 FPS"),
        ("来源",  "USB 摄像头 / 车载 CAM / RTSP / 视频文件"),
    ]
    for i, (k, v) in enumerate(specs):
        yy = y_ + 46 + i * 28
        parts.append(circle(lx + 34, yy - 4, 3, fill=PURPLE))
        parts.append(text(lx + 44, yy, k, size=12, weight=600,
                          color=BLUE_DARK))
        parts.append(text(lx + 110, yy, v, size=11, color=TEXT_MID))

    # ==== 中：处理 ====
    mx, my = 560, 380
    parts.append(rect(mx, my, 240, 200, fill="url(#gradPurpleFill)",
                      stroke=PURPLE, sw=1.5, rx=14, shadow=True))
    parts.append(circle(mx + 120, my + 35, 32, fill="url(#gradPurple)"))
    parts.append(icon_badge(mx + 120, my + 35, 18, "⚙", fill="url(#gradPurple)"))
    parts.append(text(mx + 120, my + 100, "BehaviorDetector",
                      size=14, weight=700, color=PURPLE_DARK,
                      anchor="middle", family=MONO_STACK))
    parts.append(text(mx + 120, my + 125, "predict(frame, ...)",
                      size=11, color=TEXT_MID, anchor="middle",
                      family=MONO_STACK))
    parts.append(text(mx + 120, my + 155, "YOLO + Pose + 规则", size=11,
                      color=TEXT_MID, anchor="middle"))
    parts.append(text(mx + 120, my + 175, "+ 时序滑窗 + 加权评分", size=11,
                      color=TEXT_MID, anchor="middle"))

    # 左右箭头
    parts.append(arrow_h(lx + 460, my + 100, mx - 5, my + 100, "purple", 2.5))
    parts.append(arrow_h(mx + 240, my + 100, mx + 280, my + 100, "purple", 2.5))

    # ==== 右：输出 4 象限 ====
    rx_, ry_ = 830, 160
    parts.append(rect(rx_, ry_, 710, 670, fill="#FBFCFF",
                      stroke=PURPLE_LIGHT, sw=1.5, rx=12, shadow=True))
    parts.append(icon_badge(rx_ + 30, ry_ + 20, 16, "↑", fill="url(#gradPurple)"))
    parts.append(text(rx_ + 55, ry_ + 30, "输出 OUTPUT", size=18,
                      weight=700, color=TEXT_DARK))
    parts.append(f'<rect x="{rx_+20}" y="{ry_+50}" width="40" height="3" fill="{PURPLE}" rx="1.5"/>')

    quads = [
        ("A", "原始检测", BLUE, [
            ("class_id",    "int"),
            ("class",       "str"),
            ("confidence",  "float 0-1"),
            ("bbox",        "[x1,y1,x2,y2]"),
        ]),
        ("B", "行为事件", PURPLE, [
            ("type",        "phone_use/calling/..."),
            ("label_zh",    "中文语义"),
            ("severity",    "low/med/high/crit"),
            ("duration_s",  "持续秒数"),
            ("evidence",    "判定证据字符串"),
        ]),
        ("C", "风险量化", BLUE, [
            ("risk_score",  "0–100 float"),
            ("risk_tier",   "safe / attention /"),
            ("",            "warning / danger / critical"),
            ("alert_level", "high / critical"),
            ("recommendation", "行动建议文本"),
        ]),
        ("D", "展示 / 元", PURPLE, [
            ("driver_present", "bool"),
            ("camera_ok",   "bool"),
            ("frame_id",    "int"),
            ("timestamp",   "float (echo)"),
            ("latency_ms",  "处理延迟"),
        ]),
    ]
    qw, qh = 340, 290
    for i, (letter, title_t, col, fields) in enumerate(quads):
        r, c = i // 2, i % 2
        qx = rx_ + 20 + c * (qw + 10)
        qy = ry_ + 80 + r * (qh + 10)
        parts.append(rect(qx, qy, qw, qh, fill=CARD, stroke=BORDER,
                          rx=10, shadow=True))
        grad = "url(#gradBlue)" if col == BLUE else "url(#gradPurple)"
        parts.append(icon_badge(qx + 24, qy + 24, 16, letter, fill=grad))
        parts.append(text(qx + 50, qy + 30, title_t, size=15,
                          weight=700, color=TEXT_DARK))
        for j, (k, v) in enumerate(fields):
            yy = qy + 70 + j * 36
            if k:
                parts.append(rect(qx + 20, yy - 22, qw - 40, 30,
                                  fill="#F9FAFF", stroke=BORDER, rx=5))
                parts.append(text(qx + 32, yy - 2, k, size=12,
                                  weight=700, color=BLUE_DARK,
                                  family=MONO_STACK))
                parts.append(text(qx + qw - 32, yy - 2, v, size=11,
                                  color=TEXT_MID, anchor="end",
                                  family=MONO_STACK))
            else:
                parts.append(text(qx + 32, yy - 2, v, size=11,
                                  color=TEXT_MID, italic=True,
                                  family=MONO_STACK))

    # ==== 底部扩展说明 ====
    parts.append(rect(60, 840, 1480, 40, fill="#EEF2FF",
                      stroke=BLUE_LIGHT, rx=8))
    parts.append(text(780, 865, "契约稳定性：新增字段需向后兼容 · "
                      "版本字段 schema_version 预留 · JSON 全 UTF-8 · "
                      "浮点保留 3 位小数",
                      size=12, color=BLUE_DARK,
                      anchor="middle", family=MONO_STACK))

    parts.append(svg_close())
    write_svg("docs/diagrams/03_io_schema.svg", "\n".join(parts))


# ==========================================================
# 04  算法优化对比
# ==========================================================

def build_optimizations():
    W, H = 1600, 900
    parts = [svg_open(W, H, "Optimizations")]
    parts.append(title_block(60, 40, "算法优化策略",
        "五大优化维度 · 从 CPU 4 FPS 到 50+ FPS 的路径"))
    parts.append(camera_deco(1420, 40))

    # 顶部概览条
    parts.append(rect(60, 140, 1480, 80, fill="#0F172A", rx=12))
    # 三个里程碑
    mile = [
        (150, "原始方案",  "60 ms",  "16 FPS", "#94A3B8"),
        (580, "标准优化",  "35 ms",  "28 FPS", BLUE),
        (1020, "极致加速", "18 ms",  "50+ FPS", GREEN),
    ]
    for mx, lbl, lat, fps, col in mile:
        parts.append(circle(mx, 180, 22, fill=col, stroke="#FFFFFF", sw=3))
        parts.append(text(mx, 186, "◆", size=16, color="#FFFFFF",
                          anchor="middle", weight=700))
        parts.append(text(mx, 155, lbl, size=13, color="#E2E8F0",
                          weight=700, anchor="middle"))
        parts.append(text(mx, 210, f"{lat}  |  {fps}", size=13,
                          color=col, weight=700, anchor="middle",
                          family=MONO_STACK))
    # 连接线
    parts.append(f'<line x1="172" y1="180" x2="558" y2="180" '
                 f'stroke="{BLUE}" stroke-width="3" stroke-linecap="round"/>')
    parts.append(f'<line x1="602" y1="180" x2="998" y2="180" '
                 f'stroke="{GREEN}" stroke-width="3" stroke-linecap="round"/>')
    # 箭头末端
    parts.append(text(1360, 186, "目标 >15 FPS 实时", size=13,
                      color="#CBD5E1", weight=600, anchor="middle"))

    # 5 个优化卡片 (2×3 最后一格空用于汇总)
    opts = [
        ("01", "推理分辨率下降", BLUE,
         "imgsz 640 → 384",
         "–42% 延迟",
         ["640×640: 62 ms",
          "384×384: 36 ms",
          "256×256: 30 ms",
          "小物体召回略降 1-2%"]),
        ("02", "驾驶员 Crop", PURPLE,
         "人体 bbox + 10% padding",
         "smoking 专用模型 –20% 延迟",
         ["避免整图跑大模型",
          "crop 尺寸 ≈ 原 40%",
          "保留上下文 padding",
          "精度不降反升（背景少）"]),
        ("03", "跳帧策略", BLUE,
         "infer_every=N 中间帧复用",
         "FPS 近线性 ×N",
         ["N=1 每帧精细",
          "N=2 显示端 50+ FPS",
          "N=3 约 70 FPS",
          "吸烟等慢动作完全兼容"]),
        ("04", "低光 CLAHE", PURPLE,
         "YUV Y 通道自适应均衡",
         "暗场景召回 +15%",
         ["只在 mean<80 触发",
          "tile 8×8 clip=3.0",
          "车内夜视关键",
          "开销仅 +3 ms"]),
        ("05", "统一多类模型", BLUE,
         "3 模型 → 1 个 yolov8n",
         "–70% 总延迟",
         ["原: YOLO + v11m smoke + seatbelt",
          "新: 单个 6MB 模型覆盖",
          "参数减 7 倍",
          "精度相当，速度大幅提升"]),
    ]
    card_w, card_h = 290, 240
    gap = 8
    sx = 60; sy = 250
    for i, (num, title_t, col, sub1, sub2, blist) in enumerate(opts):
        r, c = i // 3, i % 3
        x = sx + c * (card_w + gap) + (r * (card_w + gap) if r else 0)
        # 两行布局
        x = sx + c * (card_w + gap) if r == 0 else sx + (c + 0) * (card_w + gap) + 150
        y = sy + r * (card_h + 20)
        parts.append(rect(x, y, card_w, card_h, fill=CARD, stroke=BORDER,
                          rx=12, shadow=True))
        grad = "url(#gradBlue)" if col == BLUE else "url(#gradPurple)"
        parts.append(icon_badge(x + 30, y + 30, 20, num, fill=grad, size=12))
        parts.append(text(x + 60, y + 36, title_t, size=15,
                          weight=700, color=TEXT_DARK))
        parts.append(f'<rect x="{x+14}" y="{y+55}" width="32" height="3" '
                     f'fill="{col}" rx="1.5"/>')
        # 方案
        parts.append(text(x + 16, y + 85, sub1, size=12, weight=700,
                          color=BLUE_DARK, family=MONO_STACK))
        # 增益 chip
        parts.append(chip(x + 16, y + 96, 200, 22, sub2,
                          fill=GREEN, text_color="#FFFFFF", size=11))
        # 条目
        for j, line in enumerate(blist):
            yy = y + 140 + j * 20
            parts.append(circle(x + 22, yy - 4, 2.5, fill=col))
            parts.append(text(x + 32, yy, line, size=10.5,
                              color=TEXT_MID, family=MONO_STACK))

    # 右下：优化数学 / 公式
    fx, fy = 970, 730
    parts.append(rect(fx, fy, 570, 120, fill="url(#gradBlueFill)",
                      stroke=BLUE, rx=12, shadow=True))
    parts.append(text(fx + 20, fy + 26, "🔢 综合 FPS 近似公式",
                      size=14, weight=700, color=BLUE_DARK))
    parts.append(formula_box(fx + 20, fy + 38, 530, 36,
        "FPS_eff  ≈  N / ( t_det + t_pose + t_rule + t_draw )",
        color_bg="#FFFFFF", color_border=BLUE, size=14))
    parts.append(text(fx + 20, fy + 93,
        "优化 = 减小 t_*  +  提升 N（跳帧）  +  减少串行模型数",
        size=12, color=TEXT_MID, family=MONO_STACK))

    parts.append(svg_close())
    write_svg("docs/diagrams/04_optimizations.svg", "\n".join(parts))


# ==========================================================
# 05  时间窗口 + 加权风险评分
# ==========================================================

def build_risk_scoring():
    W, H = 1600, 900
    parts = [svg_open(W, H, "Risk Scoring")]
    parts.append(title_block(60, 40, "时间窗口风险评估与评分",
        "多行为并发 · 加权互补合成 · 持续时长加成 · 5 档 tier 映射"))
    parts.append(camera_deco(1420, 40))

    # ===== 上半部分：时间轴 + 滑窗 =====
    tx, ty = 60, 150
    parts.append(rect(tx, ty, 1480, 320, fill=CARD, stroke=BORDER,
                      rx=12, shadow=True))
    parts.append(text(tx + 20, ty + 30, "① 时序滑窗与 duration_s 累计",
                      size=17, weight=700, color=TEXT_DARK))
    parts.append(text(tx + 20, ty + 50, "5 帧滑窗多数投票 · 抑制单帧抖动 · "
                      "激活后开始持续计时",
                      size=12, color=TEXT_MID))

    # 帧轴
    axis_y = ty + 110
    parts.append(f'<line x1="{tx+80}" y1="{axis_y}" x2="{tx+1440}" y2="{axis_y}" '
                 f'stroke="{TEXT_LIGHT}" stroke-width="1.5"/>')
    parts.append(text(tx + 70, axis_y + 4, "frame", size=12,
                      color=TEXT_LIGHT, anchor="end", family=MONO_STACK))

    # 示例：20 帧的检测序列
    #          0 1 2 3 4 5 6 7 8 9  ...
    det_seq = [0,0,1,1,0,1,1,1,1,1,1,0,1,1,1,1,0,0,0,0]
    state_seq = [0]*len(det_seq)
    # 模拟激活状态
    window = []
    active = False
    start_frame = None
    for i, d in enumerate(det_seq):
        window.append(d)
        if len(window) > 5:
            window.pop(0)
        cnt = sum(window)
        if not active and cnt >= 3:
            active = True
            start_frame = i - 2
        elif active and (len(window) - cnt) >= 3:
            active = False
        state_seq[i] = 1 if active else 0

    # 画 frame 检测 + 激活状态
    cell_w = 65
    start_x = tx + 90
    for i, (d, s) in enumerate(zip(det_seq, state_seq)):
        cx = start_x + i * cell_w
        # 检测状态（上方）
        d_col = PURPLE if d else "#E2E8F0"
        parts.append(rect(cx, axis_y - 45, cell_w - 4, 20, fill=d_col, rx=3))
        parts.append(text(cx + (cell_w-4)/2, axis_y - 32,
                          "●" if d else "○", size=12,
                          color="#FFFFFF" if d else TEXT_LIGHT,
                          anchor="middle", weight=700))
        # 激活状态（下方）
        s_col = GREEN if s else "#F3F4F6"
        parts.append(rect(cx, axis_y + 8, cell_w - 4, 20, fill=s_col, rx=3))
        if s:
            parts.append(text(cx + (cell_w-4)/2, axis_y + 23,
                              "ON", size=11, color="#FFFFFF",
                              anchor="middle", weight=700,
                              family=MONO_STACK))
        # 帧号
        parts.append(text(cx + (cell_w-4)/2, axis_y + 50, str(i),
                          size=10, color=TEXT_LIGHT, anchor="middle",
                          family=MONO_STACK))

    # 两列标签
    parts.append(text(tx + 74, axis_y - 32, "检测", size=11,
                      color=TEXT_MID, anchor="end", family=MONO_STACK))
    parts.append(text(tx + 74, axis_y + 23, "激活", size=11,
                      color=GREEN, anchor="end", weight=700,
                      family=MONO_STACK))

    # duration 标注：找到激活段最左/最右帧
    active_indices = [i for i, s in enumerate(state_seq) if s == 1]
    if active_indices:
        real_start = active_indices[0]
        real_end = active_indices[-1]
        fx1 = start_x + real_start * cell_w + 20
        fx2 = start_x + real_end * cell_w + (cell_w - 4)
        parts.append(f'<path d="M {fx1} {axis_y+80} L {fx2} {axis_y+80}" '
                     f'stroke="{ORANGE}" stroke-width="2" '
                     f'marker-start="url(#arrowBlue)" '
                     f'marker-end="url(#arrowBlue)"/>')
        dur = (real_end - real_start + 1) * 0.04
        parts.append(text((fx1+fx2)/2, axis_y + 104,
                          f"duration_s ≈ {dur:.2f}s  "
                          f"（假设 25 FPS · 共 {real_end - real_start + 1} 帧激活）",
                          size=12, color=ORANGE, weight=700,
                          anchor="middle", family=MONO_STACK))

    # 规则说明右上
    parts.append(rect(tx + 1180, ty + 75, 290, 140, fill="#F9FAFF",
                      stroke=BORDER, rx=8))
    parts.append(text(tx + 1195, ty + 95, "触发条件", size=12,
                      weight=700, color=PURPLE_DARK))
    parts.append(text(tx + 1195, ty + 117, "激活：窗内命中 ≥ 3 / 5",
                      size=11, color=TEXT_MID, family=MONO_STACK))
    parts.append(text(tx + 1195, ty + 135, "失活：窗内未中 ≥ 3 / 5",
                      size=11, color=TEXT_MID, family=MONO_STACK))
    parts.append(text(tx + 1195, ty + 160, "⬆ 触发延迟 2-3 帧",
                      size=11, color=BLUE_DARK, family=MONO_STACK, weight=700))
    parts.append(text(tx + 1195, ty + 180, "⬇ 误报抑制 > 60%",
                      size=11, color=GREEN, family=MONO_STACK, weight=700))

    # ===== 下半左：权重表 =====
    wx, wy = 60, 490
    parts.append(rect(wx, wy, 490, 380, fill=CARD, stroke=BORDER,
                      rx=12, shadow=True))
    parts.append(text(wx + 20, wy + 30, "② 行为基础风险权重 w",
                      size=16, weight=700, color=TEXT_DARK))
    parts.append(text(wx + 20, wy + 50, "人工校准 · 符合行业 DMS 告警层级",
                      size=11, color=TEXT_MID))

    weights = [
        ("no_driver",         1.00, RED),
        ("calling",           0.80, "#DC2626"),
        ("phone_use",         0.75, "#EA580C"),
        ("no_seatbelt",       0.70, ORANGE),
        ("hands_off_wheel",   0.65, "#D97706"),
        ("smoking",           0.45, "#CA8A04"),
        ("lens_covered",      0.40, "#84CC16"),
        ("abnormal_posture",  0.30, GREEN),
    ]
    for i, (k, w, col) in enumerate(weights):
        yy = wy + 85 + i * 32
        parts.append(text(wx + 30, yy + 16, k, size=12, weight=600,
                          color=TEXT_DARK, family=MONO_STACK))
        # bar
        bar_w = int(300 * w)
        parts.append(rect(wx + 220, yy + 2, 240, 20, fill="#F3F4F6", rx=3))
        parts.append(rect(wx + 220, yy + 2, bar_w, 20, fill=col, rx=3))
        parts.append(text(wx + 220 + bar_w - 5, yy + 16, f"{w:.2f}",
                          size=11, weight=700, color="#FFFFFF",
                          anchor="end", family=MONO_STACK))

    # ===== 下半中：核心公式 =====
    cx, cy = 570, 490
    parts.append(rect(cx, cy, 500, 380, fill="url(#gradPurpleFill)",
                      stroke=PURPLE, rx=12, shadow=True))
    parts.append(text(cx + 20, cy + 30, "③ 加权互补合成公式",
                      size=16, weight=700, color=PURPLE_DARK))
    parts.append(text(cx + 20, cy + 50, "概率互补 · 避免单纯累加越界",
                      size=11, color=TEXT_MID))

    parts.append(formula_box(cx + 20, cy + 78, 460, 60,
        "score = 100 × ( 1 − ∏(1 − w_i · c_i · τ_i) )",
        color_bg="#FFFFFF", color_border=PURPLE, size=16,
        text_color=PURPLE_DARK))

    # 变量说明
    var_desc = [
        ("w_i",   "行为基础权重 (左表)", PURPLE_DARK),
        ("c_i",   "YOLO 检出置信度 0-1", BLUE_DARK),
        ("τ_i",   "时长加成 = min(1.6, 1 + 0.06 × min(dur,10))", GREEN),
        ("∏",     "所有激活行为连乘 (互补概率)", ORANGE),
    ]
    for i, (sym, desc, col) in enumerate(var_desc):
        yy = cy + 170 + i * 32
        parts.append(rect(cx + 20, yy - 14, 40, 30,
                          fill=col, rx=6))
        parts.append(text(cx + 40, yy + 6, sym, size=14, weight=700,
                          color="#FFFFFF", anchor="middle",
                          family=MONO_STACK))
        parts.append(text(cx + 72, yy + 6, desc, size=12,
                          color=TEXT_MID, family=MONO_STACK))

    # 示例
    parts.append(rect(cx + 20, cy + 310, 460, 60, fill="#FFFFFF",
                      stroke=BORDER, rx=8))
    parts.append(text(cx + 36, cy + 328, "示例：calling(0.87, 2.4s) + no_seatbelt(0.75, 3.5s)",
                      size=11, color=TEXT_MID, family=MONO_STACK))
    parts.append(text(cx + 36, cy + 352,
                      "→ 1 − (1 − 0.80·0.87·1.14) × (1 − 0.70·0.75·1.21)",
                      size=11, color=PURPLE_DARK, weight=700,
                      family=MONO_STACK))
    parts.append(text(cx + 460, cy + 352, "= 76.3",
                      size=13, weight=700, color=RED,
                      anchor="end", family=MONO_STACK))

    # ===== 下半右：5 档 tier =====
    rx_, ry_ = 1090, 490
    parts.append(rect(rx_, ry_, 450, 380, fill=CARD, stroke=BORDER,
                      rx=12, shadow=True))
    parts.append(text(rx_ + 20, ry_ + 30, "④ 风险等级映射 & 预警",
                      size=16, weight=700, color=TEXT_DARK))
    parts.append(text(rx_ + 20, ry_ + 50, "动态颜色 + 多通道告警",
                      size=11, color=TEXT_MID))

    tiers = [
        (  0,  10, "safe",      GREEN,   "无告警"),
        ( 10,  30, "attention", "#FBBF24", "屏幕柔和提示"),
        ( 30,  60, "warning",   ORANGE,  "仪表盘图标 + 柔和蜂鸣"),
        ( 60,  85, "danger",    "#DC2626", "语音警告 + 仪表闪烁"),
        ( 85, 100, "critical",  RED,     "强音警告 + 方向盘震动 + 限速"),
    ]
    for i, (lo, hi, name, col, action) in enumerate(tiers):
        yy = ry_ + 85 + i * 52
        parts.append(rect(rx_ + 20, yy, 410, 44, fill="#F9FAFF",
                          stroke=BORDER, rx=6))
        # 等级条带
        parts.append(rect(rx_ + 20, yy, 10, 44, fill=col, rx=5))
        parts.append(text(rx_ + 40, yy + 18, f"{lo}–{hi}", size=12,
                          weight=700, color=TEXT_DARK, family=MONO_STACK))
        parts.append(text(rx_ + 40, yy + 36, name.upper(), size=13,
                          weight=700, color=col, family=MONO_STACK))
        parts.append(text(rx_ + 140, yy + 28, action, size=11,
                          color=TEXT_MID))

    # 升级规则（底部小提示）
    parts.append(rect(rx_ + 20, ry_ + 350, 410, 22, fill="#FEF3C7", rx=4))
    parts.append(text(rx_ + 30, ry_ + 365,
                      "⚡ high 级连续 >2s → danger；>5s → critical",
                      size=11, color="#92400E", weight=700))

    parts.append(svg_close())
    write_svg("docs/diagrams/05_risk_scoring.svg", "\n".join(parts))


# ==========================================================

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build_yolo_architecture()
    build_inference_pipeline()
    build_io_schema()
    build_optimizations()
    build_risk_scoring()
    print("\n[done] 5 SVG diagrams generated in docs/diagrams/")
