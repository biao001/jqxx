"""
gradio_demo.py — DMS 行为识别算法 A · Web Demo

启动: python gradio_demo.py
浏览器访问: http://127.0.0.1:7860

支持三种输入:
  - 单张图片 (JPG/PNG)
  - 视频文件 (MP4/AVI)
  - 实时摄像头

输出:
  - 可视化结果图/视频
  - 结构化 JSON 告警
  - 指标面板
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from behavior_detector import BehaviorDetector


# ---------- 单例加载 ----------

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"

def _resolve(name: str) -> str:
    p = MODELS_DIR / name
    return str(p) if p.exists() else name

UNIFIED_PT = MODELS_DIR / "unified.pt"
BASE_PT = PROJECT_ROOT.parent / "yolov8n.pt"

DETECTOR = BehaviorDetector(
    unified_weights=str(UNIFIED_PT) if UNIFIED_PT.exists() else None,
    base_weights=str(BASE_PT) if BASE_PT.exists() else "yolov8n.pt",
    device="cpu",
    conf=0.30,
    imgsz=448,
    temporal_window=5,
)


# ---------- 处理函数 ----------

def process_image(img_rgb: np.ndarray):
    if img_rgb is None:
        return None, "{}", "请上传图片"
    frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    res = DETECTOR.predict(frame, frame_id=0, timestamp=time.time())
    vis = DETECTOR.visualize(frame, res)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    summary = _summary(res)
    return vis_rgb, json.dumps(res, ensure_ascii=False, indent=2), summary


def process_video(video_path: str, progress=gr.Progress()):
    if not video_path:
        return None, "{}", "请上传视频"
    DETECTOR.smoother = DETECTOR.smoother.__class__(window=5)  # 重置时序状态

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "{}", "无法打开视频"

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(out_path,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    all_results = []
    fid = 0
    alert_hist = {}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = fid / fps
            res = DETECTOR.predict(frame, frame_id=fid, timestamp=ts)
            vis = DETECTOR.visualize(frame, res)
            writer.write(vis)
            all_results.append(res)
            for b in res["behaviors"]:
                alert_hist[b["type"]] = alert_hist.get(b["type"], 0) + 1
            fid += 1
            if total > 0:
                progress((fid / total), desc=f"处理中 {fid}/{total}")
    finally:
        cap.release()
        writer.release()

    summary = _video_summary(all_results, alert_hist, fps)
    last_json = all_results[-1] if all_results else {}
    return out_path, json.dumps(last_json, ensure_ascii=False, indent=2), summary


def process_webcam_frame(img_rgb):
    """摄像头单帧处理（单张抓拍模式用）"""
    return process_image(img_rgb)


# ---------- 实时流式处理 ----------

# 全局帧计数器（流式模式复用时序滑窗）
_STREAM_FID = {"n": 0, "reset_at": 0.0}


def reset_stream():
    """用户点击重置按钮时调用，清空时序滑窗"""
    DETECTOR.smoother = DETECTOR.smoother.__class__(window=5)
    _STREAM_FID["n"] = 0
    _STREAM_FID["reset_at"] = time.time()
    return None, "{}", "**已重置时序状态**"


def stream_predict(img_rgb):
    """
    Gradio 流式回调：浏览器持续送帧进来。
    返回 (可视化帧, JSON 字符串, Markdown 摘要)
    """
    if img_rgb is None:
        return None, "{}", ""
    frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    fid = _STREAM_FID["n"]
    _STREAM_FID["n"] += 1
    res = DETECTOR.predict(frame, frame_id=fid, timestamp=time.time())
    vis = DETECTOR.visualize(frame, res)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return vis_rgb, json.dumps(res, ensure_ascii=False, indent=2), _summary(res)


# ---------- 汇总 ----------

def _summary(res: dict) -> str:
    if not res or "behaviors" not in res:
        return ""
    lvl = res.get("alert_level", "none")
    lat = res.get("latency_ms", 0)
    fps = 1000.0 / max(1e-3, lat)
    lines = [f"**预警等级:** {lvl.upper()}",
             f"**延迟:** {lat:.1f} ms (~{fps:.1f} FPS)",
             f"**建议动作:** {res.get('recommendation', '')}",
             f"**驾驶员在位:** {res.get('driver_present', '-')}",
             f"**镜头正常:** {res.get('camera_ok', '-')}",
             ""]
    if res["behaviors"]:
        lines.append("**检出行为:**")
        for b in res["behaviors"]:
            lines.append(
                f"- `{b['type']}` ({b['label_zh']}) "
                f"置信度 {b['confidence']:.2f} "
                f"持续 {b['duration_s']}s —— {b['evidence']}"
            )
    else:
        lines.append("**检出行为:** 无")
    return "\n".join(lines)


def _video_summary(results, hist, fps):
    if not results:
        return "无结果"
    N = len(results)
    avg_lat = np.mean([r["latency_ms"] for r in results])
    levels = [r["alert_level"] for r in results]
    from collections import Counter
    lvl_cnt = Counter(levels)
    lines = [
        f"**总帧数:** {N}  |  **视频 FPS:** {fps:.1f}  |  **处理延迟均值:** {avg_lat:.1f} ms",
        "",
        "**预警等级分布:**",
    ]
    for lv, c in lvl_cnt.most_common():
        lines.append(f"- {lv}: {c} 帧 ({c/N*100:.1f}%)")
    lines.append("")
    lines.append("**行为累计帧数:**")
    if not hist:
        lines.append("- 无")
    else:
        for k, v in sorted(hist.items(), key=lambda x: -x[1]):
            lines.append(f"- `{k}`: {v} 帧")
    return "\n".join(lines)


# ---------- 接口文档 ----------

IO_DOC = """
## 输入/输出规范

### 输入
- **图像/视频帧**: `np.ndarray`，shape `(H, W, 3)`，`dtype=uint8`，BGR
- **建议分辨率**: ≥ 640×480
- **建议帧率**: 15~30 FPS

### 输出 (JSON)
```jsonc
{
  "frame_id": 1024,
  "timestamp": 1713772800.123,
  "latency_ms": 38.5,
  "behaviors": [
    {
      "type": "phone_use",              // 枚举值见下
      "label_zh": "驾驶中使用手机",
      "confidence": 0.87,
      "bbox": [x1, y1, x2, y2],         // 可选
      "severity": "high",               // low/medium/high/critical
      "duration_s": 2.4,
      "evidence": "画面内检出手机"
    }
  ],
  "alert_level": "high",                // none/low/medium/high/critical
  "recommendation": "语音警告 + 仪表盘闪烁",
  "driver_present": true,
  "camera_ok": true
}
```

### 行为类型枚举
| type | 中文 | severity |
|------|------|----------|
| `phone_use` | 驾驶中使用手机 | high |
| `calling` | 驾驶中打电话 | high |
| `smoking` | 驾驶中吸烟 | medium |
| `no_seatbelt` | 未系安全带 | high |
| `hands_off_wheel` | 双手离开方向盘 | high |
| `abnormal_posture` | 驾驶姿势异常 | low |
| `no_driver` | 驾驶位无人 | critical |
| `lens_covered` | 摄像头被遮挡 | medium |

### Python 调用示例
```python
from behavior_detector import BehaviorDetector
det = BehaviorDetector(yolo_weights="models/yolov8n.pt",
                       pose_weights="models/yolov8n-pose.pt")
result = det.predict(frame, frame_id=0, timestamp=time.time())
```
"""


# ---------- UI ----------

with gr.Blocks(title="DMS · 行为识别算法 A") as demo:
    gr.Markdown("# DMS · 行为识别算法 A (3 号分工)")
    gr.Markdown(
        "监测维度：**手机使用 / 打电话 / 吸烟 / 未系安全带 / 双手离盘 / "
        "姿势异常 / 无驾驶员 / 镜头遮挡**"
    )

    with gr.Tabs():
        with gr.TabItem("📷 图片推理"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(label="输入图片", type="numpy", height=360)
                    btn_img = gr.Button("🚀 运行检测", variant="primary")
                with gr.Column():
                    img_out = gr.Image(label="可视化结果", height=360)
            with gr.Row():
                md_img = gr.Markdown(label="告警摘要")
            with gr.Accordion("📄 原始 JSON 输出", open=False):
                json_img = gr.Code(language="json", label="behavior_alert")

            btn_img.click(process_image, img_in, [img_out, json_img, md_img])

        with gr.TabItem("🎬 视频推理"):
            with gr.Row():
                with gr.Column():
                    video_in = gr.Video(label="输入视频")
                    btn_vid = gr.Button("🚀 运行检测", variant="primary")
                with gr.Column():
                    video_out = gr.Video(label="可视化结果")
            md_vid = gr.Markdown()
            with gr.Accordion("📄 末帧 JSON 输出", open=False):
                json_vid = gr.Code(language="json")
            btn_vid.click(process_video, video_in, [video_out, json_vid, md_vid])

        with gr.TabItem("🎥 摄像头（逐帧抓拍）"):
            gr.Markdown("点按钮抓一帧送入算法。需要持续实时可切到下一个 Tab。")
            with gr.Row():
                with gr.Column():
                    cam_in = gr.Image(label="摄像头捕获", sources=["webcam"],
                                      type="numpy", height=360)
                    btn_cam = gr.Button("🚀 检测当前帧", variant="primary")
                with gr.Column():
                    cam_out = gr.Image(label="结果", height=360)
            md_cam = gr.Markdown()
            with gr.Accordion("📄 JSON 输出", open=False):
                json_cam = gr.Code(language="json")
            btn_cam.click(process_webcam_frame, cam_in,
                          [cam_out, json_cam, md_cam])

        with gr.TabItem("🔴 实时流"):
            gr.Markdown(
                "浏览器持续推流，CPU 实测 **15–25 FPS**；若卡顿可降低分辨率。"
                "切换 Tab 会自动暂停。"
            )
            with gr.Row():
                with gr.Column():
                    stream_in = gr.Image(
                        label="摄像头（实时流）", sources=["webcam"],
                        streaming=True, type="numpy", height=360,
                    )
                    btn_reset = gr.Button("↻ 重置时序状态")
                with gr.Column():
                    stream_out = gr.Image(label="推理结果（实时）",
                                          height=360, streaming=True)
            md_stream = gr.Markdown()
            with gr.Accordion("📄 JSON 输出（最新帧）", open=False):
                json_stream = gr.Code(language="json")

            # 关键：stream 事件实现连续推理
            stream_in.stream(
                stream_predict, inputs=stream_in,
                outputs=[stream_out, json_stream, md_stream],
                stream_every=0.05,  # 最高 20 FPS 采样
                show_progress="hidden",
            )
            btn_reset.click(reset_stream, None,
                            [stream_out, json_stream, md_stream])

        with gr.TabItem("📘 接口文档"):
            gr.Markdown(IO_DOC)


if __name__ == "__main__":
    # 默认仅本机；想要局域网访问设 GRADIO_SERVER_NAME=0.0.0.0
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=server_name, server_port=server_port,
                inbrowser=False, theme=gr.themes.Soft())
