"""
detect_json.py — 把视频/摄像头的每帧动作识别结果输出为 JSON

用法:
  # 摄像头 0
  python detect_json.py --source 0

  # 视频文件
  python detect_json.py --source path/to/video.mp4

  # 写到文件而不是 stdout
  python detect_json.py --source 0 --out results.ndjson

  # 用基础模型 (没训出 unified.pt 时)
  python detect_json.py --source 0 --unified ""

  # 同时打开可视化窗口
  python detect_json.py --source 0 --show

输出格式: 一行一个 JSON 对象 (ndjson)，每行对应一帧。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from behavior_detector import BehaviorDetector


ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="0/1/... 摄像头, 或 视频/图片/RTSP 路径")
    ap.add_argument("--unified", default=str(ROOT / "models" / "unified.pt"),
                    help="unified.pt 路径；传空字符串强制回退基础模型")
    ap.add_argument("--base", default=str(ROOT.parent / "yolov8n.pt"),
                    help="基础 YOLO 权重（unified.pt 缺失时使用）")
    ap.add_argument("--device", default="cpu", help="cpu / 0 / 0,1")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="-", help="输出文件路径，- 表示 stdout")
    ap.add_argument("--show", action="store_true", help="弹出可视化窗口")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="处理最多多少帧后退出 (0 = 不限制)")
    args = ap.parse_args()

    detector = BehaviorDetector(
        unified_weights=args.unified or None,
        base_weights=args.base,
        device=args.device,
        conf=args.conf,
        imgsz=args.imgsz,
    )

    # 打开视频源
    src = args.source
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[error] 无法打开视频源: {src}", file=sys.stderr)
        sys.exit(2)

    # 输出 sink
    if args.out == "-":
        sink = sys.stdout
        close_sink = False
    else:
        sink = open(args.out, "w", encoding="utf-8", buffering=1)  # 行缓冲
        close_sink = True

    try:
        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = time.time()
            result = detector.predict(frame, frame_id=frame_id, timestamp=ts)
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            sink.flush()

            if args.show:
                vis = detector.visualize(frame, result)
                cv2.imshow("DMS behavior (q to quit)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_id += 1
            if args.max_frames and frame_id >= args.max_frames:
                break
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        if close_sink:
            sink.close()


if __name__ == "__main__":
    main()
