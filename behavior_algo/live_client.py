"""
live_client.py — DMS 行为识别算法 A · 本地实时客户端

优势（相对 Gradio 流）：
  - 无浏览器 JPEG 编解码 + HTTP 传输，延迟降至 ~推理耗时
  - 异步读帧线程，I/O 与推理重叠
  - 可跳帧（--infer-every N）提升实测 FPS
  - 推理分辨率可调（--imgsz 384/256）进一步加速

默认按键:
  q / ESC   退出
  s         截图保存到 outputs/
  r         重置时序滑窗
  p         暂停 / 恢复推理
  + / -     动态调整跳帧步长

运行示例:
  # 默认（imgsz=384，每帧都推）
  python live_client.py

  # 激进提速：imgsz=256 + 每 2 帧推一次
  python live_client.py --imgsz 256 --infer-every 2

  # GPU
  python live_client.py --device 0 --imgsz 640
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 环境变量必须在 cv2 前
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

import cv2
import numpy as np

from behavior_detector import BehaviorDetector, CameraStream, open_capture


BANNER = r"""
 ____    __  __   ____      __  __  ___     ____
|  _ \  |  \/  | / ___|    / / |  \/  |/ _ \   / ___|
| | | | | |\/| | \___ \   / /  | |\/| | | | | | |   _
| |_| | | |  | |  ___) | / /   | |  | | |_| | | |__| |
|____/  |_|  |_| |____/ /_/    |_|  |_|\___/   \____|

   Behavior Algorithm A — Live Client (OpenCV)
"""


def check_camera_permission(index: int) -> bool:
    """
    Windows 上尝试打开 + 读一帧；若失败给出权限提示。
    返回 True=可用，False=不可用（并已打印说明）。
    """
    print(f"[check] 请求摄像头 index={index} ...")
    cap, desc = open_capture(index)
    if cap is None:
        print("\n[!] 摄像头不可用。可能原因与处理:")
        print("    1) 被其他程序占用 (Teams / Zoom / 浏览器 / 旧 Python 进程)")
        print("       → 关闭它们，再运行本脚本")
        print("    2) Windows 隐私权限被拒绝")
        print("       → 设置 → 隐私和安全性 → 相机")
        print("         · '允许访问此设备上的相机' 打开")
        print("         · '允许桌面应用访问你的相机' 打开")
        print("    3) 摄像头驱动故障")
        print("       → 设备管理器 → 照相机 → 右键更新 / 卸载后重启")
        print(f"\n    详细: {desc}\n")
        return False

    # 读一帧验证
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"[!] 摄像头打开了但首帧读取失败: {desc}")
        return False
    print(f"[ok] {desc}   首帧形状 {frame.shape}\n")
    return True


def parse_args():
    ap = argparse.ArgumentParser(description="DMS 行为识别算法 A 本地实时客户端")
    ap.add_argument("--source", default="0",
                    help="摄像头 index 或 视频/RTSP 路径（默认 0）")
    ap.add_argument("--yolo",  default="models/yolov8n.pt")
    ap.add_argument("--pose",  default="models/yolov8n-pose.pt")
    ap.add_argument("--seatbelt", default="models/seatbelt.pt")
    ap.add_argument("--smoking",  default="models/smoking.pt")
    ap.add_argument("--unified",  default="models/unified.pt",
                    help="训练好的 8 类统一模型；存在则覆盖 smoking/seatbelt 独立模型")
    ap.add_argument("--device", default="cpu",
                    help="cpu / 0 / cuda:0")
    ap.add_argument("--imgsz", type=int, default=384,
                    help="推理分辨率 (256/320/384/512/640)。越小越快")
    ap.add_argument("--phone-conf", type=float, default=0.20,
                    help="手机类别独立置信度阈值 (更低=更敏感)")
    ap.add_argument("--phone-recheck-imgsz", type=int, default=640,
                    help="驾驶员 crop 内手机重检的 imgsz (0=禁用)")
    ap.add_argument("--no-low-light-enhance", action="store_true",
                    help="关闭低光 CLAHE 增强")
    ap.add_argument("--style", choices=["debug", "monitor"], default="monitor",
                    help="可视化风格：debug(分析用) / monitor(演示用，默认)")
    ap.add_argument("--no-trackbars", action="store_true",
                    help="关闭阈值滑块（主窗口更干净，用快捷键替代）")
    ap.add_argument("--infer-every", type=int, default=1,
                    help="每 N 帧推理一次，中间帧复用上次结果 (1=每帧)")
    ap.add_argument("--width",  type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--target-fps", type=int, default=30)
    ap.add_argument("--save-video", default=None,
                    help="输出 MP4 路径")
    ap.add_argument("--log-json",  default=None,
                    help="按行追加每帧 JSON 到该路径")
    ap.add_argument("--no-pre-check", action="store_true",
                    help="跳过摄像头权限预检")
    return ap.parse_args()


def main():
    print(BANNER)
    args = parse_args()

    src_is_cam = args.source.isdigit()

    # 1. 权限预检（仅摄像头）
    if src_is_cam and not args.no_pre_check:
        if not check_camera_permission(int(args.source)):
            sys.exit(2)

    # 2. 加载检测器
    det = BehaviorDetector(
        yolo_weights=args.yolo,
        pose_weights=args.pose,
        seatbelt_weights=args.seatbelt if Path(args.seatbelt).exists() else None,
        smoking_weights=args.smoking if Path(args.smoking).exists() else None,
        unified_weights=args.unified if Path(args.unified).exists() else None,
        device=args.device,
        imgsz=args.imgsz,
        phone_conf=args.phone_conf,
        phone_recheck_imgsz=args.phone_recheck_imgsz if args.phone_recheck_imgsz > 0 else args.imgsz,
        low_light_enhance=not args.no_low_light_enhance,
    )

    # 3. 打开源（摄像头用异步流，其他用同步 VideoCapture）
    stream = None
    sync_cap = None
    if src_is_cam:
        stream = CameraStream(
            int(args.source),
            width=args.width, height=args.height,
            target_fps=args.target_fps,
        ).start()
        print(f"[open] {stream.desc} (threaded, {args.width}x{args.height})")
    else:
        sync_cap, desc = open_capture(args.source)
        if sync_cap is None:
            print(f"[!] {desc}")
            sys.exit(2)
        print(f"[open] {desc}")

    def read_frame():
        if stream is not None:
            return stream.read()
        ok, f = sync_cap.read()
        return f if ok else None

    # 4. 预置状态
    writer = None
    json_fp = None
    if args.log_json:
        json_fp = open(args.log_json, "a", encoding="utf-8")

    fid = 0
    paused = False
    infer_every = max(1, args.infer_every)
    last_result = None
    fps_deque = []
    last_t = time.time()

    # ---------- 运行时窗口 ----------
    WIN = "DMS Behavior Algo A"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    control_panel = None
    if not args.no_trackbars:
        from control_panel import ControlPanel, Slider
        # 闭包持有状态；skip 在外部 infer_every 变量里
        def _get_skip(): return infer_every
        def _set_skip(v):
            nonlocal infer_every
            infer_every = max(1, int(v))

        sliders = [
            Slider("conf",        lambda: det.conf * 100,
                   lambda v: setattr(det, "conf", max(0.05, v/100)),
                   5, 95, step=1, fmt="{:.0f}%"),
            Slider("phone conf",  lambda: det.phone_conf * 100,
                   lambda v: setattr(det, "phone_conf", max(0.05, v/100)),
                   5, 95, step=1, fmt="{:.0f}%"),
            Slider("skip frames", _get_skip, _set_skip,
                   1, 10, step=1, fmt="1/{:.0f}"),
            Slider("imgsz",       lambda: det.imgsz,
                   lambda v: setattr(det, "imgsz",
                                     max(160, (int(v) // 32) * 32)),
                   160, 960, step=32, fmt="{:.0f}"),
            Slider("low light",   lambda: 1.0 if det.low_light_enhance else 0.0,
                   lambda v: setattr(det, "low_light_enhance", v > 0.5),
                   0, 1, step=1, fmt="{:.0f}"),
        ]
        control_panel = ControlPanel(sliders)

    print("[ctrl] q/ESC 退出  s 截图  r 重置时序  p 暂停  + - 调整跳帧")
    if args.no_trackbars:
        print("[ctrl] --no-trackbars 模式, 运行时仅支持 +/- 调节跳帧")
    else:
        print("[ctrl] 旁边的 'DMS CONTROLS' 面板可拖动滑块 "
              "(conf / phone_conf / skip / imgsz / low_light)")
    print()
    Path("outputs").mkdir(exist_ok=True)

    try:
        while True:
            frame = read_frame()
            if frame is None:
                # 视频文件结束 / 摄像头尚未给出首帧
                if stream is None:
                    print("[end] 视频读取结束")
                    break
                time.sleep(0.01)
                continue

            t_infer_start = time.time()
            if paused:
                res = last_result or {"frame_id": fid, "timestamp": time.time(),
                                      "latency_ms": 0.0, "behaviors": [],
                                      "alert_level": "none",
                                      "recommendation": "已暂停",
                                      "driver_present": True, "camera_ok": True}
            elif (fid % infer_every) == 0:
                res = det.predict(frame, frame_id=fid, timestamp=time.time())
                last_result = res
                if json_fp:
                    json_fp.write(json.dumps(res, ensure_ascii=False) + "\n")
                    json_fp.flush()
            else:
                # 跳帧复用：仅更新 frame_id/timestamp，behaviors 不变
                if last_result is None:
                    res = det.predict(frame, frame_id=fid, timestamp=time.time())
                    last_result = res
                else:
                    res = dict(last_result)
                    res["frame_id"] = fid
                    res["timestamp"] = round(time.time(), 3)
                    res["latency_ms"] = 0.0  # 复用帧无新推理

            # 5. 端到端 FPS（含显示）
            now = time.time()
            fps_deque.append(1.0 / max(1e-6, now - last_t))
            last_t = now
            if len(fps_deque) > 30:
                fps_deque.pop(0)
            e2e_fps = sum(fps_deque) / len(fps_deque)

            vis = det.visualize(frame, res, style=args.style)
            if args.style == "debug":
                info = f"e2e {e2e_fps:4.1f} FPS  imgsz={args.imgsz}  skip=1/{infer_every}"
                cv2.putText(vis, info, (vis.shape[1] - 320, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            elif args.style == "monitor" and infer_every > 1:
                # monitor 模式仅在跳帧时补充显示 skip 信息
                cv2.putText(vis, f"skip 1/{infer_every}",
                            (vis.shape[1] - 110, vis.shape[0] - 78),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (140, 140, 140), 1, cv2.LINE_AA)
            if paused:
                ph, pw = vis.shape[:2]
                overlay = vis.copy()
                cv2.rectangle(overlay, (pw // 2 - 80, 50),
                              (pw // 2 + 80, 90), (18, 18, 22), -1)
                vis = cv2.addWeighted(overlay, 0.6, vis, 0.4, 0)
                cv2.putText(vis, "PAUSED", (pw // 2 - 55, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 220, 255), 2,
                            cv2.LINE_AA)

            # 6. 写视频
            if args.save_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(
                        args.save_video,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        20.0, (w, h))
                writer.write(vis)

            cv2.imshow(WIN, vis)
            if control_panel is not None:
                control_panel.render(
                    extra_info=f"e2e {e2e_fps:4.1f} FPS  latency {res['latency_ms']:.0f}ms")
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):     # q / ESC
                break
            elif key == ord("s"):
                shot = Path("outputs") / f"shot_{int(time.time())}.jpg"
                cv2.imwrite(str(shot), vis)
                print(f"[shot] {shot}")
            elif key == ord("r"):
                det.smoother = det.smoother.__class__(window=5)
                last_result = None
                print("[reset] 时序滑窗已清")
            elif key == ord("p"):
                paused = not paused
                print(f"[pause] {paused}")
            elif key in (ord("+"), ord("=")):
                infer_every = min(10, infer_every + 1)
                print(f"[skip] 每 {infer_every} 帧推理一次")
            elif key in (ord("-"), ord("_")):
                infer_every = max(1, infer_every - 1)
                print(f"[skip] 每 {infer_every} 帧推理一次")

            fid += 1

    except KeyboardInterrupt:
        print("\n[Ctrl-C] 退出")
    finally:
        if stream is not None:
            stream.stop()
        if sync_cap is not None:
            sync_cap.release()
        if writer is not None:
            writer.release()
        if json_fp is not None:
            json_fp.close()
        cv2.destroyAllWindows()
        print("[bye]")


if __name__ == "__main__":
    main()
