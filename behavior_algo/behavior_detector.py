"""
behavior_detector.py
====================
DMS 项目 - 3 号分工：行为识别算法 A

输入: 单帧图像 np.ndarray (H,W,3) BGR
输出: 结构化 JSON 行为告警

目标行为:
  phone_use / calling / smoking / no_seatbelt /
  hands_off_wheel / abnormal_posture / no_driver / lens_covered
"""
from __future__ import annotations

import math
import os
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 抑制 Windows MSMF 后端告警日志（必须在 import cv2 之前）
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

import cv2
import numpy as np
from ultralytics import YOLO


# ---------- 摄像头/视频打开（多后端降级 + 摄像头索引枚举） ----------

def open_capture(source, prefer_backends=None, try_indices=(0, 1, 2),
                 warmup_reads: int = 3, verbose: bool = True):
    """
    鲁棒地打开视频源。
    - 文件/URL/RTSP 路径：直接尝试 ANY 后端
    - 整数 index：依次 DSHOW → MSMF → V4L2 → ANY；失败时枚举 0/1/2

    返回 (cap, actual_source_desc) 或 (None, "错误原因")
    """
    if prefer_backends is None:
        prefer_backends = [
            (cv2.CAP_DSHOW, "DSHOW"),   # Windows DirectShow，兼容性最好
            (cv2.CAP_MSMF,  "MSMF"),    # Windows Media Foundation
            (cv2.CAP_V4L2,  "V4L2"),    # Linux
            (cv2.CAP_ANY,   "ANY"),     # 让 OpenCV 自己挑
        ]

    # 非整数 index：直接 ANY 后端打开文件 / RTSP / 图像
    if not (isinstance(source, int) or (isinstance(source, str) and source.isdigit())):
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap, f"file: {source}"
        return None, f"无法打开文件/流: {source}"

    idx = int(source) if isinstance(source, str) else source
    candidates = [idx] + [i for i in try_indices if i != idx]

    for ci in candidates:
        for be, name in prefer_backends:
            cap = cv2.VideoCapture(ci, be)
            if not cap.isOpened():
                cap.release()
                continue
            # 有些后端 isOpened() 为 True 但首帧仍失败，做几次 warmup
            ok_once = False
            for _ in range(warmup_reads):
                ok, _ = cap.read()
                if ok:
                    ok_once = True
                    break
                time.sleep(0.05)
            if ok_once:
                if verbose:
                    print(f"[Camera] 打开成功 index={ci} backend={name}")
                return cap, f"index={ci} backend={name}"
            cap.release()
            if verbose:
                print(f"[Camera] index={ci} backend={name} 可打开但读帧失败，尝试下一个")

    return None, "所有后端/索引均失败。检查：① 摄像头是否被其他程序占用 ② 系统→隐私→相机 是否允许"


class CameraStream:
    """
    后台线程读摄像头，始终只保留最新帧；推理线程 read() 立即拿到最新帧，
    不被 I/O 阻塞。可显著降低 CPU 推理场景下的"花屏"与卡顿。
    """
    def __init__(self, source=0, width=640, height=480, target_fps=30):
        import threading
        self._threading = threading
        self.source = source
        self.cap, self.desc = open_capture(source)
        if self.cap is None:
            raise RuntimeError(f"摄像头打开失败: {self.desc}")
        # 尝试按需设置分辨率（降低硬件编码开销）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame = None
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        while not self._stopped:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        """返回最新帧的副本；若还没有则返回 None"""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._stopped = True
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self.cap.release()


# ---------- 常量 ----------

COCO_PERSON = 0
COCO_CELL_PHONE = 67

# COCO Pose 17 关键点索引
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12

SEVERITY_MAP = {
    "no_driver":        "critical",
    "phone_use":        "high",
    "calling":          "high",
    "no_seatbelt":      "high",
    "hands_off_wheel":  "high",
    "smoking":          "medium",
    "lens_covered":     "medium",
    "abnormal_posture": "low",
}

LABEL_ZH = {
    "no_driver":        "驾驶位无人",
    "phone_use":        "驾驶中使用手机",
    "calling":          "驾驶中打电话",
    "no_seatbelt":      "未系安全带",
    "hands_off_wheel":  "双手离开方向盘",
    "smoking":          "驾驶中吸烟",
    "lens_covered":     "摄像头被遮挡",
    "abnormal_posture": "驾驶姿势异常",
}

RECOMMEND = {
    "critical": "立即语音警告 + 方向盘震动 + 限速",
    "high":     "语音警告 + 仪表盘闪烁",
    "medium":   "仪表盘图标提示",
    "low":      "屏幕柔和提示",
    "none":     "正常驾驶",
}

_SEV_ORDER = {"none": -1, "low": 0, "medium": 1, "high": 2, "critical": 3}

# ---------- 风险评分（加权转化） ----------
# 每个行为的基础风险权重（0~1）
RISK_WEIGHT = {
    "no_driver":        1.00,
    "phone_use":        0.75,
    "calling":          0.80,
    "no_seatbelt":      0.70,
    "hands_off_wheel":  0.65,
    "smoking":          0.45,
    "lens_covered":     0.40,
    "abnormal_posture": 0.30,
}


def compute_risk_score(behaviors):
    """
    输入: behaviors 列表（每个含 type / confidence / duration_s）
    输出: 0~100 的综合风险分数

    公式: score = min(100, 100 * (1 - Π(1 - w_i * c_i * dur_factor)))
    - 多行为用概率互补组合（避免单纯累加超上限）
    - 持续时间越长风险越高 (1 + 0.1 * min(dur_s, 10))
    """
    if not behaviors:
        return 0.0
    prob_safe = 1.0
    for b in behaviors:
        w = RISK_WEIGHT.get(b["type"], 0.2)
        c = b.get("confidence", 0.5)
        d = b.get("duration_s", 0.0)
        dur_factor = min(1.6, 1.0 + 0.06 * min(d, 10.0))
        contrib = max(0.0, min(1.0, w * c * dur_factor))
        prob_safe *= (1.0 - contrib)
    return round(100 * (1 - prob_safe), 1)


def risk_tier(score):
    """将 0-100 映射到人读等级"""
    if score < 10:   return "safe"
    if score < 30:   return "attention"
    if score < 60:   return "warning"
    if score < 85:   return "danger"
    return "critical"


# ---------- 数据结构 ----------

@dataclass
class BehaviorEvent:
    type: str
    label_zh: str
    confidence: float
    bbox: Optional[List[int]] = None
    severity: str = "low"
    duration_s: float = 0.0
    evidence: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- 时序滑窗 ----------

class TemporalSmoother:
    """K 帧滑窗多数投票，抑制单帧抖动"""

    def __init__(self, window: int = 5, activate: int = 3, deactivate: int = 3):
        self.window = window
        self.activate = activate
        self.deactivate = deactivate
        self.history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window))
        self.active: Dict[str, bool] = {}
        self.start_ts: Dict[str, float] = {}

    def update(self, behaviors: List[str], timestamp: float) -> List[str]:
        current = set(behaviors)
        for k in set(self.history.keys()) | current:
            self.history[k].append(1 if k in current else 0)

        stable = []
        for k in list(self.history.keys()):
            count = sum(self.history[k])
            is_on = self.active.get(k, False)
            if not is_on and count >= self.activate:
                self.active[k] = True
                self.start_ts[k] = timestamp
            elif is_on and count <= (self.window - self.deactivate):
                self.active[k] = False
                self.start_ts.pop(k, None)
            if self.active.get(k, False):
                stable.append(k)
        return stable

    def duration(self, behavior: str, now: float) -> float:
        ts = self.start_ts.get(behavior)
        return 0.0 if ts is None else max(0.0, now - ts)


# ---------- 主检测器 ----------

class BehaviorDetector:
    """多分支融合行为识别器"""

    def __init__(
        self,
        yolo_weights: str = "yolov8n.pt",
        pose_weights: str = "yolov8n-pose.pt",
        seatbelt_weights: Optional[str] = None,
        smoking_weights: Optional[str] = None,
        device: str = "cpu",
        conf: float = 0.35,
        iou: float = 0.5,
        phone_conf: float = 0.20,
        imgsz: int = 384,
        phone_recheck_imgsz: int = 640,
        low_light_enhance: bool = True,
        temporal_window: int = 5,
        wheel_roi_norm: Tuple[float, float, float, float] = (0.15, 0.5, 0.85, 1.0),
        use_seatbelt_heuristic: bool = True,
        unified_weights: Optional[str] = None,
    ):
        self.device = device
        self.conf = conf
        self.iou = iou
        self.phone_conf = phone_conf
        self.imgsz = imgsz
        self.phone_recheck_imgsz = phone_recheck_imgsz
        self.low_light_enhance = low_light_enhance
        self.wheel_roi_norm = wheel_roi_norm

        print(f"[Init] 加载主检测器: {yolo_weights}  imgsz={imgsz}")
        self.yolo = YOLO(yolo_weights)
        print(f"[Init] 加载姿态估计: {pose_weights}")
        self.pose = YOLO(pose_weights)

        self.seatbelt = None
        self.use_seatbelt_heuristic = use_seatbelt_heuristic
        if seatbelt_weights and Path(seatbelt_weights).exists():
            print(f"[Init] 加载安全带模型: {seatbelt_weights}")
            self.seatbelt = YOLO(seatbelt_weights)
        elif use_seatbelt_heuristic:
            print("[Init] 未配置安全带模型，B4 启用启发式（肩-腰斜带检测，精度有限）")
        else:
            print("[Init] 未配置安全带模型，B4 检测跳过")

        self.smoking = None
        if smoking_weights and Path(smoking_weights).exists():
            print(f"[Init] 加载抽烟模型: {smoking_weights}")
            self.smoking = YOLO(smoking_weights)
        else:
            print("[Init] 未配置抽烟模型，B3 检测回退到启发式")

        # unified 多类模型（优先级最高，覆盖 smoking/seatbelt/phone/calling）
        self.unified = None
        self.unified_names = {}
        if unified_weights and Path(unified_weights).exists():
            print(f"[Init] 加载 unified 8 类模型: {unified_weights}")
            self.unified = YOLO(unified_weights)
            self.unified_names = self.unified.names or {}
            print(f"       类别: {self.unified_names}")
        else:
            print("[Init] 未配置 unified 模型，使用独立分支检测")

        self.smoother = TemporalSmoother(window=temporal_window)

        # 跳帧模式：缓存上一次 raw 检测与事件，跳过帧直接复用
        self._last_raw: List[str] = []
        self._last_events: List[BehaviorEvent] = []
        self._last_present: bool = True
        self._last_camera_ok: bool = True

    # ---------- 工具 ----------

    @staticmethod
    def _lens_covered(frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean = gray.mean()
        return lap < 10.0 or mean < 15.0

    @staticmethod
    def _enhance_low_light(frame: np.ndarray) -> np.ndarray:
        """YUV 域 CLAHE + 轻度去噪，提升暗光 / 红外摄像头下的小物体（手机）可见度"""
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        out = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return out

    @staticmethod
    def _is_low_light(frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray.mean() < 80

    @staticmethod
    def _point_in_bbox(p, b) -> bool:
        return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]

    def _wheel_roi(self, H: int, W: int) -> List[float]:
        x1, y1, x2, y2 = self.wheel_roi_norm
        return [x1 * W, y1 * H, x2 * W, y2 * H]

    # ---------- 主推理 ----------

    def predict(self, frame: np.ndarray, frame_id: int = 0,
                timestamp: Optional[float] = None) -> Dict:
        t0 = time.time()
        if timestamp is None:
            timestamp = t0
        H, W = frame.shape[:2]
        events: List[BehaviorEvent] = []
        raw: List[str] = []

        # 0. 镜头遮挡
        if self._lens_covered(frame):
            events.append(BehaviorEvent(
                "lens_covered", LABEL_ZH["lens_covered"], 1.0,
                severity=SEVERITY_MAP["lens_covered"],
                evidence="图像方差/均值过低"))
            raw.append("lens_covered")
            return self._finalize(frame_id, timestamp, events, t0,
                                  driver_present=False, camera_ok=False,
                                  raw=raw)

        # 1. 通用目标检测（使用较低阈值 min(conf, phone_conf) 捞回小手机）
        infer_frame = frame
        if self.low_light_enhance and self._is_low_light(frame):
            infer_frame = self._enhance_low_light(frame)

        detect_conf = min(self.conf, self.phone_conf)
        r = self.yolo(infer_frame, conf=detect_conf, iou=self.iou,
                      imgsz=self.imgsz, verbose=False, device=self.device)[0]
        persons, phones = [], []
        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls.item())
                cf = float(box.conf.item())
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                if cls == COCO_PERSON and cf >= self.conf:
                    persons.append((xyxy, cf))
                elif cls == COCO_CELL_PHONE and cf >= self.phone_conf:
                    phones.append((xyxy, cf))

        # 2. 无驾驶员
        if not persons:
            events.append(BehaviorEvent(
                "no_driver", LABEL_ZH["no_driver"], 0.9,
                severity=SEVERITY_MAP["no_driver"],
                evidence="未检测到人体"))
            raw.append("no_driver")
            return self._finalize(frame_id, timestamp, events, t0,
                                  driver_present=False, camera_ok=True,
                                  raw=raw)

        driver_box, driver_cf = max(
            persons,
            key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1]))

        # 1.5 手机二次高分辨率重检：在驾驶员 crop 内以更大 imgsz 再跑一次
        # 触发条件：① 首轮无手机  或  ② 最高置信度 < 0.40（边界样本再确认）
        need_phone_recheck = (
            self.phone_recheck_imgsz > self.imgsz and (
                not phones or max(p[1] for p in phones) < 0.40
            )
        )
        if need_phone_recheck:
            x1, y1, x2, y2 = driver_box
            W = frame.shape[1]; H = frame.shape[0]
            # 聚焦上半身（手机一般在头/胸位置）
            cy2 = int(min(H, y1 + (y2 - y1) * 0.75))
            cx1 = max(0, int(x1 - (x2 - x1) * 0.1))
            cx2 = min(W, int(x2 + (x2 - x1) * 0.1))
            cy1 = max(0, int(y1))
            crop = infer_frame[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                rc = self.yolo(crop, conf=self.phone_conf, iou=self.iou,
                               imgsz=self.phone_recheck_imgsz,
                               verbose=False, device=self.device)[0]
                if rc.boxes is not None:
                    for box in rc.boxes:
                        if int(box.cls.item()) != COCO_CELL_PHONE:
                            continue
                        cf = float(box.conf.item())
                        if cf < self.phone_conf:
                            continue
                        xy = box.xyxy[0].cpu().numpy().tolist()
                        phones.append(([xy[0]+cx1, xy[1]+cy1,
                                         xy[2]+cx1, xy[3]+cy1], cf))

        # 3. 姿态关键点
        kpts = kconf = None
        pr = self.pose(frame, conf=0.3, imgsz=self.imgsz,
                       verbose=False, device=self.device)[0]
        if pr.keypoints is not None and len(pr.keypoints) > 0:
            kpts = pr.keypoints.xy[0].cpu().numpy()
            if pr.keypoints.conf is not None:
                kconf = pr.keypoints.conf[0].cpu().numpy()
            else:
                kconf = np.ones(17)

        # 4. 手机使用 / 打电话
        if phones:
            is_calling, phone_evt = self._judge_phone(phones, kpts, kconf)
            if phone_evt is not None:
                events.append(phone_evt)
                raw.append(phone_evt.type)

        # 5. 双手离盘
        if kpts is not None:
            off_evt = self._judge_hands(kpts, kconf, H, W)
            if off_evt is not None:
                events.append(off_evt)
                raw.append(off_evt.type)

        # 6. 姿势异常
        if kpts is not None:
            pos_evt = self._judge_posture(kpts, kconf)
            if pos_evt is not None:
                events.append(pos_evt)
                raw.append(pos_evt.type)

        # 7. 统一 8 类模型（如果配置了）- 一次推理覆盖 smoking/seatbelt/phone/calling
        if self.unified is not None:
            unified_events = self._judge_unified(frame, driver_bbox=driver_box)
            # 合并，相同类型去重（unified 的置信度更高则保留它）
            existing_types = {e.type for e in events}
            for ue in unified_events:
                if ue.type == "hand_on_wheel":
                    # unified 检出手在方向盘，移除规则法产生的 hands_off_wheel 告警
                    events = [e for e in events if e.type != "hands_off_wheel"]
                    raw = [t for t in raw if t != "hands_off_wheel"]
                    continue
                if ue.type == "seatbelt":
                    # 检出已系则移除未系告警
                    events = [e for e in events if e.type != "no_seatbelt"]
                    raw = [t for t in raw if t != "no_seatbelt"]
                    continue
                if ue.type in existing_types:
                    # 保留置信度较高的
                    events = [e for e in events if e.type != ue.type or e.confidence >= ue.confidence]
                    if any(e.type == ue.type for e in events):
                        continue
                events.append(ue)
                raw.append(ue.type)
                existing_types.add(ue.type)
        else:
            # 7a. 安全带（优先专用模型 → 回退启发式）
            if self.seatbelt is not None:
                sb_evt = self._judge_seatbelt(frame)
                if sb_evt is not None:
                    events.append(sb_evt)
                    raw.append(sb_evt.type)
            elif self.use_seatbelt_heuristic and kpts is not None:
                sb_evt = self._judge_seatbelt_heuristic(frame, kpts, kconf)
                if sb_evt is not None:
                    events.append(sb_evt)
                    raw.append(sb_evt.type)

            # 8a. 抽烟（专用模型，仅在驾驶员 bbox 内推理以加速）
            if self.smoking is not None:
                sm_evt = self._judge_smoking(frame, driver_bbox=driver_box)
                if sm_evt is not None:
                    events.append(sm_evt)
                    raw.append(sm_evt.type)

        return self._finalize(frame_id, timestamp, events, t0,
                              driver_present=True, camera_ok=True, raw=raw)

    # ---------- 子判定 ----------

    def _judge_phone(self, phones, kpts, kconf):
        """
        互斥判定：calling > phone_use，只返回一个事件。
        判定综合两路证据：
          (1) 手机中心到最近的头部锚点(鼻/左右耳)距离 d_phone
          (2) 任一手腕到锚点距离 d_wrist
        calling 需要: d_phone < 0.6 * face_w  AND  d_wrist < 0.8 * face_w
        phone_use = 检出手机但未满足 calling
        """
        phone_box, phone_cf = max(phones, key=lambda x: x[1])
        pc = np.array([(phone_box[0] + phone_box[2]) / 2,
                       (phone_box[1] + phone_box[3]) / 2])
        bbox_int = [int(v) for v in phone_box]

        face_w = None
        anchors = []
        if kpts is not None:
            if kconf[KP_LEFT_EAR] > 0.3 and kconf[KP_RIGHT_EAR] > 0.3:
                face_w = np.hypot(kpts[KP_LEFT_EAR][0] - kpts[KP_RIGHT_EAR][0],
                                  kpts[KP_LEFT_EAR][1] - kpts[KP_RIGHT_EAR][1])
            elif kconf[KP_LEFT_SHOULDER] > 0.3 and kconf[KP_RIGHT_SHOULDER] > 0.3:
                face_w = np.hypot(
                    kpts[KP_LEFT_SHOULDER][0] - kpts[KP_RIGHT_SHOULDER][0],
                    kpts[KP_LEFT_SHOULDER][1] - kpts[KP_RIGHT_SHOULDER][1]) * 0.4
            for idx in (KP_NOSE, KP_LEFT_EAR, KP_RIGHT_EAR):
                if kconf[idx] > 0.3:
                    anchors.append(kpts[idx])

        is_calling = False
        evidence = "画面内检出手机"

        if face_w and anchors:
            # 证据 A: 手机近头（放宽到 0.9 face_w，兼容自拍/驾驶位多角度）
            d_phone = min(np.hypot(pc[0] - a[0], pc[1] - a[1])
                          for a in anchors)
            phone_near_head = d_phone < 0.9 * face_w

            # 证据 B: 手机在面部高度（Y 轴接近鼻子，排除放桌上）
            nose_y = kpts[KP_NOSE][1] if kconf[KP_NOSE] > 0.3 else None
            phone_at_face_y = (nose_y is not None
                               and abs(pc[1] - nose_y) < 1.0 * face_w)

            # 证据 C: 至少一只手腕抬起（<= 1.2 face_w 近头，宽松）
            wrist_near_head = False
            d_wrist = None
            wrists = []
            if kconf[KP_LEFT_WRIST] > 0.3:
                wrists.append(kpts[KP_LEFT_WRIST])
            if kconf[KP_RIGHT_WRIST] > 0.3:
                wrists.append(kpts[KP_RIGHT_WRIST])
            if wrists and anchors:
                d_wrist = min(np.hypot(w[0] - a[0], w[1] - a[1])
                              for w in wrists for a in anchors)
                wrist_near_head = d_wrist < 1.2 * face_w

            # 规则: A 必须满足；B 或 C 任一满足就是 calling
            d_wrist_str = f"{d_wrist:.0f}" if d_wrist is not None else "?"
            if phone_near_head and (phone_at_face_y or wrist_near_head):
                is_calling = True
                evidence = (f"calling: d_phone={d_phone:.0f} "
                            f"d_wrist={d_wrist_str} "
                            f"face_w={face_w:.0f}")
            elif phone_near_head:
                evidence = f"phone_use: 手机在头旁但非通话姿态 (d_phone={d_phone:.0f})"
            else:
                evidence = f"phone_use: 手机远离头部 (d_phone={d_phone:.0f})"

        if is_calling:
            return True, BehaviorEvent(
                "calling", LABEL_ZH["calling"], phone_cf, bbox=bbox_int,
                severity=SEVERITY_MAP["calling"],
                evidence=evidence)
        return False, BehaviorEvent(
            "phone_use", LABEL_ZH["phone_use"], phone_cf, bbox=bbox_int,
            severity=SEVERITY_MAP["phone_use"],
            evidence=evidence)

    def _judge_hands(self, kpts, kconf, H, W):
        roi = self._wheel_roi(H, W)
        l = kpts[KP_LEFT_WRIST] if kconf[KP_LEFT_WRIST] > 0.3 else None
        r = kpts[KP_RIGHT_WRIST] if kconf[KP_RIGHT_WRIST] > 0.3 else None
        l_in = l is not None and self._point_in_bbox(l, roi)
        r_in = r is not None and self._point_in_bbox(r, roi)
        if (l is None and r is None):
            return None
        if not l_in and not r_in:
            return BehaviorEvent(
                "hands_off_wheel", LABEL_ZH["hands_off_wheel"], 0.75,
                severity=SEVERITY_MAP["hands_off_wheel"],
                evidence="双手均未在方向盘 ROI 内")
        return None

    def _judge_posture(self, kpts, kconf):
        if kconf[KP_LEFT_SHOULDER] < 0.3 or kconf[KP_RIGHT_SHOULDER] < 0.3:
            return None
        lsh = kpts[KP_LEFT_SHOULDER]
        rsh = kpts[KP_RIGHT_SHOULDER]
        dy = abs(lsh[1] - rsh[1])
        dx = abs(lsh[0] - rsh[0]) + 1e-6
        tilt = math.degrees(math.atan2(dy, dx))
        if tilt > 30:
            return BehaviorEvent(
                "abnormal_posture", LABEL_ZH["abnormal_posture"],
                min(1.0, tilt / 60),
                severity=SEVERITY_MAP["abnormal_posture"],
                evidence=f"肩部倾角 {tilt:.1f}°")
        return None

    def _judge_seatbelt_heuristic(self, frame, kpts, kconf):
        """
        启发式安全带检测（无专用模型时使用）:
          - 从左肩到右髋、右肩到左髋两条对角线
          - 沿线取条带 ROI（窄，约肩宽 15%），检查暗色像素占比与
            方向一致的梯度能量
          - 两条任一"暗条带"命中 → 认为已系；两条全不命中 → no_seatbelt
        精度有限，仅作训练模型前的兜底，置信度 ≤ 0.45
        """
        need = [KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP]
        if any(kconf[i] < 0.3 for i in need):
            return None
        lsh, rsh = kpts[KP_LEFT_SHOULDER], kpts[KP_RIGHT_SHOULDER]
        lhp, rhp = kpts[KP_LEFT_HIP],      kpts[KP_RIGHT_HIP]
        shoulder_w = np.hypot(lsh[0]-rsh[0], lsh[1]-rsh[1])
        if shoulder_w < 40:  # 人像过小
            return None

        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        def band_is_belt(p_top, p_bot):
            """沿 p_top→p_bot 画一条带，返回是否像安全带（暗+方向一致）"""
            # 沿线采样 N 个点，每个点法向取 ±half_w 像素求均值
            N = 24
            half_w = max(3, int(shoulder_w * 0.05))
            vec = p_bot - p_top
            length = np.hypot(*vec)
            if length < 10:
                return False, 0.0
            dir_ = vec / length
            normal = np.array([-dir_[1], dir_[0]])
            dark_votes = 0
            samples = []
            for t in np.linspace(0.15, 0.9, N):  # 跳过颈部/腰带边缘
                center = p_top + vec * t
                # 带上的暗度最小值
                best = 255
                for s in np.linspace(-half_w, half_w, 5):
                    px = center + normal * s
                    x, y = int(px[0]), int(px[1])
                    if 0 <= x < W and 0 <= y < H:
                        best = min(best, int(gray[y, x]))
                samples.append(best)
                if best < 90:
                    dark_votes += 1
            # 暗样本占比
            ratio = dark_votes / N
            # 方差（安全带颜色较均匀 → 小方差；杂乱衣服 → 大方差）
            arr = np.array(samples, dtype=np.float32)
            var = arr.std()
            # 判定：至少 40% 暗且方差不大
            ok = (ratio >= 0.4) and (var < 35)
            return ok, ratio

        # 两条对角线（驾驶员通常只有一条可见，取命中即可）
        belt_l, r_l = band_is_belt(lsh, rhp)
        belt_r, r_r = band_is_belt(rsh, lhp)
        belt_ok = belt_l or belt_r
        if belt_ok:
            return None  # 有安全带，不告警

        # 两条都未命中 → 可能未系
        cf = 0.45  # 启发式天花板
        bbox = [int(min(lsh[0], rsh[0], lhp[0], rhp[0])),
                int(min(lsh[1], rsh[1])),
                int(max(lsh[0], rsh[0], lhp[0], rhp[0])),
                int(max(lhp[1], rhp[1]))]
        return BehaviorEvent(
            "no_seatbelt", LABEL_ZH["no_seatbelt"], cf, bbox=bbox,
            severity=SEVERITY_MAP["no_seatbelt"],
            evidence=f"启发式：肩-腰对角线未检出暗条带 (L={r_l:.2f}, R={r_r:.2f})")

    def _judge_seatbelt(self, frame):
        r = self.seatbelt(frame, conf=0.35, imgsz=self.imgsz,
                          verbose=False, device=self.device)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        names = r.names or {}
        no_belt_cf = 0.0
        no_belt_box = None
        for box in r.boxes:
            cls = int(box.cls.item())
            name = names.get(cls, "").lower()
            cf = float(box.conf.item())
            if ("no" in name) or ("without" in name) or ("unbelted" in name):
                if cf > no_belt_cf:
                    no_belt_cf = cf
                    no_belt_box = box.xyxy[0].cpu().numpy().tolist()
        if no_belt_cf > 0 and no_belt_box:
            return BehaviorEvent(
                "no_seatbelt", LABEL_ZH["no_seatbelt"], no_belt_cf,
                bbox=[int(v) for v in no_belt_box],
                severity=SEVERITY_MAP["no_seatbelt"],
                evidence="模型输出 no_belt 类")
        return None

    def _judge_smoking(self, frame, driver_bbox=None):
        # 仅在驾驶员 bbox 范围内推理（可显著提速）。上下左右加 10% padding
        if driver_bbox is not None:
            H, W = frame.shape[:2]
            x1, y1, x2, y2 = driver_bbox
            pw = (x2 - x1) * 0.1
            ph = (y2 - y1) * 0.1
            cx1 = max(0, int(x1 - pw))
            cy1 = max(0, int(y1 - ph))
            cx2 = min(W, int(x2 + pw))
            cy2 = min(H, int(y2 + ph))
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                return None
        else:
            cx1 = cy1 = 0
            crop = frame

        r = self.smoking(crop, conf=0.35, imgsz=self.imgsz,
                         verbose=False, device=self.device)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        best = max(r.boxes, key=lambda b: float(b.conf.item()))
        cf = float(best.conf.item())
        xyxy = best.xyxy[0].cpu().numpy().tolist()
        # crop 坐标 → 原图坐标
        xyxy_full = [xyxy[0] + cx1, xyxy[1] + cy1,
                     xyxy[2] + cx1, xyxy[3] + cy1]
        return BehaviorEvent(
            "smoking", LABEL_ZH["smoking"], cf,
            bbox=[int(v) for v in xyxy_full],
            severity=SEVERITY_MAP["smoking"],
            evidence="检出 cigarette")

    # ---------- unified 8 类模型 ----------

    # unified 类名 → behavior type 映射
    _UNIFIED_TYPE_MAP = {
        "phone_use":     ("phone_use",       "unified 模型检出 texting"),
        "calling":       ("calling",         "unified 模型检出 calling"),
        "cigarette":     ("smoking",         "unified 模型检出 cigarette"),
        "no_seatbelt":   ("no_seatbelt",     "unified 模型检出 no_seatbelt"),
        "seatbelt":      ("seatbelt",        "unified 模型检出 seatbelt"),
        "hand_on_wheel": ("hand_on_wheel",   "unified 模型检出 hand_on_wheel"),
        "drinking":      ("abnormal_posture","unified 模型检出 drinking"),
        "reach_behind":  ("abnormal_posture","unified 模型检出 reach_behind"),
    }

    def _judge_unified(self, frame, driver_bbox=None):
        """对驾驶员 crop 跑 unified 8 类模型，返回 BehaviorEvent 列表"""
        if driver_bbox is not None:
            H, W = frame.shape[:2]
            x1, y1, x2, y2 = driver_bbox
            pw = (x2 - x1) * 0.15
            ph = (y2 - y1) * 0.10
            cx1 = max(0, int(x1 - pw))
            cy1 = max(0, int(y1 - ph))
            cx2 = min(W, int(x2 + pw))
            cy2 = min(H, int(y2 + ph))
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                return []
        else:
            cx1 = cy1 = 0
            crop = frame

        r = self.unified(crop, conf=self.phone_conf, iou=self.iou,
                         imgsz=self.imgsz, verbose=False,
                         device=self.device)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        # 按类别取最高置信度 bbox
        by_cls = {}
        for box in r.boxes:
            cls = int(box.cls.item())
            cf = float(box.conf.item())
            name = self.unified_names.get(cls, "").lower()
            if not name:
                continue
            if name not in by_cls or cf > by_cls[name][0]:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                by_cls[name] = (cf, xyxy)

        events = []
        for cls_name, (cf, xyxy) in by_cls.items():
            if cls_name not in self._UNIFIED_TYPE_MAP:
                continue
            btype, evidence = self._UNIFIED_TYPE_MAP[cls_name]
            # hand_on_wheel 和 seatbelt 是"安全基线"，不产生告警事件但用于抑制
            if btype in ("hand_on_wheel", "seatbelt"):
                events.append(BehaviorEvent(
                    btype, btype, cf, bbox=None, severity="low",
                    evidence=evidence))
                continue
            # 其它是危险行为
            bbox_full = [int(xyxy[0] + cx1), int(xyxy[1] + cy1),
                         int(xyxy[2] + cx1), int(xyxy[3] + cy1)]
            events.append(BehaviorEvent(
                btype, LABEL_ZH.get(btype, btype), cf, bbox=bbox_full,
                severity=SEVERITY_MAP.get(btype, "medium"),
                evidence=evidence))
        return events

    # ---------- 打包 ----------

    def _finalize(self, frame_id, timestamp, events, start_ts,
                  driver_present, camera_ok, raw):
        stable_set = set(self.smoother.update(raw, timestamp))
        stable_events = []
        for e in events:
            if e.type in stable_set:
                e.duration_s = round(self.smoother.duration(e.type, timestamp), 2)
                stable_events.append(e)

        if stable_events:
            max_sev = max(stable_events,
                          key=lambda x: _SEV_ORDER[x.severity]).severity
        else:
            max_sev = "none"

        latency_ms = round((time.time() - start_ts) * 1000, 2)
        behaviors_dict = [e.to_dict() for e in stable_events]
        score = compute_risk_score(behaviors_dict)
        return {
            "frame_id": frame_id,
            "timestamp": round(timestamp, 3),
            "latency_ms": latency_ms,
            "behaviors": behaviors_dict,
            "alert_level": max_sev,
            "risk_score": score,
            "risk_tier": risk_tier(score),
            "recommendation": RECOMMEND.get(max_sev, "正常驾驶"),
            "driver_present": driver_present,
            "camera_ok": camera_ok,
        }

    # ---------- 可视化 ----------

    def visualize(self, frame: np.ndarray, result: Dict,
                  style: str = "debug") -> np.ndarray:
        """
        style:
          - "debug": 调试视图 (方向盘 ROI / 行为列表 / FPS)
          - "monitor": 监控视频样式，仿真车载 DMS 显示效果
        """
        if style == "monitor":
            return self._visualize_monitor(frame, result)
        return self._visualize_debug(frame, result)

    def _visualize_debug(self, frame, result):
        vis = frame.copy()
        H, W = vis.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in self._wheel_roi(H, W)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(vis, "wheel ROI", (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
        color_map = {
            "critical": (0, 0, 255), "high": (0, 64, 255),
            "medium":   (0, 165, 255), "low": (0, 255, 255),
        }
        for i, b in enumerate(result["behaviors"]):
            color = color_map.get(b["severity"], (0, 255, 0))
            if b.get("bbox"):
                bx1, by1, bx2, by2 = b["bbox"]
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 2)
            txt = f"{b['type']} {b['confidence']:.2f} ({b['duration_s']}s)"
            cv2.putText(vis, txt, (10, 30 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        lvl = result["alert_level"]
        fps = 1000 / max(1e-3, result["latency_ms"])
        banner_color = color_map.get(lvl, (0, 255, 0))
        cv2.putText(vis, f"ALERT: {lvl.upper()}  ~{fps:.1f} FPS",
                    (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    banner_color, 2)
        return vis

    @staticmethod
    def _draw_rounded_rect(img, pt1, pt2, color, thickness,
                           radius=12, fill_color=None):
        """绘制圆角矩形（支持填充 + 描边）"""
        x1, y1 = pt1
        x2, y2 = pt2
        r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
        if fill_color is not None:
            # 填充
            cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), fill_color, -1)
            cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), fill_color, -1)
            cv2.circle(img, (x1 + r, y1 + r), r, fill_color, -1)
            cv2.circle(img, (x2 - r, y1 + r), r, fill_color, -1)
            cv2.circle(img, (x1 + r, y2 - r), r, fill_color, -1)
            cv2.circle(img, (x2 - r, y2 - r), r, fill_color, -1)
        if thickness > 0:
            # 描边四条直线 + 四个圆弧
            cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)

    def _visualize_monitor(self, frame, result):
        """深色科技风车载 DMS 监控界面
          · 深色半透明顶栏 + 时间 / 状态指示灯
          · 行为 bbox 带发光描边 + 类别标签
          · 右侧圆角毛玻璃告警面板
          · 左侧驾驶员状态仪表 (驾驶员/摄像头/吸烟模型)
          · 底部渐变风险仪表条 + 居中大字主状态
        """
        import datetime as _dt
        vis = frame.copy()
        H, W = vis.shape[:2]

        # -- 配色表 --
        CYAN    = (230, 216, 0)     # 霓虹青
        GREEN   = (80, 255, 80)
        DGREEN  = (60, 180, 60)
        YELLOW  = (60, 220, 255)
        ORANGE  = (0, 165, 255)
        RED     = (80, 80, 255)
        DARK    = (18, 18, 22)
        PANEL   = (30, 30, 38)
        WHITE   = (240, 240, 240)
        DIM     = (140, 140, 140)
        sev_color = {"critical": RED, "high": (60, 80, 255), "medium": ORANGE,
                     "low": YELLOW, "none": CYAN}
        tier_color = {"safe": CYAN, "attention": YELLOW,
                      "warning": ORANGE, "danger": RED, "critical": RED}

        behaviors = result.get("behaviors", [])
        score = result.get("risk_score", 0.0)
        tier = result.get("risk_tier", "safe")
        latency = result.get("latency_ms", 0.0)

        # ======== 1. 顶栏 (半透明深色条) ========
        top_h = 42
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (W, top_h), DARK, -1)
        vis = cv2.addWeighted(overlay, 0.7, vis, 0.3, 0)
        # 外框发光线
        cv2.line(vis, (0, top_h), (W, top_h), CYAN, 1, cv2.LINE_AA)

        # 左侧: DMS 标题
        cv2.putText(vis, "DMS", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2, cv2.LINE_AA)
        cv2.putText(vis, "MONITOR", (62, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1, cv2.LINE_AA)

        # 中间: 时间
        ts = _dt.datetime.fromtimestamp(
            result.get("timestamp", time.time())
        ).strftime("%Y-%m-%d  %H:%M:%S")
        (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(vis, ts, (W // 2 - tw // 2, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1, cv2.LINE_AA)

        # 右侧: FPS + 延迟
        fps_val = 1000.0 / max(1e-3, latency)
        fps_txt = f"{fps_val:.0f}FPS  {latency:.0f}ms"
        (fw, _), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(vis, fps_txt, (W - fw - 14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, DGREEN, 1, cv2.LINE_AA)

        # ======== 2. 左侧状态指示灯面板 ========
        ind_x, ind_y0 = 12, top_h + 14
        indicators = [
            ("DRIVER", result.get("driver_present", True)),
            ("CAMERA", result.get("camera_ok", True)),
            ("SMOKE-DET", self.smoking is not None),
        ]
        for i, (label, ok) in enumerate(indicators):
            y = ind_y0 + i * 22
            dot_color = GREEN if ok else RED
            cv2.circle(vis, (ind_x + 6, y + 4), 5, dot_color, -1, cv2.LINE_AA)
            # 发光效果
            glow = vis.copy()
            cv2.circle(glow, (ind_x + 6, y + 4), 10, dot_color, -1)
            vis = cv2.addWeighted(glow, 0.15, vis, 0.85, 0)
            cv2.putText(vis, label, (ind_x + 18, y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)

        # ======== 3. 行为 bbox + 发光描边 + 标签 ========
        for b in behaviors:
            col = sev_color.get(b["severity"], CYAN)
            if b.get("bbox"):
                bx1, by1, bx2, by2 = b["bbox"]
                # 外发光 (粗 + 半透明)
                glow = vis.copy()
                cv2.rectangle(glow, (bx1 - 2, by1 - 2), (bx2 + 2, by2 + 2),
                              col, 4, cv2.LINE_AA)
                vis = cv2.addWeighted(glow, 0.3, vis, 0.7, 0)
                # 主框
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), col, 2, cv2.LINE_AA)
                # 四角强调线 (科技感)
                clen = min(16, (bx2 - bx1) // 4, (by2 - by1) // 4)
                for (cx, cy), (dx, dy) in [
                    ((bx1, by1), (1, 1)), ((bx2, by1), (-1, 1)),
                    ((bx1, by2), (1, -1)), ((bx2, by2), (-1, -1)),
                ]:
                    cv2.line(vis, (cx, cy), (cx + dx * clen, cy), col, 2, cv2.LINE_AA)
                    cv2.line(vis, (cx, cy), (cx, cy + dy * clen), col, 2, cv2.LINE_AA)
                # 标签背景 (圆角小条)
                lab = b["type"].replace("_", " ")
                (ttw, tth), _ = cv2.getTextSize(
                    lab, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                lx1, ly1 = bx1, by1 - tth - 10
                lx2, ly2 = bx1 + ttw + 12, by1
                self._draw_rounded_rect(vis, (lx1, ly1), (lx2, ly2),
                                        col, 0, radius=6, fill_color=col)
                cv2.putText(vis, lab, (lx1 + 6, ly2 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, DARK, 1, cv2.LINE_AA)

        # ======== 4. 右侧告警面板 (毛玻璃) ========
        panel_w = 220
        panel_x = W - panel_w - 14
        panel_y0 = top_h + 14
        shown = behaviors[:5]
        panel_h = 36 + max(1, len(shown)) * 40
        # 半透明背景
        overlay = vis.copy()
        self._draw_rounded_rect(overlay, (panel_x, panel_y0),
                                (panel_x + panel_w, panel_y0 + panel_h),
                                CYAN, 0, radius=10, fill_color=PANEL)
        vis = cv2.addWeighted(overlay, 0.65, vis, 0.35, 0)
        # 边框
        self._draw_rounded_rect(vis, (panel_x, panel_y0),
                                (panel_x + panel_w, panel_y0 + panel_h),
                                CYAN, 1, radius=10)
        # 标题
        cv2.putText(vis, "ACTIVE ALERTS", (panel_x + 10, panel_y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1, cv2.LINE_AA)
        cv2.line(vis, (panel_x + 10, panel_y0 + 28),
                 (panel_x + panel_w - 10, panel_y0 + 28), CYAN, 1, cv2.LINE_AA)

        if not shown:
            cv2.putText(vis, "ALL CLEAR",
                        (panel_x + 10, panel_y0 + 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
        for i, b in enumerate(shown):
            col = sev_color.get(b["severity"], WHITE)
            y = panel_y0 + 48 + i * 40
            # 严重度色条
            cv2.rectangle(vis, (panel_x + 8, y - 12),
                          (panel_x + 12, y + 14), col, -1)
            # 行为名称
            cv2.putText(vis, b["type"].replace("_", " "),
                        (panel_x + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
            # 置信度 + 持续时间
            sub = f"conf {b['confidence']:.0%}  {b['duration_s']:.1f}s"
            cv2.putText(vis, sub,
                        (panel_x + 20, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM, 1, cv2.LINE_AA)

        # ======== 5. 底部区域 ========
        bot_h = 70
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, H - bot_h), (W, H), DARK, -1)
        vis = cv2.addWeighted(overlay, 0.65, vis, 0.35, 0)
        cv2.line(vis, (0, H - bot_h), (W, H - bot_h), CYAN, 1, cv2.LINE_AA)

        # 5a. 渐变风险仪表条
        bar_x, bar_y, bar_w, bar_h = 16, H - 56, min(300, W // 2 - 20), 14
        # 背景
        cv2.rectangle(vis, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (40, 40, 48), -1)
        # 填充 (逐像素渐变: 绿→黄→红)
        fill_w = max(0, int(bar_w * score / 100))
        for px in range(fill_w):
            ratio = px / max(1, bar_w)
            if ratio < 0.4:
                r2 = ratio / 0.4
                c = (0, int(255 * (1 - r2 * 0.3)), int(230 * (1 - r2) + 80 * r2))
            elif ratio < 0.7:
                r2 = (ratio - 0.4) / 0.3
                c = (0, int(180 + 40 * (1 - r2)), int(80 + 175 * r2))
            else:
                r2 = (ratio - 0.7) / 0.3
                c = (int(60 * r2), int(80 * (1 - r2)), 255)
            cv2.line(vis, (bar_x + px, bar_y + 1),
                     (bar_x + px, bar_y + bar_h - 1), c, 1)
        # 外框
        cv2.rectangle(vis, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), CYAN, 1)
        # 刻度
        for m in (25, 50, 75):
            tx = bar_x + int(bar_w * m / 100)
            cv2.line(vis, (tx, bar_y), (tx, bar_y + bar_h), (60, 60, 70), 1)
        # 标签
        gcol = tier_color.get(tier, CYAN)
        cv2.putText(vis,
                    f"RISK {score:.0f}/100",
                    (bar_x, bar_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, gcol, 1, cv2.LINE_AA)
        cv2.putText(vis,
                    tier.upper(),
                    (bar_x + bar_w + 8, bar_y + bar_h - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, gcol, 1, cv2.LINE_AA)

        # 5b. 底部居中主状态标签
        if behaviors:
            top = max(behaviors,
                      key=lambda x: _SEV_ORDER.get(x["severity"], 0))
            primary = top["type"].replace("_", " ").upper()
            primary_color = sev_color.get(top["severity"], CYAN)
        else:
            primary = "NORMAL"
            primary_color = CYAN

        (lw, lh), _ = cv2.getTextSize(primary, cv2.FONT_HERSHEY_SIMPLEX,
                                       1.0, 2)
        px_c = W // 2 - lw // 2
        py_c = H - 16
        # 发光底色
        glow = vis.copy()
        cv2.rectangle(glow,
                      (px_c - 16, py_c - lh - 8),
                      (px_c + lw + 16, py_c + 8),
                      primary_color, -1)
        vis = cv2.addWeighted(glow, 0.12, vis, 0.88, 0)
        # 文字
        cv2.putText(vis, primary, (px_c, py_c),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    primary_color, 2, cv2.LINE_AA)

        # 5c. 右下角 REC 闪烁指示
        rec_x = W - 70
        rec_y = H - 18
        # 用帧号模拟闪烁 (偶数帧显示)
        fid = result.get("frame_id", 0)
        if fid % 2 == 0:
            cv2.circle(vis, (rec_x, rec_y), 5, RED, -1, cv2.LINE_AA)
        cv2.putText(vis, "REC", (rec_x + 10, rec_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, RED, 1, cv2.LINE_AA)

        return vis


# ---------- CLI 入口（用于快速测试） ----------

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0=摄像头, 或视频/图像路径")
    ap.add_argument("--yolo", default="yolov8n.pt")
    ap.add_argument("--pose", default="yolov8n-pose.pt")
    ap.add_argument("--seatbelt", default=None)
    ap.add_argument("--smoking", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save", default=None, help="输出视频路径")
    args = ap.parse_args()

    det = BehaviorDetector(
        yolo_weights=args.yolo,
        pose_weights=args.pose,
        seatbelt_weights=args.seatbelt,
        smoking_weights=args.smoking,
        device=args.device,
    )

    src = int(args.source) if args.source.isdigit() else args.source
    cap, src_desc = open_capture(src)
    if cap is None:
        print(f"[CLI] 摄像头打开失败: {src_desc}")
        raise SystemExit(1)
    print(f"[CLI] 视频源: {src_desc}")
    writer = None
    fid = 0
    fail_streak = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                fail_streak += 1
                if fail_streak > 10:
                    print("[CLI] 连续读帧失败，退出")
                    break
                time.sleep(0.02)
                continue
            fail_streak = 0
            res = det.predict(frame, frame_id=fid, timestamp=time.time())
            vis = det.visualize(frame, res)
            if args.save:
                if writer is None:
                    H, W = vis.shape[:2]
                    writer = cv2.VideoWriter(
                        args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                        20.0, (W, H))
                writer.write(vis)
            cv2.imshow("behavior_algo_a", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if fid % 30 == 0:
                print(json.dumps(res, ensure_ascii=False))
            fid += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
