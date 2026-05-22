"""
behavior_detector.py
====================
DMS 项目 - 行为识别器（单 YOLO 模型版本）

输入: 单帧图像 np.ndarray (H,W,3) BGR
输出: 结构化 JSON 行为告警

设计:
  * 只依赖一个 YOLO 模型
      - 优先 unified.pt (本仓库训练的 8 类自定义模型)
      - 没有 unified.pt 时回退到 yolov8n.pt (COCO)，只能识别 person / cell phone
  * 去掉 pose / seatbelt.pt / smoking.pt 多模型分支，单次推理出全部动作
  * 保留: 镜头遮挡检测、时序滑窗平滑、风险评分、人读告警等级
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

import cv2
import numpy as np
from ultralytics import YOLO


# ---------- 常量 ----------

COCO_PERSON = 0
COCO_CELL_PHONE = 67

SEVERITY_MAP = {
    "no_driver":        "critical",
    "phone_use":        "high",
    "no_seatbelt":      "high",
    "hands_off_wheel":  "high",
    "smoking":          "medium",
    "drinking":         "low",
    "eating":           "medium",
    "lens_covered":     "medium",
    "hand_on_wheel":    "low",   # 安全基线
    "seatbelt":         "low",   # 安全基线
}

LABEL_ZH = {
    "no_driver":        "驾驶位无人",
    "phone_use":        "驾驶中使用手机",
    "no_seatbelt":      "未系安全带",
    "hands_off_wheel":  "双手离开方向盘",
    "smoking":          "驾驶中吸烟",
    "drinking":         "驾驶中饮水",
    "eating":           "驾驶中进食",
    "lens_covered":     "摄像头被遮挡",
    "hand_on_wheel":    "双手在方向盘",
    "seatbelt":         "已系安全带",
}

RECOMMEND = {
    "critical": "立即语音警告 + 方向盘震动 + 限速",
    "high":     "语音警告 + 仪表盘闪烁",
    "medium":   "仪表盘图标提示",
    "low":      "屏幕柔和提示",
    "none":     "正常驾驶",
}

_SEV_ORDER = {"none": -1, "low": 0, "medium": 1, "high": 2, "critical": 3}

RISK_WEIGHT = {
    "no_driver":        1.00,
    "phone_use":        0.80,
    "no_seatbelt":      0.70,
    "hands_off_wheel":  0.65,
    "smoking":          0.45,
    "eating":           0.55,
    "drinking":         0.30,
    "lens_covered":     0.40,
}


def compute_risk_score(behaviors):
    """0~100 的综合风险分数，多行为用概率互补组合"""
    if not behaviors:
        return 0.0
    prob_safe = 1.0
    for b in behaviors:
        w = RISK_WEIGHT.get(b["type"], 0.0)
        if w <= 0:
            continue
        c = b.get("confidence", 0.5)
        d = b.get("duration_s", 0.0)
        dur_factor = min(1.6, 1.0 + 0.06 * min(d, 10.0))
        contrib = max(0.0, min(1.0, w * c * dur_factor))
        prob_safe *= (1.0 - contrib)
    return round(100 * (1 - prob_safe), 1)


def risk_tier(score):
    if score < 10:
        return "safe"
    if score < 30:
        return "attention"
    if score < 60:
        return "warning"
    if score < 85:
        return "danger"
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
        # 1) 推入观察到的行为
        seen = set(behaviors)
        for b in seen:
            self.history[b].append(1)
        # 没看到的行为推 0
        for b in list(self.history.keys()):
            if b not in seen:
                self.history[b].append(0)

        # 2) 多数票判稳态
        stable = []
        for b, h in self.history.items():
            on = sum(h)
            if not self.active.get(b, False) and on >= self.activate:
                self.active[b] = True
                self.start_ts[b] = timestamp
            elif self.active.get(b, False) and (self.window - on) >= self.deactivate:
                self.active[b] = False
            if self.active.get(b, False):
                stable.append(b)
        return stable

    def duration(self, behavior: str, now: float) -> float:
        if not self.active.get(behavior, False):
            return 0.0
        return max(0.0, now - self.start_ts.get(behavior, now))


# ---------- 主检测器 ----------

class BehaviorDetector:
    """单 YOLO 模型驾驶员行为识别器。

    优先加载 unified.pt（自训 8 类），无则回退 yolov8n.pt (COCO)。
    返回 JSON 友好的 dict。
    """

    # unified.pt 训练时的 8 类（与 scripts/merge_datasets.py 对应）
    # 动作类（非基线）类名 -> (behavior type, evidence)。
    # hand_on_wheel / seatbelt / no_seatbelt 走专门逻辑，不在此表。
    UNIFIED_TO_BEHAVIOR = {
        "phone_use":       ("phone_use",       "unified: phone_use"),
        "smoking":         ("smoking",         "unified: smoking"),
        "drinking":        ("drinking",        "unified: drinking"),
        "eating":          ("eating",          "unified: eating"),
        # 双手离盘直接检出即告警（样本充足，比反推 hand_on_wheel 可靠）
        "hands_off_wheel": ("hands_off_wheel", "unified: hands_off_wheel"),
    }

    # 走"默认告警 / 检出安全基线才解除"逻辑的成对类
    # 安全基线类名 -> (告警 behavior, 检出告警类名)
    # 仅安全带：默认未系，检出 seatbelt 才解除。
    _SAFE_BASELINE_PAIRS = {
        "seatbelt": ("no_seatbelt", "no_seatbelt"),
    }

    def __init__(
        self,
        unified_weights: Optional[str] = None,
        base_weights: str = "yolov8n.pt",
        device: str = "cpu",
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 640,
        low_light_enhance: bool = True,
        temporal_window: int = 5,
        verify_phone: bool = True,
        phone_verify_conf: float = 0.15,
    ):
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.low_light_enhance = low_light_enhance

        # 唯一的 YOLO 模型
        upath = Path(unified_weights) if unified_weights else None
        if upath and upath.exists():
            print(f"[Init] 加载 unified 模型: {upath}")
            self.model = YOLO(str(upath))
            self.mode = "unified"
        else:
            print(f"[Init] unified 未配置/不存在，回退基础检测: {base_weights}")
            self.model = YOLO(base_weights)
            self.mode = "base"
        self.class_names = {int(k): v for k, v in (self.model.names or {}).items()}
        print(f"       mode={self.mode}  classes={list(self.class_names.values())[:10]}")

        # phone_use 共现确认器：用 COCO 模型检 cell phone，
        # 只有真检测到手机时才认可 phone_use（抑制"手靠近脸"等误报）
        self.phone_verify = None
        self.phone_verify_conf = phone_verify_conf
        if verify_phone and self.mode == "unified":
            bpath = Path(base_weights)
            if bpath.exists():
                print(f"[Init] 加载手机确认模型(COCO): {base_weights}")
                self.phone_verify = YOLO(str(bpath))

        self.smoother = TemporalSmoother(window=temporal_window)

    def _has_cell_phone(self, frame: np.ndarray) -> bool:
        """用 COCO 模型判断帧内是否存在手机"""
        if self.phone_verify is None:
            return True  # 没有确认器则不拦截
        r = self.phone_verify(frame, conf=self.phone_verify_conf, iou=self.iou,
                              imgsz=self.imgsz, verbose=False, device=self.device)[0]
        if r.boxes is None:
            return False
        return any(int(b.cls.item()) == COCO_CELL_PHONE for b in r.boxes)

    # ---------- 工具 ----------

    @staticmethod
    def _lens_covered(frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean = gray.mean()
        return lap < 10.0 or mean < 15.0

    @staticmethod
    def _enhance_low_light(frame: np.ndarray) -> np.ndarray:
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def _is_low_light(frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray.mean() < 80

    # ---------- 主推理 ----------

    def predict(self, frame: np.ndarray, frame_id: int = 0,
                timestamp: Optional[float] = None) -> Dict:
        t0 = time.time()
        if timestamp is None:
            timestamp = t0

        events: List[BehaviorEvent] = []
        raw: List[str] = []

        # 0. 镜头遮挡 → 直接返回
        if self._lens_covered(frame):
            events.append(BehaviorEvent(
                "lens_covered", LABEL_ZH["lens_covered"], 1.0,
                severity=SEVERITY_MAP["lens_covered"],
                evidence="图像方差/均值过低"))
            raw.append("lens_covered")
            return self._finalize(frame_id, timestamp, events, t0,
                                  driver_present=False, camera_ok=False,
                                  raw=raw)

        # 1. 单次 YOLO 推理
        infer_frame = frame
        if self.low_light_enhance and self._is_low_light(frame):
            infer_frame = self._enhance_low_light(frame)
        r = self.model(infer_frame, conf=self.conf, iou=self.iou,
                       imgsz=self.imgsz, verbose=False,
                       device=self.device)[0]

        if self.mode == "unified":
            events, raw, driver_present = self._from_unified(r, events, raw)
            # phone_use 需手机共现确认：没真检测到手机就抑制
            if "phone_use" in raw and not self._has_cell_phone(infer_frame):
                events = [e for e in events if e.type != "phone_use"]
                raw = [t for t in raw if t != "phone_use"]
        else:
            events, raw, driver_present = self._from_base(r, events, raw)

        return self._finalize(frame_id, timestamp, events, t0,
                              driver_present=driver_present, camera_ok=True,
                              raw=raw)

    # ---------- 类别 → BehaviorEvent ----------

    def _from_unified(self, r, events, raw):
        """从 unified 8 类输出生成 BehaviorEvent 列表"""
        # 按类别名只保留最高 conf 的 bbox
        by_name: Dict[str, Tuple[float, List[float]]] = {}
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                cf = float(box.conf.item())
                name = (self.class_names.get(cls_id) or "").lower()
                if not name:
                    continue
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                if name not in by_name or cf > by_name[name][0]:
                    by_name[name] = (cf, xyxy)

        # 完全无检测 → 驾驶位无人
        if not by_name:
            events.append(BehaviorEvent(
                "no_driver", LABEL_ZH["no_driver"], 0.9,
                severity=SEVERITY_MAP["no_driver"],
                evidence="unified 未检出任何类"))
            raw.append("no_driver")
            return events, raw, False

        # 成对安全基线类（安全带 / 方向盘）走专门逻辑，普通动作类直接出事件
        baseline_names = set(self._SAFE_BASELINE_PAIRS)
        alarm_names = {a for _, a in self._SAFE_BASELINE_PAIRS.values()}

        for name, (cf, xyxy) in by_name.items():
            if name in baseline_names or name in alarm_names:
                continue  # 安全带/方向盘单独处理
            spec = self.UNIFIED_TO_BEHAVIOR.get(name)
            if not spec:
                continue
            btype, evidence = spec
            events.append(BehaviorEvent(
                btype, LABEL_ZH.get(btype, btype), cf,
                bbox=[int(v) for v in xyxy],
                severity=SEVERITY_MAP.get(btype, "medium"),
                evidence=evidence))
            raw.append(btype)

        # 安全带 / 双手离盘：默认告警，检出安全基线（seatbelt / hand_on_wheel）才解除
        for baseline, (alarm_type, alarm_name) in self._SAFE_BASELINE_PAIRS.items():
            if baseline in by_name:
                cf, xyxy = by_name[baseline]
                events.append(BehaviorEvent(
                    baseline, LABEL_ZH.get(baseline, baseline), cf,
                    bbox=[int(v) for v in xyxy], severity="low",
                    evidence=f"检出{LABEL_ZH.get(baseline, baseline)}"))
                raw.append(baseline)
            else:
                if alarm_name in by_name:
                    cf, xyxy = by_name[alarm_name]
                    bbox = [int(v) for v in xyxy]
                    evidence = f"检出{LABEL_ZH.get(alarm_type, alarm_type)}"
                else:
                    cf, bbox, evidence = 0.6, None, \
                        f"未检出{LABEL_ZH.get(baseline, baseline)}（默认告警）"
                events.append(BehaviorEvent(
                    alarm_type, LABEL_ZH.get(alarm_type, alarm_type), cf,
                    bbox=bbox, severity=SEVERITY_MAP.get(alarm_type, "high"),
                    evidence=evidence))
                raw.append(alarm_type)

        return events, raw, True

    def _from_base(self, r, events, raw):
        """yolov8n.pt(COCO) 回退路径：仅识别 person / cell phone"""
        persons: List[Tuple[List[float], float]] = []
        phones: List[Tuple[List[float], float]] = []
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                cf = float(box.conf.item())
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                if cls_id == COCO_PERSON and cf >= self.conf:
                    persons.append((xyxy, cf))
                elif cls_id == COCO_CELL_PHONE and cf >= self.conf * 0.5:
                    phones.append((xyxy, cf))

        if not persons:
            events.append(BehaviorEvent(
                "no_driver", LABEL_ZH["no_driver"], 0.9,
                severity=SEVERITY_MAP["no_driver"],
                evidence="base: 未检出 person"))
            raw.append("no_driver")
            return events, raw, False

        if phones:
            best = max(phones, key=lambda x: x[1])
            events.append(BehaviorEvent(
                "phone_use", LABEL_ZH["phone_use"], best[1],
                bbox=[int(v) for v in best[0]],
                severity=SEVERITY_MAP["phone_use"],
                evidence="base: COCO cell phone 检出"))
            raw.append("phone_use")

        return events, raw, True

    # ---------- 打包 ----------

    def _finalize(self, frame_id, timestamp, events, start_ts,
                  driver_present, camera_ok, raw):
        stable_set = set(self.smoother.update(raw, timestamp))
        stable_events: List[BehaviorEvent] = []
        for e in events:
            if e.type in stable_set:
                e.duration_s = round(self.smoother.duration(e.type, timestamp), 2)
                stable_events.append(e)

        # 告警等级取风险事件的最高 severity（hand_on_wheel/seatbelt 等"安全"类不算）
        alarm_events = [
            e for e in stable_events
            if e.type not in ("hand_on_wheel", "seatbelt") and e.severity != "low"
        ] or [e for e in stable_events if e.type not in ("hand_on_wheel", "seatbelt")]
        if alarm_events:
            max_sev = max(alarm_events, key=lambda x: _SEV_ORDER[x.severity]).severity
        else:
            max_sev = "none"

        latency_ms = round((time.time() - start_ts) * 1000, 2)
        behaviors_dict = [e.to_dict() for e in stable_events]
        score = compute_risk_score(behaviors_dict)
        return {
            "frame_id": frame_id,
            "timestamp": round(timestamp, 3),
            "latency_ms": latency_ms,
            "mode": self.mode,
            "behaviors": behaviors_dict,
            "alert_level": max_sev,
            "risk_score": score,
            "risk_tier": risk_tier(score),
            "recommendation": RECOMMEND.get(max_sev, "正常驾驶"),
            "driver_present": driver_present,
            "camera_ok": camera_ok,
        }

    # ---------- 可视化 ----------

    def visualize(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """在帧上画出 bbox + 标签，用于调试"""
        vis = frame.copy()
        color_map = {
            "critical": (0, 0, 255),
            "high":     (0, 64, 255),
            "medium":   (0, 165, 255),
            "low":      (0, 255, 255),
        }
        for e in result.get("behaviors", []):
            if not e.get("bbox"):
                continue
            x1, y1, x2, y2 = e["bbox"]
            color = color_map.get(e["severity"], (200, 200, 200))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{e['type']} {e['confidence']:.2f}"
            cv2.putText(vis, label, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        bar = f"score={result.get('risk_score',0):.1f} tier={result.get('risk_tier','')} alert={result.get('alert_level','')}"
        cv2.putText(vis, bar, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return vis
