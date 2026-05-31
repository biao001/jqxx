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
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

import cv2
import numpy as np
from ultralytics import YOLO


# ---------- 常量 ----------

COCO_PERSON = 0
COCO_BOTTLE = 39
COCO_CUP = 41
COCO_CELL_PHONE = 67

# 实时物体追踪：COCO 类 → DMS 行为类型（用追踪框稳定对应行为的显示框）。
# 手机→phone_use；杯/瓶→drinking。smoking 无对应 COCO 类，不追踪。
COCO_TO_BEHAVIOR = {
    COCO_CELL_PHONE: "phone_use",
    COCO_CUP:        "drinking",
    COCO_BOTTLE:     "drinking",
}
COCO_TRACK_CLASSES = tuple(COCO_TO_BEHAVIOR.keys())


def _center_inside(inner, outer) -> bool:
    """inner 框中心是否落在 outer 框内(含少量外扩容差)。用于手机须在人体范围内的空间约束。"""
    if not inner or not outer:
        return False
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    cx, cy = (ix1 + ix2) / 2, (iy1 + iy2) / 2
    pad = 0.1 * max(1.0, ox2 - ox1)  # 边缘 10% 容差，避免贴边手机被误拒
    return (ox1 - pad) <= cx <= (ox2 + pad) and (oy1 - pad) <= cy <= (oy2 + pad)


def _phone_near_face(phone_bbox, person_bbox, upper_ratio: float = 0.55) -> bool:
    """手机中心须落在人体上半身区域（头部/耳部/胸口高度）才判定为使用手机。

    驾驶场景摄像头通常拍到头+上半身，手机放到面部/耳旁才是真正接听/刷屏；
    放在腿上、座位旁、车门口等下半身区域不应触发 phone_use。
    upper_ratio=0.55 表示只看人体 bbox 从顶部往下 55% 的区域。
    """
    if not phone_bbox or not person_bbox:
        return False
    px1, py1, px2, py2 = person_bbox
    fx1, fy1, fx2, fy2 = phone_bbox
    phone_cy = (fy1 + fy2) / 2
    upper_bound = py1 + upper_ratio * max(1.0, py2 - py1)
    return phone_cy <= upper_bound

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
            misses = len(h) - on
            if b in seen and not self.active.get(b, False) and on >= self.activate:
                self.active[b] = True
                self.start_ts[b] = timestamp
            elif self.active.get(b, False) and misses >= self.deactivate:
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
        # hands_off_wheel 不在此直接告警，改走下方反逻辑（检出 hand_on_wheel 才安全）
    }

    # 走"默认告警 / 检出安全基线才解除"逻辑的成对类
    # 安全基线类名 -> (告警 behavior, 检出告警类名)
    # 安全带：默认未系，检出 seatbelt 才解除。
    # 双手不走默认告警逻辑 —— 默认在盘(安全)，只有"在盘→离盘"迁移才告警(见 _hands_event)。
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
        smoking_conf: float = 0.55,
        phone_use_conf: float = 0.55,
        drink_eat_conf: float = 0.55,
    ):
        self.device = device
        self.conf = conf
        # smoking 单独阈值（与 conf 解耦，可高可低）；推理时取两者较小值出框，
        # 再在 _from_unified 里按 smoking_conf 过滤。
        self.smoking_conf = smoking_conf
        # phone_use 接受阈值：COCO 仍用低 phone_verify_conf 推理(保证 person 灵敏)，
        # 但只有手机置信度 >= phone_use_conf 才上报，抑制误报。
        self.phone_use_conf = phone_use_conf
        # drinking/eating 同属"手到嘴边"动作，易把举手机/打电话误判，单独抬高阈值
        self.drink_eat_conf = drink_eat_conf
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

        # activate=1：检出即上报(零确认延迟，更跟手，误报由置信度阈值兜底)；
        # window=6/deactivate=3：需连续/近 3 帧漏检才清除——放宽 gap 容忍，配合下方
        # 「轨迹保持出框」让框在零星漏检的中间帧不消失(避免一闪一闪)。
        self.smoother = TemporalSmoother(window=6, activate=1, deactivate=3)

        # 轨迹保持：实时模式下，某行为仍被平滑器判为"活跃"但这一帧没检到框时，
        # 沿用该类型上次的框继续输出(标 evidence=轨迹保持)，使框连续、不闪。
        # 只在 realtime 用；新流(frame_id=0)清空。
        self._held_box: Dict[str, Tuple[List[int], str, str, float]] = {}

        # 双手离盘状态机：默认在盘(安全)。检出离盘需连续 N 帧才告警(防抖)。
        self.hands_off_streak_min = 2
        self._hands_off_streak = 0

        # ByteTrack 单目标追踪：实时模式用 model.track(persist=True) 给每个框稳定 ID +
        # 卡尔曼运动平滑 + 低分检测关联，使识别框跟手、漏检几帧也不闪。
        # 新视频流(frame_id=0)时重置轨迹，避免上次会话残留。
        self.use_tracker = True
        self.tracker_cfg = "bytetrack.yaml"

        # 物体追踪：实时模式额外用 COCO(yolov8n) 检测手机/水杯/水瓶并做 ByteTrack 追踪，
        # 给物体稳定 ID + 卡尔曼平滑。当 unified 判定对应行为(phone_use/drinking)时，用追踪到的、
        # 且落在 unified 动作框内的物体框替换显示框 —— 框更紧、更跟手、帧间更稳。
        # 触发仍由 unified 决定（场景内召回高，不改触发避免漏检），COCO 只稳框。
        # 注：smoking 无对应 COCO 类，故不追踪香烟。
        self.track_objects = True
        self.obj_track_ttl = 6        # 物体框短暂丢失时沿用上次的最大帧数(桥接漏检不闪)
        self._last_obj_box: Dict[str, Tuple[List[int], float]] = {}
        self._last_obj_frame: Dict[str, int] = {}
        self._obj_fresh: Dict[str, Tuple[List[int], float]] = {}  # 本帧"新鲜"检出(非TTL沿用)，供触发判定

        # 背景人误识别抑制：实时同时检 COCO person，驾驶员=前景(最大)人框。
        # 若某行为框明显落在"背景人"(较小且不在驾驶员框内)身上 → 丢弃(如背后有人拿手机)。
        # 保守门控：只在确信是背景人时才丢，避免误删驾驶员自身动作(见 driver-no-detection 教训)。
        self.suppress_background = True
        self._coco_persons: List[List[int]] = []
        self.bg_person_area_ratio = 0.6  # 面积 < 驾驶员 * 此比例 才算"背景人"(更远更小)

        # COCO 手机参与"触发"：realtime.pt 的 phone_use 只用 26 个同场景框训练、易漏检；
        # COCO yolov8n 是通用手机检测器、泛化强，故当它稳定检到手机而 unified 漏检时补触发。
        # 防误报：置信度达标 + 连续 N 帧 + 手机不在画面最底部(排除腿上/杯架)。
        self.phone_coco_trigger = True
        self.phone_coco_conf = 0.25    # COCO 手机触发用的置信度门槛(高于追踪用的 0.15)
        self.phone_streak_min = 2      # 连续检到 N 帧才触发(防单帧噪声误报)
        self.phone_upper_ratio = 0.85  # 手机中心须在画面上方 85% 区域(排除最底部)
        self._phone_streak = 0

    def _reset_tracker(self) -> None:
        """重置 ByteTrack 轨迹状态(新流开始时调用)，清除上次会话的旧轨迹。"""
        self._reset_model_tracker(self.model)

    @staticmethod
    def _reset_model_tracker(model) -> None:
        try:
            predictor = getattr(model, "predictor", None)
            trackers = getattr(predictor, "trackers", None) if predictor else None
            for t in (trackers or []):
                if hasattr(t, "reset"):
                    t.reset()
        except Exception:
            pass

    def _track_coco_objects(self, frame: np.ndarray, frame_id: int) -> Dict[str, Tuple[List[int], float]]:
        """实时模式：用 COCO(yolov8n) 检测+ByteTrack 追踪手机/水杯/水瓶，返回每个行为类型
        最高 conf 的物体框 {behavior_type: (bbox, conf)}。短暂丢失时 TTL 内沿用上次框防闪。

        只追踪 COCO_TRACK_CLASSES（手机/杯/瓶），更快且不被 person 等干扰；每帧调用以保持轨迹连续。
        """
        if self.phone_verify is None:
            return {}
        if frame_id == 0:
            self._reset_model_tracker(self.phone_verify)
            self._last_obj_box = {}
            self._last_obj_frame = {}
            self._phone_streak = 0
        try:
            # 同时检 person(class 0)：用于区分驾驶员(前景最大人)与背景人，抑制背景误识别。
            r = self.phone_verify.track(
                frame, conf=self.phone_verify_conf, iou=self.iou, imgsz=self.imgsz,
                verbose=False, device=self.device, persist=True,
                tracker=self.tracker_cfg, classes=list(COCO_TRACK_CLASSES) + [COCO_PERSON])[0]
        except Exception:
            self._coco_persons = []
            return {}
        # 本帧每个行为类型保留最高 conf 的物体框；person 单独收集
        seen: Dict[str, Tuple[List[int], float]] = {}
        persons: List[List[int]] = []
        if r.boxes is not None:
            for b in r.boxes:
                cid = int(b.cls.item())
                cf = float(b.conf.item())
                xyxy = [int(v) for v in b.xyxy[0].cpu().numpy().tolist()]
                if cid == COCO_PERSON:
                    if cf >= 0.35:  # person 用稍高阈值，避免杂框
                        persons.append(xyxy)
                    continue
                btype = COCO_TO_BEHAVIOR.get(cid)
                if btype is None:
                    continue
                if btype not in seen or cf > seen[btype][1]:
                    seen[btype] = (xyxy, cf)
        self._coco_persons = persons  # 供背景人抑制使用
        self._obj_fresh = seen  # 本帧新鲜检出(供触发判定，区别于下方 TTL 沿用)
        # 更新本帧检出的类型；未检出的类型在 TTL 内沿用上次框
        out: Dict[str, Tuple[List[int], float]] = {}
        for btype, box in seen.items():
            self._last_obj_box[btype] = box
            self._last_obj_frame[btype] = frame_id
            out[btype] = box
        for btype, box in self._last_obj_box.items():
            if btype not in out and (frame_id - self._last_obj_frame.get(btype, -999)) <= self.obj_track_ttl:
                out[btype] = box
        return out

    def _update_phone_streak(self, frame_h: int) -> None:
        """根据本帧 COCO 手机"新鲜检出"更新连续命中计数(用于 COCO 触发兜底)。

        命中条件：本帧检到手机 + 置信度≥phone_coco_conf + 中心在画面上方 phone_upper_ratio
        区域(排除腿上/杯架的手机)。命中则 +1，否则衰减归零(连续性要求过滤单帧噪声)。
        """
        fresh = (self._obj_fresh or {}).get("phone_use")
        ok = False
        if fresh is not None:
            (x1, y1, x2, y2), cf = fresh
            cy = (y1 + y2) / 2.0
            if cf >= self.phone_coco_conf and cy <= self.phone_upper_ratio * max(1, frame_h):
                ok = True
        self._phone_streak = self._phone_streak + 1 if ok else 0

    def _suppress_background(self, events: List["BehaviorEvent"], raw: List[str]):
        """丢弃明显属于"背景人"的行为(如背后有人拿手机)。

        驾驶员=前景(面积最大)人框。某行为框中心若落在"背景人"(面积 < 驾驶员*bg_person_area_ratio)
        内、且不在驾驶员框内 → 判为背景误识别丢弃。无人框/只有驾驶员时不处理(保守，保召回)。
        """
        persons = self._coco_persons or []
        if len(persons) < 2:
            return events, raw  # 没有第二个人，谈不上背景人误识别
        def area(p):
            return max(0, p[2] - p[0]) * max(0, p[3] - p[1])
        driver = max(persons, key=area)
        d_area = area(driver) or 1
        bg = [p for p in persons if p is not driver and area(p) < self.bg_person_area_ratio * d_area]
        if not bg:
            return events, raw

        def center_in(box, outer):
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]

        kept: List[BehaviorEvent] = []
        for e in events:
            if e.bbox and not center_in(e.bbox, driver) and any(center_in(e.bbox, p) for p in bg):
                continue  # 在背景人身上、且不在驾驶员框内 → 丢弃
            kept.append(e)
        kept_types = {e.type for e in kept}
        raw = [t for t in raw if t in kept_types or t in ("hand_on_wheel", "seatbelt", "no_seatbelt")]
        return kept, raw

    def _coco_scan(self, frame: np.ndarray) -> Dict[str, Optional[Tuple[List[int], float]]]:
        """COCO(yolov8n) 单次推理，返回最高 conf 的 person / cell phone (bbox, conf)。

        unified.pt 没有 person 类，且对车外/通用场景的 phone_use 很弱，
        因此用 COCO 通用模型补充：
          - person → 真正判断驾驶位是否有人（替代"模型无输出=无人"的误判）
          - cell phone → 直接驱动 phone_use（举手机即可识别，不再只做抑制）
        """
        out: Dict[str, Any] = {"person": None, "phone": None, "persons": []}
        if self.phone_verify is None:
            return out
        r = self.phone_verify(frame, conf=self.phone_verify_conf, iou=self.iou,
                              imgsz=self.imgsz, verbose=False, device=self.device)[0]
        if r.boxes is None:
            return out
        for b in r.boxes:
            cid = int(b.cls.item())
            key = "person" if cid == COCO_PERSON else "phone" if cid == COCO_CELL_PHONE else None
            if key is None:
                continue
            cf = float(b.conf.item())
            xyxy = [int(v) for v in b.xyxy[0].cpu().numpy().tolist()]
            if key == "person":
                out["persons"].append(xyxy)  # 收集全部人体框(用于驾驶员 ROI 收口)
            if out[key] is None or cf > out[key][1]:
                out[key] = (xyxy, cf)
        return out

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
                timestamp: Optional[float] = None, realtime: bool = False) -> Dict:
        """realtime=True 走实时监测精简逻辑（只检危险动作、即时响应）；
        realtime=False 走上传视频完整逻辑（含无人/安全带/方向盘）。"""
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

        # 1. YOLO 推理。用较低阈值，保留 smoking 弱检测；其余类在 _from_unified 按 self.conf 过滤。
        infer_frame = frame
        if self.low_light_enhance and self._is_low_light(frame):
            infer_frame = self._enhance_low_light(frame)
        infer_conf = min(self.conf, self.smoking_conf)
        if realtime and self.use_tracker:
            # 实时流：ByteTrack 追踪(稳定ID+卡尔曼平滑+低分关联)，框跟手、漏检不闪。
            if frame_id == 0:
                self._reset_tracker()  # 新流重置轨迹
                self._held_box = {}    # 清空轨迹保持框，避免上次会话残留
            r = self.model.track(infer_frame, conf=infer_conf, iou=self.iou,
                                 imgsz=self.imgsz, verbose=False, device=self.device,
                                 persist=True, tracker=self.tracker_cfg)[0]
        else:
            # 上传视频是跳帧采样，帧不连续，追踪无意义 → 普通检测
            r = self.model(infer_frame, conf=infer_conf, iou=self.iou,
                           imgsz=self.imgsz, verbose=False, device=self.device)[0]

        persons: List[List[int]] = []
        if self.mode == "unified":
            # 有效检测（已按各类阈值过滤），杂框不计入——这才是"有没有驾驶员行为"的依据
            by_name = self._build_by_name(r)
            # COCO 仅用于视频模式判断"驾驶位无人"：实时模式不判无人，故永不跑；
            # 视频模式也只在 unified 无有效检测时才跑（有有效检测=有驾驶员，跳过省 ~6ms）。
            need_coco = (not realtime) and (len(by_name) == 0)
            coco = self._coco_scan(infer_frame) if need_coco else {"person": None, "phone": None, "persons": []}
            persons = coco.get("persons", [])
            # 实时：每帧追踪 COCO 物体(手机/杯/瓶，保持轨迹连续)，供下方稳定显示框
            obj_tracks = (self._track_coco_objects(infer_frame, frame_id)
                          if (realtime and self.track_objects) else {})
            # 实时把 COCO person 框透出(供上层判定驾驶位无人/区分背景人)
            if realtime and self.track_objects:
                persons = self._coco_persons
            # 更新手机"连续检出"计数(用本帧新鲜检出，不含 TTL 沿用)，供 COCO 触发判定
            if realtime and self.phone_coco_trigger:
                self._update_phone_streak(infer_frame.shape[0])
            events, raw, driver_present = self._from_unified(by_name, events, raw, coco, realtime=realtime)
            # phone_use：unified(realtime.pt) 专门训练了此类，直接信任其作为触发；
            # 但显示框优先用 COCO 追踪到的手机框(更紧、更跟手、帧间更稳)。
            phone_events = [e for e in events if e.type == "phone_use"]
            if phone_events:
                # 过滤：手机 bbox 中心须在其自身检出框上半区（防止低位误检）
                # unified bbox 是行为整体框，通常已含人体上半身，保留置信度门控即可
                events = [e for e in events if e.type != "phone_use"]
                raw = [t for t in raw if t != "phone_use"]
                phone_track = obj_tracks.get("phone_use")
                for pe in phone_events:
                    if pe.confidence >= self.phone_use_conf:
                        # 追踪到的手机框若落在 unified 动作框内 → 用它替换显示框(焊在手机上)
                        if phone_track is not None and pe.bbox and _center_inside(phone_track[0], pe.bbox):
                            pe.bbox = list(phone_track[0])
                            pe.evidence = (pe.evidence + " + COCO手机追踪框") if pe.evidence else "COCO手机追踪框"
                        events.append(pe)
                        raw.append("phone_use")
            # COCO 触发兜底：unified 漏检 phone_use 时，若 COCO 稳定检到手机(连续≥N帧)，补一个。
            # 解决 realtime.pt phone_use 训练样本少、泛化弱导致的漏检。框用 COCO 手机框。
            if (realtime and self.phone_coco_trigger
                    and not any(e.type == "phone_use" for e in events)
                    and self._phone_streak >= self.phone_streak_min):
                pbox, pconf = obj_tracks.get("phone_use") or (None, 0.0)
                if pbox is not None:
                    events.append(BehaviorEvent(
                        "phone_use", LABEL_ZH["phone_use"], max(pconf, self.phone_use_conf),
                        bbox=list(pbox), severity=SEVERITY_MAP["phone_use"],
                        evidence="COCO手机检出(unified漏检补充)"))
                    raw.append("phone_use")
            # 有 phone_use(任一来源) → 抑制易混淆的 drinking/eating
            if any(e.type == "phone_use" for e in events):
                events = [e for e in events if e.type not in ("drinking", "eating")]
                raw = [t for t in raw if t not in ("drinking", "eating")]
            # drinking：同理用 COCO 追踪到的水杯/水瓶框稳定显示框(触发仍靠 unified)
            drink_track = obj_tracks.get("drinking")
            if drink_track is not None:
                for e in events:
                    if e.type == "drinking" and e.bbox and _center_inside(drink_track[0], e.bbox):
                        e.bbox = list(drink_track[0])
                        e.evidence = (e.evidence + " + COCO水杯追踪框") if e.evidence else "COCO水杯追踪框"
            # 背景人误识别抑制(如背后有人拿手机)：丢弃明显落在背景人身上的行为
            if realtime and self.suppress_background:
                events, raw = self._suppress_background(events, raw)
        else:
            events, raw, driver_present = self._from_base(r, events, raw)

        return self._finalize(frame_id, timestamp, events, t0,
                              driver_present=driver_present, camera_ok=True,
                              raw=raw, persons=persons, hold=realtime)

    # ---------- 类别 → BehaviorEvent ----------

    def _hands_event(self, by_name, events, raw):
        """双手在盘/离盘状态机。

        默认在盘(安全)；检出 hand_on_wheel → 重置为安全；
        检出 hands_off_wheel(且未同时检出 hand_on_wheel) → 视为"在盘→离盘"迁移，
        需连续 hands_off_streak_min 帧才告警，避免瞬时误检导致频繁误报。
        """
        on_w = "hand_on_wheel" in by_name
        off_w = "hands_off_wheel" in by_name
        if on_w and not off_w:               # 明确在盘
            self._hands_off_streak = 0
            return
        if off_w and not on_w:               # 明确离盘 → 累积迁移
            self._hands_off_streak += 1
            if self._hands_off_streak >= self.hands_off_streak_min:
                cf, xyxy = by_name["hands_off_wheel"]
                events.append(BehaviorEvent(
                    "hands_off_wheel", LABEL_ZH["hands_off_wheel"], cf,
                    bbox=[int(v) for v in xyxy],
                    severity=SEVERITY_MAP["hands_off_wheel"],
                    evidence="在盘→离盘 迁移"))
                raw.append("hands_off_wheel")
            return
        # 两者都检出 / 都没检出 → 不明确，缓慢回落到安全
        self._hands_off_streak = max(0, self._hands_off_streak - 1)

    def _build_by_name(self, r) -> Dict[str, Tuple[float, List[float]]]:
        """按类别名归并 unified 输出，每类保留最高 conf 的有效框（已按分级阈值过滤）。

        注意：推理时用低阈值 min(conf, smoking_conf) 出框以保灵敏，这里才是真正的
        "有效检测"判定——低于各类阈值的杂框在此被剔除。判断"是否有驾驶员"必须基于
        本方法的结果，而不是原始框数量（杂框不代表有人）。
        """
        by_name: Dict[str, Tuple[float, List[float]]] = {}
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                cf = float(box.conf.item())
                name = (self.class_names.get(cls_id) or "").lower()
                if not name:
                    continue
                # 按类分级阈值：smoking 用 smoking_conf；drinking/eating 用更高的
                # drink_eat_conf(抑制举手机/打电话被误判成喝水/进食)；其余类用 self.conf。
                if name == "smoking":
                    min_conf = self.smoking_conf
                elif name in ("drinking", "eating"):
                    min_conf = self.drink_eat_conf
                else:
                    min_conf = self.conf
                if cf < min_conf:
                    continue
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                if name not in by_name or cf > by_name[name][0]:
                    by_name[name] = (cf, xyxy)
        return by_name

    def _from_unified(self, by_name, events, raw, coco=None, realtime=False):
        """从 unified 8 类有效检测(by_name)生成 BehaviorEvent 列表。

        realtime=True（实时监测）：只输出危险动作(phone_use/smoking/drinking/eating)，
            不判定"驾驶位无人"、安全带、方向盘——即时响应，避免误报。
        realtime=False（上传视频）：完整逻辑，含 no_driver / 安全带 / 双手离盘。
        """
        # 完全无检测：unified 没有 person 类，不能直接当成"驾驶位无人"。
        if not by_name:
            if realtime:
                # 实时模式不判定"驾驶位无人"，恒视为有驾驶员（不产生告警）
                return events, raw, True
            # 视频模式：用 COCO 确认画面里到底有没有人——确实无人才报 no_driver。
            person = coco.get("person") if coco else None
            if person is None:
                events.append(BehaviorEvent(
                    "no_driver", LABEL_ZH["no_driver"], 0.9,
                    severity=SEVERITY_MAP["no_driver"],
                    evidence="unified 未检出任何类，且 COCO 未检出 person"))
                raw.append("no_driver")
                return events, raw, False
            # 有人但未检出具体行为 → 正常驾驶，不产生告警事件
            return events, raw, True

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

        # 实时模式到此结束：只要危险动作，不判定方向盘/安全带（即时响应）
        if realtime:
            return events, raw, True

        # 双手离盘：状态迁移逻辑（默认在盘=安全，仅"在盘→离盘"才告警）
        self._hands_event(by_name, events, raw)

        # 安全带：默认告警，检出 seatbelt 才解除
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
                  driver_present, camera_ok, raw, persons=None, hold=False):
        stable_set = set(self.smoother.update(raw, timestamp))
        stable_events: List[BehaviorEvent] = []
        present: set = set()
        for e in events:
            if e.type in stable_set:
                e.duration_s = round(self.smoother.duration(e.type, timestamp), 2)
                stable_events.append(e)
                present.add(e.type)
                if e.bbox:  # 记住每类最近一次的框，供"轨迹保持"补帧
                    self._held_box[e.type] = (list(e.bbox), e.label_zh, e.severity, float(e.confidence))

        # 轨迹保持(仅 realtime)：某类型仍活跃但本帧没检到框 → 沿用上次框继续输出，
        # 让框在零星漏检的中间帧不消失(连续平滑、不闪)。频繁漏检由 deactivate 兜底清除。
        if hold:
            for t in stable_set:
                if t not in present and t in self._held_box:
                    bbox, label, sev, conf = self._held_box[t]
                    stable_events.append(BehaviorEvent(
                        t, label, conf, bbox=list(bbox), severity=sev,
                        duration_s=round(self.smoother.duration(t, timestamp), 2),
                        evidence="轨迹保持(本帧漏检，沿用上次框)"))
            # 平滑器已判不活跃的类型，清掉其保持框，避免下次复现时用到旧位置
            for t in list(self._held_box):
                if t not in stable_set:
                    self._held_box.pop(t, None)

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
            "persons": persons or [],
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
