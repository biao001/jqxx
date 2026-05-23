from __future__ import annotations

import base64
import importlib.util
import json
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from .config import Settings
from .llm import generate_llm_analysis
from .reports import write_markdown_report
from .schemas import BEHAVIOR_LABELS, FATIGUE_LABELS, RISK_TO_SEVERITY
from .scoring import compute_driving_stats, normalize_detection, risk_level_to_score


def decode_image_bytes(payload: bytes) -> np.ndarray:
    if payload.startswith(b"data:image"):
        payload = payload.split(b",", 1)[1]
        payload = base64.b64decode(payload)
    image_array = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("无法解码图像帧")
    return frame


def encode_upload_name(filename: str | None) -> str:
    clean = Path(filename or "upload.mp4").name
    return clean or "upload.mp4"


class BehaviorRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.detector: Any | None = None
        self.load_error: str | None = None
        self.last_mode = "not_loaded"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def capability(self) -> dict[str, Any]:
        if self.detector is not None and self.last_mode == "model":
            return {"name": "behavior_detector", "mode": "model", "error": None}
        if self.load_error:
            return {"name": "behavior_detector", "mode": "fallback", "error": self.load_error}
        if importlib.util.find_spec("ultralytics") is None:
            return {
                "name": "behavior_detector",
                "mode": "fallback",
                "error": "Missing dependency: ultralytics. Use the Python environment from backend/requirements.txt.",
            }
        return {"name": "behavior_detector", "mode": self.last_mode, "error": None}

    def _load_detector(self) -> Any | None:
        if self.detector is not None or self.load_error is not None:
            return self.detector
        behavior_dir = self.settings.repo_root / "behavior_algo"
        sys.path.insert(0, str(behavior_dir))
        try:
            from behavior_detector import BehaviorDetector

            self.detector = BehaviorDetector(
                unified_weights=str(behavior_dir / "models" / "unified.pt"),
                base_weights=str(self.settings.repo_root / "yolov8n.pt"),
                device="cpu",
                imgsz=640,
            )
        except Exception as exc:
            self.load_error = str(exc)
            self.detector = None
            self.last_mode = "fallback"
        finally:
            try:
                sys.path.remove(str(behavior_dir))
            except ValueError:
                pass
        return self.detector

    def analyze(self, frame: np.ndarray, frame_id: int, timestamp: float) -> dict[str, Any]:
        detector = self._load_detector()
        if detector is not None:
            try:
                result = detector.predict(frame, frame_id=frame_id, timestamp=timestamp)
                self.last_mode = "model"
                result["capability"] = self.capability()
                return result
            except Exception as exc:
                self.load_error = str(exc)

        result = self._fallback(frame, frame_id, timestamp)
        self.last_mode = "fallback"
        result["capability"] = self.capability()
        return result

    def _fallback(self, frame: np.ndarray, frame_id: int, timestamp: float) -> dict[str, Any]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        events: list[dict[str, Any]] = []
        camera_ok = mean >= 15.0 and blur >= 8.0

        if not camera_ok:
            events.append(
                {
                    "type": "lens_covered",
                    "label_zh": BEHAVIOR_LABELS["lens_covered"],
                    "confidence": 0.95,
                    "severity": "medium",
                    "duration_s": 0.0,
                    "evidence": "图像亮度或清晰度低于阈值",
                }
            )

        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        driver_present = len(faces) > 0
        if camera_ok and not driver_present:
            events.append(
                {
                    "type": "no_driver",
                    "label_zh": BEHAVIOR_LABELS["no_driver"],
                    "confidence": 0.72,
                    "severity": "critical",
                    "duration_s": 0.0,
                    "evidence": "轻量人脸检测未发现驾驶员面部",
                }
            )

        behavior_risk = 0.0
        if events:
            severity_factor = {"medium": 40.0, "critical": 90.0}
            behavior_risk = max(severity_factor.get(event["severity"], 20.0) for event in events)

        return {
            "frame_id": frame_id,
            "timestamp": round(timestamp, 3),
            "latency_ms": 0.0,
            "behaviors": events,
            "alert_level": events[0]["severity"] if events else "none",
            "risk_score": behavior_risk,
            "risk_tier": "warning" if behavior_risk >= 40 else "safe",
            "recommendation": "请检查摄像头画面和驾驶员状态" if events else "正常驾驶",
            "driver_present": driver_present,
            "camera_ok": camera_ok,
        }


class FatigueRuntime:
    def __init__(self, settings: Settings, window_size: int = 16):
        self.settings = settings
        self.window_size = window_size
        self.feature_window: deque[np.ndarray] = deque(maxlen=window_size)
        # EAR 历史(约 3 秒)用于 PERCLOS 闭眼率统计
        self.ear_history: deque[float] = deque(maxlen=90)
        self.predictor: Any | None = None
        self.predictor_error: str | None = None
        self.feature_error: str | None = None
        self.last_mode = "not_loaded"
        self.thresholds = {"yawn": 0.55, "look_away": 0.55, "fatigue": 0.60}
        self.feature_tools: Any | None = None
        self.landmarker: Any | None = None
        self.frontal_cascade: Any | None = None
        self.profile_cascade: Any | None = None
        self._last_mp_timestamp_ms = -1
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._load_feature_extractor()
        self._load_predictor()

    @property
    def load_error(self) -> str | None:
        errors = [error for error in (self.feature_error, self.predictor_error) if error]
        return "; ".join(errors) if errors else None

    def capability(self) -> dict[str, Any]:
        feature_mode = "mediapipe" if self.feature_tools is not None and self.landmarker is not None else "fallback"
        predictor_mode = "model" if self.predictor is not None else "heuristic"
        if predictor_mode == "model" and feature_mode == "mediapipe":
            mode = "model"
        elif predictor_mode == "model":
            mode = "partial"
        else:
            mode = "fallback"
        return {
            "name": "fatigue_b",
            "mode": mode,
            "error": self.load_error,
            "predictor": predictor_mode,
            "feature_extractor": feature_mode,
        }

    def _load_feature_extractor(self) -> None:
        try:
            fatigue_root = self.settings.repo_root / "fatigue"
            sys.path.insert(0, str(fatigue_root))
            from src.fatigue_b import extract_features as feature_module

            model_path = feature_module.ensure_face_landmarker_model(
                fatigue_root / "models" / "mediapipe" / "face_landmarker.task"
            )
            self.landmarker = feature_module.create_landmarker(model_path)
            self.frontal_cascade, self.profile_cascade = feature_module.create_fallback_cascades()
            self.feature_tools = SimpleNamespace(
                mp=feature_module.mp,
                left_eye=feature_module.LEFT_EYE,
                right_eye=feature_module.RIGHT_EYE,
                eye_aspect_ratio=feature_module.eye_aspect_ratio,
                mouth_aspect_ratio=feature_module.mouth_aspect_ratio,
                estimate_head_pose=feature_module.estimate_head_pose,
                estimate_drowsy_prob=feature_module.estimate_drowsy_prob,
                detect_face_bbox_fallback=feature_module.detect_face_bbox_fallback,
                approximate_mouth_aspect_ratio=feature_module.approximate_mouth_aspect_ratio,
            )
            self.feature_error = None
        except Exception as exc:
            self.feature_error = str(exc)
            self.landmarker = None
            self.feature_tools = None
            self.frontal_cascade = self.face_cascade
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
        finally:
            try:
                sys.path.remove(str(self.settings.repo_root / "fatigue"))
            except ValueError:
                pass

    def _load_predictor(self) -> None:
        checkpoint = self.settings.repo_root / "fatigue" / "outputs" / "fatigue_b" / "checkpoints" / "best_model.pt"
        if not checkpoint.exists():
            self.predictor_error = "fatigue checkpoint not found"
            return
        try:
            fatigue_root = self.settings.repo_root / "fatigue"
            sys.path.insert(0, str(fatigue_root))
            from src.fatigue_b.infer import FatigueBPredictor

            self.predictor = FatigueBPredictor(checkpoint, device="auto")
            self.thresholds.update(getattr(self.predictor, "thresholds", {}) or {})
            self.predictor_error = None
        except Exception as exc:
            self.predictor_error = str(exc)
            self.predictor = None
        finally:
            try:
                sys.path.remove(str(self.settings.repo_root / "fatigue"))
            except ValueError:
                pass

    def _next_mp_timestamp_ms(self, timestamp: float | None) -> int:
        candidate = int((time.time() if timestamp is None else timestamp) * 1000.0)
        if candidate <= self._last_mp_timestamp_ms:
            candidate = self._last_mp_timestamp_ms + 1
        self._last_mp_timestamp_ms = candidate
        return candidate

    def extract_features(self, frame: np.ndarray, timestamp: float | None = None) -> np.ndarray:
        if self.feature_tools is not None and self.landmarker is not None:
            return self._extract_mediapipe_features(frame, timestamp)
        return self._extract_legacy_features(frame)

    def _extract_mediapipe_features(self, frame: np.ndarray, timestamp: float | None) -> np.ndarray:
        tools = self.feature_tools
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = tools.mp.Image(image_format=tools.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, self._next_mp_timestamp_ms(timestamp))

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            points = np.array([[point.x * width, point.y * height] for point in landmarks], dtype=np.float32)
            left_ear = tools.eye_aspect_ratio(points, tools.left_eye)
            right_ear = tools.eye_aspect_ratio(points, tools.right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = tools.mouth_aspect_ratio(points)
            head_yaw, head_pitch, head_roll = tools.estimate_head_pose(points, width, height)
            drowsy_prob = tools.estimate_drowsy_prob(ear, mar, head_pitch)
        else:
            bbox, is_profile = tools.detect_face_bbox_fallback(frame, self.frontal_cascade, self.profile_cascade)
            if bbox is None:
                ear = mar = head_yaw = head_pitch = head_roll = drowsy_prob = 0.0
            else:
                ear = 0.25
                mar = tools.approximate_mouth_aspect_ratio(frame, bbox)
                head_yaw = 25.0 if is_profile else 0.0
                head_pitch = 0.0
                head_roll = 0.0
                drowsy_prob = tools.estimate_drowsy_prob(ear, mar, head_pitch)

        return np.array(
            [
                round(float(ear), 6),
                round(float(mar), 6),
                round(float(head_yaw), 6),
                round(float(head_pitch), 6),
                round(float(head_roll), 6),
                round(float(drowsy_prob), 6),
            ]
        )

    def _extract_legacy_features(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        if len(faces) == 0:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        roi = gray[y + int(h * 0.52) : y + int(h * 0.88), x + int(w * 0.18) : x + int(w * 0.82)]
        dark_ratio = float((roi < 55).mean()) if roi.size else 0.0
        mar = float(0.18 + dark_ratio * 1.2)
        face_center = x + w / 2.0
        yaw = float(((face_center / max(1, frame.shape[1])) - 0.5) * 50.0)
        ear = 0.24
        drowsy_prob = float(np.clip((mar - 0.45) * 1.5 + max(0.0, abs(yaw) - 15.0) / 35.0, 0.0, 1.0))
        return np.array([ear, mar, yaw, 0.0, 0.0, drowsy_prob], dtype=np.float32)

    def analyze_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> dict[str, Any]:
        features = self.extract_features(frame, timestamp=timestamp)
        self.feature_window.append(features)
        self.ear_history.append(float(features[0]))  # 累积 EAR 供 PERCLOS
        return self._predict_from_window(frame_id, timestamp)

    # EAR 低于此值视为闭眼
    EYE_CLOSED_EAR = 0.18

    def _gaze_zone(self, yaw: float, pitch: float, ear: float) -> str:
        if 0 < ear < 0.13:
            return "闭眼"
        if abs(pitch) > 18:
            return "低头/点头"
        if yaw > 20:
            return "右顾"
        if yaw < -20:
            return "左盼"
        return "前方"

    def _observable_signals(self, window: np.ndarray) -> dict[str, Any]:
        """从特征窗口 + EAR 历史导出可观测信号：PERCLOS、头部姿态、视线区域。"""
        latest = window[-1]
        ear, _mar, yaw, pitch, roll = (float(latest[i]) for i in range(5))
        valid = [e for e in self.ear_history if e > 0]  # 0 表示无脸，剔除
        perclos = sum(1 for e in valid if e < self.EYE_CLOSED_EAR) / len(valid) if valid else 0.0
        return {
            "perclos": round(perclos, 4),
            "ear": round(ear, 4),
            "eyes_closed": bool(0 < ear < self.EYE_CLOSED_EAR),
            "head_yaw": round(yaw, 1),
            "head_pitch": round(pitch, 1),
            "head_roll": round(roll, 1),
            "gaze_zone": self._gaze_zone(yaw, pitch, ear),
        }

    def _apply_perclos(self, result: dict[str, Any], signals: dict[str, Any]) -> dict[str, Any]:
        """PERCLOS 是疲劳的金标准指标：持续闭眼占比过高时强制升级为高风险疲劳。"""
        perclos = float(signals.get("perclos", 0.0))
        if perclos >= 0.45:
            result["risk_level"] = "high"
            if result.get("label") in (None, "Normal", "Looking Around"):
                result["label"] = "Fatigued Driving"
            ind = dict(result.get("indicators") or {})
            ind["fatigue_score"] = round(max(float(ind.get("fatigue_score", 0.0)), min(1.0, perclos + 0.3)), 4)
            result["indicators"] = ind
        return result

    def _predict_from_window(self, frame_id: int, timestamp: float) -> dict[str, Any]:
        if not self.feature_window:
            window = np.zeros((self.window_size, 6), dtype=np.float32)
        else:
            rows = list(self.feature_window)
            while len(rows) < self.window_size:
                rows.append(rows[-1])
            window = np.stack(rows[-self.window_size :], axis=0)

        signals = self._observable_signals(window)

        if self.predictor is not None:
            try:
                result = self.predictor.predict_window(window, frame_id=frame_id, timestamp=timestamp)
                result = self._merge_observable_scores(result, window)
                self.last_mode = "model"
                result["capability"] = self.capability()
                result["signals"] = signals
                return self._apply_perclos(result, signals)
            except Exception as exc:
                self.predictor_error = str(exc)
                self.predictor = None

        yawn_score, look_score, fatigue_score = self._heuristic_scores(window)
        if yawn_score >= self.thresholds["yawn"]:
            label = "Yawning"
        elif look_score >= self.thresholds["look_away"]:
            label = "Looking Around"
        elif fatigue_score >= self.thresholds["fatigue"]:
            label = "Fatigued Driving"
        else:
            label = "Normal"
        risk_level = (
            "high"
            if fatigue_score >= self.thresholds["fatigue"]
            else "medium"
            if (yawn_score >= self.thresholds["yawn"] or look_score >= self.thresholds["look_away"])
            else "low"
        )
        public_scores = {
            "Normal": float(np.clip(1.0 - max(yawn_score, look_score, fatigue_score), 0.0, 1.0)),
            "Yawning": yawn_score,
            "Looking Around": look_score,
            "Fatigued Driving": fatigue_score,
        }
        denominator = sum(public_scores.values())
        confidence = public_scores[label] / denominator if denominator > 0 else 0.0
        self.last_mode = "fallback"
        return self._apply_perclos({
            "module": "fatigue",
            "model_name": "fatigue_b_fallback",
            "frame_id": frame_id,
            "timestamp": round(timestamp, 4),
            "label": label,
            "confidence": round(float(confidence), 4),
            "indicators": {
                "yawn_score": round(yawn_score, 4),
                "look_away_score": round(look_score, 4),
                "fatigue_score": round(fatigue_score, 4),
            },
            "risk_level": risk_level,
            "public_scores": {key: round(float(value), 4) for key, value in public_scores.items()},
            "signals": signals,
            "capability": self.capability(),
        }, signals)

    def _merge_observable_scores(self, result: dict[str, Any], window: np.ndarray) -> dict[str, Any]:
        observed_yawn, observed_look, observed_fatigue = self._heuristic_scores(window)
        indicators = dict(result.get("indicators") or {})
        yawn_score = max(float(indicators.get("yawn_score", 0.0)), observed_yawn)
        look_score = max(float(indicators.get("look_away_score", 0.0)), observed_look)
        fatigue_score = max(float(indicators.get("fatigue_score", 0.0)), observed_fatigue)
        summary = self._public_result(yawn_score, look_score, fatigue_score)
        merged = dict(result)
        merged.update(summary)
        return merged

    def _public_result(self, yawn_score: float, look_score: float, fatigue_score: float) -> dict[str, Any]:
        if yawn_score >= self.thresholds["yawn"]:
            label = "Yawning"
        elif look_score >= self.thresholds["look_away"]:
            label = "Looking Around"
        elif fatigue_score >= self.thresholds["fatigue"]:
            label = "Fatigued Driving"
        else:
            label = "Normal"

        risk_level = (
            "high"
            if fatigue_score >= self.thresholds["fatigue"]
            else "medium"
            if (yawn_score >= self.thresholds["yawn"] or look_score >= self.thresholds["look_away"])
            else "low"
        )
        public_scores = {
            "Normal": float(np.clip(1.0 - max(yawn_score, look_score, fatigue_score), 0.0, 1.0)),
            "Yawning": yawn_score,
            "Looking Around": look_score,
            "Fatigued Driving": fatigue_score,
        }
        denominator = sum(public_scores.values())
        confidence = public_scores[label] / denominator if denominator > 0 else 0.0
        return {
            "label": label,
            "confidence": round(float(confidence), 4),
            "indicators": {
                "yawn_score": round(float(yawn_score), 4),
                "look_away_score": round(float(look_score), 4),
                "fatigue_score": round(float(fatigue_score), 4),
            },
            "risk_level": risk_level,
            "public_scores": {key: round(float(value), 4) for key, value in public_scores.items()},
        }

    @staticmethod
    def _heuristic_scores(window: np.ndarray) -> tuple[float, float, float]:
        valid_face = (window[:, 0] > 0.0) | (window[:, 1] > 0.0) | (np.abs(window[:, 2]) > 0.0) | (window[:, 5] > 0.0)
        if not np.any(valid_face):
            return 0.0, 0.0, 0.0

        ear = window[:, 0]
        mar = window[:, 1]
        head_yaw = np.abs(window[:, 2])
        head_pitch = window[:, 3]
        drowsy_prob = window[:, 5]

        yawn_signal = np.clip((mar - 0.48) / 0.25, 0.0, 1.0)
        look_signal = np.clip((head_yaw - 12.0) / 18.0, 0.0, 1.0)
        low_ear_score = np.clip((0.20 - ear) / 0.07, 0.0, 1.0)
        eye_closure_ratio = np.clip(np.mean(ear < 0.18) * 2.4, 0.0, 1.0)
        pitch_center = float(np.median(head_pitch))
        pitch_delta = np.abs(head_pitch - pitch_center)
        nod_score = np.clip((pitch_delta - 8.0) / 10.0, 0.0, 1.0)
        drowsy_signal = np.clip((drowsy_prob - 0.55) / 0.35, 0.0, 1.0)
        fatigue_signal = (
            0.50 * eye_closure_ratio
            + 0.25 * float(np.percentile(low_ear_score, 90))
            + 0.15 * float(np.percentile(nod_score, 90))
            + 0.10 * float(np.percentile(drowsy_signal, 90))
        )

        return (
            float(np.max(yawn_signal).clip(0.0, 1.0)),
            float(np.max(look_signal).clip(0.0, 1.0)),
            float(np.clip(fatigue_signal, 0.0, 1.0)),
        )


class AnalysisService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.behavior = BehaviorRuntime(settings)
        self.fatigue = FatigueRuntime(settings)
        self.recent_detections: deque[dict[str, Any]] = deque(maxlen=50)
        self.last_llm_at = time.time()
        self.last_llm_text = "等待足够的真实检测数据生成大模型分析。"
        self.llm_inflight = False

    def capabilities(self) -> dict[str, Any]:
        return {
            "behavior": self.behavior.capability(),
            "fatigue": self.fatigue.capability(),
            "llm": self._llm_capability(),
        }

    def _llm_capability(self) -> dict[str, Any]:
        return {
            "provider": "siliconflow",
            "model": self.settings.llm.model,
            "configured": bool(self.settings.llm.api_key),
        }

    def analyze_frame(self, frame: np.ndarray, frame_id: int, timestamp: float | None = None, include_llm: bool = False) -> dict[str, Any]:
        started = time.time()
        ts = timestamp if timestamp is not None else time.time()
        behavior = self.behavior.analyze(frame, frame_id=frame_id, timestamp=ts)
        fatigue = self.fatigue.analyze_frame(frame, frame_id=frame_id, timestamp=ts)
        result = self._combine(
            job_id=None,
            source={"kind": "camera", "name": "本地相机"},
            behavior=behavior,
            fatigue=fatigue,
            elapsed_ms=(time.time() - started) * 1000,
        )

        for detection in result["detections"]:
            self.recent_detections.append(detection)
        result["detections"] = list(self.recent_detections)[-12:]

        now = time.time()
        if include_llm or now - self.last_llm_at > 20:
            self._refresh_llm_async(
                {
                    "score": result["stats"]["score"],
                    "stats": result["stats"],
                    "detections": result["detections"],
                    "source": result["source"],
                }
            )
        result["llm_analysis"] = self.last_llm_text
        return result

    def _refresh_llm_async(self, summary: dict[str, Any]) -> None:
        if self.llm_inflight:
            return
        self.llm_inflight = True
        self.last_llm_at = time.time()

        def worker() -> None:
            try:
                self.last_llm_text = generate_llm_analysis(self.settings.llm, summary)
            finally:
                self.llm_inflight = False

        threading.Thread(target=worker, daemon=True).start()

    def analyze_video(self, video_path: Path, original_name: str) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("无法打开上传视频")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        stride = self.settings.frame_stride
        # 自适应步长：让 max_video_frames 个采样覆盖整段视频，而不是只分析开头
        # （否则长视频后半段的行为，如靠后才出现的吸烟，会被 240 帧上限截断而漏掉）
        if frame_count > 0:
            need = -(-frame_count // self.settings.max_video_frames)  # ceil
            stride = max(stride, need)
        sampled_results: list[dict[str, Any]] = []
        frame_id = 0
        processed = 0
        started = time.time()

        try:
            while processed < self.settings.max_video_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_id % stride == 0:
                    timestamp = frame_id / fps
                    behavior = self.behavior.analyze(frame, frame_id=frame_id, timestamp=timestamp)
                    fatigue = self.fatigue.analyze_frame(frame, frame_id=frame_id, timestamp=timestamp)
                    sampled_results.append(
                        self._combine(
                            job_id=job_id,
                            source={"kind": "upload", "name": original_name},
                            behavior=behavior,
                            fatigue=fatigue,
                            elapsed_ms=0.0,
                        )
                    )
                    processed += 1
                frame_id += 1
        finally:
            capture.release()

        if not sampled_results:
            raise ValueError("上传视频未读取到有效帧")

        result = self._aggregate_video(
            job_id=job_id,
            source={"kind": "upload", "name": original_name},
            frame_count=frame_count,
            fps=fps,
            sampled_results=sampled_results,
            elapsed_ms=(time.time() - started) * 1000,
        )
        result["llm_analysis"] = generate_llm_analysis(
            self.settings.llm,
            {
                "score": result["stats"]["score"],
                "stats": result["stats"],
                "detections": result["detections"][:20],
                "source": result["source"],
                "metrics": result["metrics"],
            },
        )
        report_path = write_markdown_report(self.settings.runtime_dir / "reports", result)
        result["report_url"] = f"/api/reports/{report_path.stem}"
        return result

    def write_result_report(self, result: dict[str, Any]) -> dict[str, Any]:
        job_id = result.get("job_id") or f"live-{uuid.uuid4().hex}"
        report_result = dict(result)
        report_result["job_id"] = job_id
        report_path = write_markdown_report(self.settings.runtime_dir / "reports", report_result)
        return {"job_id": job_id, "report_url": f"/api/reports/{report_path.stem}"}

    def _combine(
        self,
        job_id: str | None,
        source: dict[str, Any],
        behavior: dict[str, Any],
        fatigue: dict[str, Any],
        elapsed_ms: float,
    ) -> dict[str, Any]:
        detections: list[dict[str, Any]] = []
        timestamp = float(behavior.get("timestamp") or fatigue.get("timestamp") or 0.0)
        for index, event in enumerate(behavior.get("behaviors", [])):
            detections.append(normalize_detection(event, timestamp, "behavior", index))

        fatigue_label = fatigue.get("label", "Normal")
        if fatigue_label != "Normal":
            detections.append(
                normalize_detection(
                    {
                        "type": fatigue_label,
                        "label_zh": FATIGUE_LABELS.get(str(fatigue_label), str(fatigue_label)),
                        "confidence": fatigue.get("confidence", 0.0),
                        "severity": RISK_TO_SEVERITY.get(str(fatigue.get("risk_level")), "low"),
                    },
                    timestamp,
                    "fatigue",
                    len(detections),
                )
            )

        behavior_risk = float(behavior.get("risk_score", 0.0))
        fatigue_risk = risk_level_to_score(str(fatigue.get("risk_level")))
        stats = compute_driving_stats(
            behavior_risk=behavior_risk,
            fatigue_risk=fatigue_risk,
            driver_present=bool(behavior.get("driver_present", True)),
            camera_ok=bool(behavior.get("camera_ok", True)),
            detections=detections,
        )
        current_behavior = self._current_behavior_summary(behavior)
        current_fatigue = self._current_fatigue_summary(fatigue)
        return {
            "job_id": job_id,
            "source": source,
            "frame_id": behavior.get("frame_id"),
            "timestamp": timestamp,
            "stats": stats,
            "detections": detections,
            "current_behavior": current_behavior,
            "current_fatigue": current_fatigue,
            "llm_analysis": self.last_llm_text,
            "report_url": None,
            "capabilities": {
                "behavior": behavior.get("capability"),
                "fatigue": fatigue.get("capability"),
                "llm": self._llm_capability(),
            },
            "metrics": {
                "latency_ms": round(float(elapsed_ms or behavior.get("latency_ms", 0.0)), 2),
                "behavior_risk": round(behavior_risk, 2),
                "fatigue_risk": round(fatigue_risk, 2),
            },
        }

    def _current_behavior_summary(self, behavior: dict[str, Any]) -> dict[str, Any]:
        events = behavior.get("behaviors") or []
        if events:
            severity_rank = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
            top_event = max(events, key=lambda item: severity_rank.get(str(item.get("severity", "low")), 1))
            return {
                "algorithm_label": "行为识别",
                "label": top_event.get("label_zh") or BEHAVIOR_LABELS.get(str(top_event.get("type")), str(top_event.get("type", "风险行为"))),
                "confidence": round(float(top_event.get("confidence", 0.0)), 4),
                "severity": str(top_event.get("severity", "low")),
                "bbox": top_event.get("bbox"),
                "recommendation": behavior.get("recommendation") or "请关注当前驾驶行为",
            }
        return {
            "algorithm_label": "行为识别",
            "label": "未检测到行为风险",
            "confidence": 1.0,
            "severity": "none",
            "bbox": None,
            "recommendation": behavior.get("recommendation") or "正常驾驶",
        }

    def _current_fatigue_summary(self, fatigue: dict[str, Any]) -> dict[str, Any]:
        label = str(fatigue.get("label", "Normal"))
        return {
            "algorithm_label": "疲劳检测",
            "label": FATIGUE_LABELS.get(label, label),
            "confidence": round(float(fatigue.get("confidence", 0.0)), 4),
            "risk_level": str(fatigue.get("risk_level", "low")),
            "indicators": fatigue.get("indicators", {}),
            "signals": fatigue.get("signals", {}),
        }

    def _aggregate_video(
        self,
        job_id: str,
        source: dict[str, Any],
        frame_count: int,
        fps: float,
        sampled_results: list[dict[str, Any]],
        elapsed_ms: float,
    ) -> dict[str, Any]:
        detections: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for frame_result in sampled_results:
            for detection in frame_result["detections"]:
                key = (detection["timestamp"], detection["type"])
                if key not in seen:
                    detections.append(detection)
                    seen.add(key)

        behavior_risk = max((item["metrics"]["behavior_risk"] for item in sampled_results), default=0.0)
        fatigue_risk = max((item["metrics"]["fatigue_risk"] for item in sampled_results), default=0.0)
        driver_present = sampled_results[-1]["stats"]["focus"] > 0
        stats = compute_driving_stats(
            behavior_risk=behavior_risk,
            fatigue_risk=fatigue_risk,
            driver_present=driver_present,
            camera_ok=True,
            detections=detections,
        )
        result = dict(sampled_results[-1])
        result.update(
            {
                "job_id": job_id,
                "source": source,
                "stats": stats,
                "detections": detections[:200],
                "metrics": {
                    "latency_ms": round(elapsed_ms, 2),
                    "frames_total": frame_count,
                    "frames_processed": len(sampled_results),
                    "fps": round(fps, 2),
                    "behavior_risk": round(behavior_risk, 2),
                    "fatigue_risk": round(fatigue_risk, 2),
                },
            }
        )
        return result


def result_to_json(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)
