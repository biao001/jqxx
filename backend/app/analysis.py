from __future__ import annotations

import base64
import json
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
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
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _load_detector(self) -> Any | None:
        if self.detector is not None or self.load_error is not None:
            return self.detector
        behavior_dir = self.settings.repo_root / "behavior_algo"
        sys.path.insert(0, str(behavior_dir))
        try:
            from behavior_detector import BehaviorDetector

            self.detector = BehaviorDetector(
                yolo_weights="yolov8n.pt",
                pose_weights="yolov8n-pose.pt",
                seatbelt_weights=str(behavior_dir / "models" / "seatbelt.pt"),
                smoking_weights=str(behavior_dir / "models" / "smoking.pt"),
                unified_weights=str(behavior_dir / "models" / "unified.pt"),
                device="cpu",
                imgsz=384,
            )
        except Exception as exc:
            self.load_error = str(exc)
            self.detector = None
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
                result["capability"] = {"name": "behavior_detector", "mode": "model", "error": None}
                return result
            except Exception as exc:
                self.load_error = str(exc)

        result = self._fallback(frame, frame_id, timestamp)
        result["capability"] = {"name": "behavior_detector", "mode": "fallback", "error": self.load_error}
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
        self.predictor: Any | None = None
        self.load_error: str | None = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._load_predictor()

    def _load_predictor(self) -> None:
        checkpoint = self.settings.repo_root / "fatigue" / "outputs" / "fatigue_b" / "checkpoints" / "best_model.pt"
        if not checkpoint.exists():
            self.load_error = "fatigue checkpoint not found"
            return
        try:
            fatigue_root = self.settings.repo_root / "fatigue"
            sys.path.insert(0, str(fatigue_root))
            from src.fatigue_b.infer import FatigueBPredictor

            self.predictor = FatigueBPredictor(checkpoint, device="auto")
        except Exception as exc:
            self.load_error = str(exc)
            self.predictor = None
        finally:
            try:
                sys.path.remove(str(self.settings.repo_root / "fatigue"))
            except ValueError:
                pass

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        if len(faces) == 0:
            return np.array([0.0, 0.0, 25.0, 0.0, 0.0, 0.55], dtype=np.float32)

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
        features = self.extract_features(frame)
        self.feature_window.append(features)
        return self._predict_from_window(frame_id, timestamp)

    def _predict_from_window(self, frame_id: int, timestamp: float) -> dict[str, Any]:
        if not self.feature_window:
            window = np.zeros((self.window_size, 6), dtype=np.float32)
        else:
            rows = list(self.feature_window)
            while len(rows) < self.window_size:
                rows.append(rows[-1])
            window = np.stack(rows[-self.window_size :], axis=0)

        if self.predictor is not None:
            try:
                result = self.predictor.predict_window(window, frame_id=frame_id, timestamp=timestamp)
                result["capability"] = {"name": "fatigue_b", "mode": "model", "error": None}
                return result
            except Exception as exc:
                self.load_error = str(exc)
                self.predictor = None

        yawn_score = float(np.clip(np.max((window[:, 1] - 0.48) / 0.25), 0.0, 1.0))
        look_score = float(np.clip(np.max((np.abs(window[:, 2]) - 12.0) / 18.0), 0.0, 1.0))
        fatigue_score = float(np.clip(np.percentile(window[:, 5], 90), 0.0, 1.0))
        if yawn_score >= 0.55:
            label = "Yawning"
        elif look_score >= 0.55:
            label = "Looking Around"
        elif fatigue_score >= 0.60:
            label = "Fatigued Driving"
        else:
            label = "Normal"
        risk_level = "high" if fatigue_score >= 0.60 else "medium" if (yawn_score >= 0.55 or look_score >= 0.55) else "low"
        return {
            "module": "fatigue",
            "model_name": "fatigue_b_fallback",
            "frame_id": frame_id,
            "timestamp": round(timestamp, 4),
            "label": label,
            "confidence": round(max(yawn_score, look_score, fatigue_score, 1.0 - max(yawn_score, look_score, fatigue_score)), 4),
            "indicators": {
                "yawn_score": round(yawn_score, 4),
                "look_away_score": round(look_score, 4),
                "fatigue_score": round(fatigue_score, 4),
            },
            "risk_level": risk_level,
            "public_scores": {
                "Normal": round(1.0 - max(yawn_score, look_score, fatigue_score), 4),
                "Yawning": round(yawn_score, 4),
                "Looking Around": round(look_score, 4),
                "Fatigued Driving": round(fatigue_score, 4),
            },
            "capability": {"name": "fatigue_b", "mode": "fallback", "error": self.load_error},
        }


class AnalysisService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.behavior = BehaviorRuntime(settings)
        self.fatigue = FatigueRuntime(settings)
        self.recent_detections: deque[dict[str, Any]] = deque(maxlen=50)
        self.last_llm_at = time.time()
        self.last_llm_text = "等待足够的真实检测数据生成大模型分析。"
        self.llm_inflight = False

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
                "llm": {
                    "provider": "siliconflow",
                    "model": self.settings.llm.model,
                    "configured": bool(self.settings.llm.api_key),
                },
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
                "label": top_event.get("label_zh") or BEHAVIOR_LABELS.get(str(top_event.get("type")), str(top_event.get("type", "风险行为"))),
                "confidence": round(float(top_event.get("confidence", 0.0)), 4),
                "severity": str(top_event.get("severity", "low")),
                "recommendation": behavior.get("recommendation") or "请关注当前驾驶行为",
            }
        return {
            "label": "未检测到行为风险",
            "confidence": 1.0,
            "severity": "none",
            "recommendation": behavior.get("recommendation") or "正常驾驶",
        }

    def _current_fatigue_summary(self, fatigue: dict[str, Any]) -> dict[str, Any]:
        label = str(fatigue.get("label", "Normal"))
        return {
            "label": FATIGUE_LABELS.get(label, label),
            "confidence": round(float(fatigue.get("confidence", 0.0)), 4),
            "risk_level": str(fatigue.get("risk_level", "low")),
            "indicators": fatigue.get("indicators", {}),
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
