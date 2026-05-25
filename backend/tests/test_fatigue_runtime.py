from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from backend.app.analysis import FatigueRuntime


def test_fatigue_runtime_applies_env_thresholds_after_predictor(monkeypatch):
    def fake_load_feature_extractor(self):
        self.feature_error = None

    def fake_load_predictor(self):
        self.thresholds.update({"yawn": 0.55, "look_away": 0.55, "fatigue": 0.60})

    monkeypatch.setattr(FatigueRuntime, "_load_feature_extractor", fake_load_feature_extractor)
    monkeypatch.setattr(FatigueRuntime, "_load_predictor", fake_load_predictor)

    runtime = FatigueRuntime(
        SimpleNamespace(
            repo_root=None,
            fatigue_yawn_threshold=0.7,
            fatigue_look_away_threshold=0.8,
            fatigue_fatigue_threshold=0.9,
        )
    )

    assert runtime.thresholds["yawn"] == 0.7
    assert runtime.thresholds["look_away"] == 0.8
    assert runtime.thresholds["fatigue"] == 0.9


def test_predict_from_window_flags_yawning_when_mouth_ratio_is_high():
    runtime = FatigueRuntime.__new__(FatigueRuntime)
    runtime.window_size = 16
    runtime.predictor = None
    runtime.predictor_error = None
    runtime.feature_error = None
    runtime.thresholds = {"yawn": 0.55, "look_away": 0.55, "fatigue": 0.60}
    runtime.feature_tools = None
    runtime.landmarker = None
    runtime.feature_window = []

    high_mar_window = np.array([[0.24, 0.76, 0.0, 0.0, 0.0, 0.1]] * 16, dtype=np.float32)
    runtime.feature_window.extend(high_mar_window)

    result = runtime._predict_from_window(frame_id=7, timestamp=1.4)

    assert result["label"] == "Yawning"
    assert result["risk_level"] == "medium"
    assert result["indicators"]["yawn_score"] >= 0.55


def test_predict_from_window_keeps_no_face_features_neutral():
    runtime = FatigueRuntime.__new__(FatigueRuntime)
    runtime.window_size = 16
    runtime.predictor = None
    runtime.predictor_error = None
    runtime.feature_error = None
    runtime.thresholds = {"yawn": 0.55, "look_away": 0.55, "fatigue": 0.60}
    runtime.feature_tools = None
    runtime.landmarker = None
    runtime.feature_window = []
    runtime.feature_window.extend(np.zeros((16, 6), dtype=np.float32))

    result = runtime._predict_from_window(frame_id=1, timestamp=0.2)

    assert result["label"] == "Normal"
    assert result["risk_level"] == "low"
    assert result["indicators"]["fatigue_score"] == 0.0


def test_predict_from_window_uses_higher_look_away_threshold():
    runtime = FatigueRuntime.__new__(FatigueRuntime)
    runtime.window_size = 16
    runtime.predictor = None
    runtime.predictor_error = None
    runtime.feature_error = None
    runtime.thresholds = {"yawn": 0.55, "look_away": 0.80, "fatigue": 0.60}
    runtime.feature_tools = None
    runtime.landmarker = None
    runtime.feature_window = []

    near_threshold_window = np.array([[0.24, 0.2, 26.22, 0.0, 0.0, 0.1]] * 16, dtype=np.float32)
    runtime.feature_window.extend(near_threshold_window)

    result = runtime._predict_from_window(frame_id=2, timestamp=0.4)

    assert result["label"] == "Normal"
    assert result["risk_level"] == "low"
    assert result["indicators"]["look_away_score"] < 0.8


def test_predict_from_window_fuses_model_with_observable_yawn_signal():
    class NormalPredictor:
        thresholds = {"yawn": 0.55, "look_away": 0.55, "fatigue": 0.60}

        def __init__(self):
            self.called = False

        def predict_window(self, _window, frame_id, timestamp):
            self.called = True
            return {
                "module": "fatigue",
                "model_name": "fatigue_b",
                "frame_id": frame_id,
                "timestamp": timestamp,
                "label": "Normal",
                "confidence": 0.7,
                "indicators": {"yawn_score": 0.1, "look_away_score": 0.1, "fatigue_score": 0.1},
                "risk_level": "low",
                "public_scores": {"Normal": 0.9, "Yawning": 0.1, "Looking Around": 0.1, "Fatigued Driving": 0.1},
            }

    predictor = NormalPredictor()
    runtime = FatigueRuntime.__new__(FatigueRuntime)
    runtime.window_size = 16
    runtime.predictor = predictor
    runtime.predictor_error = None
    runtime.feature_error = None
    runtime.thresholds = predictor.thresholds
    runtime.feature_tools = None
    runtime.landmarker = None
    runtime.feature_window = []
    runtime.feature_window.extend(np.array([[0.24, 0.76, 0.0, 0.0, 0.0, 0.1]] * 16, dtype=np.float32))

    result = runtime._predict_from_window(frame_id=8, timestamp=1.6)

    assert predictor.called is True
    assert result["model_name"] == "fatigue_b"
    assert result["label"] == "Yawning"
    assert result["risk_level"] == "medium"
    assert result["indicators"]["yawn_score"] >= 0.55


def test_predict_from_window_clears_yawning_when_latest_mouth_ratio_is_normal():
    class StaleYawningPredictor:
        thresholds = {"yawn": 0.55, "look_away": 0.55, "fatigue": 0.60}

        def predict_window(self, _window, frame_id, timestamp):
            return {
                "module": "fatigue",
                "model_name": "fatigue_b",
                "frame_id": frame_id,
                "timestamp": timestamp,
                "label": "Yawning",
                "confidence": 0.8,
                "indicators": {"yawn_score": 0.9, "look_away_score": 0.1, "fatigue_score": 0.1},
                "risk_level": "medium",
                "public_scores": {"Normal": 0.1, "Yawning": 0.9, "Looking Around": 0.1, "Fatigued Driving": 0.1},
            }

    predictor = StaleYawningPredictor()
    runtime = FatigueRuntime.__new__(FatigueRuntime)
    runtime.window_size = 16
    runtime.predictor = predictor
    runtime.predictor_error = None
    runtime.feature_error = None
    runtime.thresholds = predictor.thresholds
    runtime.feature_tools = None
    runtime.landmarker = None
    runtime.feature_window = []
    stale_yawn_frames = np.array([[0.24, 0.76, 0.0, 0.0, 0.0, 0.1]] * 15, dtype=np.float32)
    latest_normal_frame = np.array([[0.24, 0.20, 0.0, 0.0, 0.0, 0.1]], dtype=np.float32)
    runtime.feature_window.extend(np.concatenate([stale_yawn_frames, latest_normal_frame], axis=0))

    result = runtime._predict_from_window(frame_id=9, timestamp=1.8)

    assert result["label"] == "Normal"
    assert result["risk_level"] == "low"
    assert result["indicators"]["yawn_score"] < runtime.thresholds["yawn"]


def test_predict_from_window_clears_look_away_when_latest_head_yaw_is_normal():
    class StaleLookAwayPredictor:
        thresholds = {"yawn": 0.55, "look_away": 0.55, "fatigue": 0.60}

        def predict_window(self, _window, frame_id, timestamp):
            return {
                "module": "fatigue",
                "model_name": "fatigue_b",
                "frame_id": frame_id,
                "timestamp": timestamp,
                "label": "Looking Around",
                "confidence": 0.8,
                "indicators": {"yawn_score": 0.1, "look_away_score": 0.9, "fatigue_score": 0.1},
                "risk_level": "medium",
                "public_scores": {"Normal": 0.1, "Yawning": 0.1, "Looking Around": 0.9, "Fatigued Driving": 0.1},
            }

    predictor = StaleLookAwayPredictor()
    runtime = FatigueRuntime.__new__(FatigueRuntime)
    runtime.window_size = 16
    runtime.predictor = predictor
    runtime.predictor_error = None
    runtime.feature_error = None
    runtime.thresholds = predictor.thresholds
    runtime.feature_tools = None
    runtime.landmarker = None
    runtime.feature_window = []
    stale_look_frames = np.array([[0.24, 0.20, 30.0, 0.0, 0.0, 0.1]] * 15, dtype=np.float32)
    latest_forward_frame = np.array([[0.24, 0.20, 0.0, 0.0, 0.0, 0.1]], dtype=np.float32)
    runtime.feature_window.extend(np.concatenate([stale_look_frames, latest_forward_frame], axis=0))

    result = runtime._predict_from_window(frame_id=10, timestamp=2.0)

    assert result["label"] == "Normal"
    assert result["risk_level"] == "low"
    assert result["indicators"]["look_away_score"] < runtime.thresholds["look_away"]


def test_extract_features_uses_original_mediapipe_feature_chain():
    runtime = FatigueRuntime.__new__(FatigueRuntime)
    runtime.settings = SimpleNamespace(repo_root=None)
    runtime.landmarker = SimpleNamespace(
        detect_for_video=lambda _image, _timestamp_ms: SimpleNamespace(
            face_landmarks=[
                [
                    SimpleNamespace(x=0.25, y=0.25),
                    SimpleNamespace(x=0.75, y=0.75),
                ]
            ]
        )
    )
    runtime.frontal_cascade = object()
    runtime.profile_cascade = object()
    runtime._last_mp_timestamp_ms = -1
    calls: list[str] = []

    def eye_aspect_ratio(_points, eye_indices):
        calls.append(f"ear:{eye_indices[0]}")
        return 0.2 if eye_indices[0] == 33 else 0.4

    def mouth_aspect_ratio(_points):
        calls.append("mar")
        return 0.82

    def estimate_head_pose(_points, width, height):
        calls.append(f"pose:{width}x{height}")
        return 9.0, -4.0, 2.0

    def estimate_drowsy_prob(ear, mar, head_pitch):
        calls.append(f"drowsy:{ear:.2f}:{mar:.2f}:{head_pitch:.2f}")
        return 0.67

    runtime.feature_tools = SimpleNamespace(
        mp=SimpleNamespace(
            Image=lambda image_format, data: {"format": image_format, "data": data},
            ImageFormat=SimpleNamespace(SRGB="SRGB"),
        ),
        left_eye=[33],
        right_eye=[362],
        eye_aspect_ratio=eye_aspect_ratio,
        mouth_aspect_ratio=mouth_aspect_ratio,
        estimate_head_pose=estimate_head_pose,
        estimate_drowsy_prob=estimate_drowsy_prob,
        detect_face_bbox_fallback=lambda *_args: (None, None),
        approximate_mouth_aspect_ratio=lambda *_args: 0.0,
    )

    features = runtime.extract_features(np.zeros((80, 100, 3), dtype=np.uint8), timestamp=0.2)

    assert calls == ["ear:33", "ear:362", "mar", "pose:100x80", "drowsy:0.30:0.82:-4.00"]
    assert features.tolist() == [0.3, 0.82, 9.0, -4.0, 2.0, 0.67]
