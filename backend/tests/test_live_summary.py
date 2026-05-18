from types import SimpleNamespace

from backend.app.analysis import AnalysisService


def test_combine_exposes_current_behavior_and_fatigue_summaries():
    service = AnalysisService.__new__(AnalysisService)
    service.last_llm_text = "pending"
    service.settings = SimpleNamespace(llm=SimpleNamespace(model="Qwen/Qwen3-8B", api_key="secret"))

    result = service._combine(
        job_id=None,
        source={"kind": "upload-live", "name": "drive.mp4"},
        behavior={
            "frame_id": 42,
            "timestamp": 7.5,
            "behaviors": [
                {
                    "type": "phone_use",
                    "label_zh": "驾驶中使用手机",
                    "confidence": 0.93,
                    "severity": "high",
                }
            ],
            "risk_score": 70.0,
            "driver_present": True,
            "camera_ok": True,
            "capability": {"name": "behavior_detector", "mode": "model", "error": None},
        },
        fatigue={
            "label": "Yawning",
            "confidence": 0.81,
            "risk_level": "medium",
            "indicators": {"yawn_score": 0.81, "look_away_score": 0.2, "fatigue_score": 0.3},
            "capability": {"name": "fatigue_b", "mode": "model", "error": None},
        },
        elapsed_ms=32.0,
    )

    assert result["current_behavior"]["label"] == "驾驶中使用手机"
    assert result["current_behavior"]["algorithm_label"] == "行为识别"
    assert result["current_behavior"]["confidence"] == 0.93
    assert result["current_behavior"]["severity"] == "high"
    assert result["current_fatigue"]["label"] == "打哈欠"
    assert result["current_fatigue"]["algorithm_label"] == "疲劳检测"
    assert result["current_fatigue"]["confidence"] == 0.81
    assert result["current_fatigue"]["risk_level"] == "medium"
