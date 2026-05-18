from __future__ import annotations

from types import SimpleNamespace

from backend.app.analysis import AnalysisService


def test_service_capabilities_exposes_algorithm_modes_and_errors():
    service = AnalysisService.__new__(AnalysisService)
    service.behavior = SimpleNamespace(
        capability=lambda: {"name": "behavior_detector", "mode": "fallback", "error": "No module named 'ultralytics'"}
    )
    service.fatigue = SimpleNamespace(
        capability=lambda: {
            "name": "fatigue_b",
            "mode": "partial",
            "error": "No module named 'mediapipe'",
            "predictor": "model",
            "feature_extractor": "unavailable",
        }
    )
    service.settings = SimpleNamespace(llm=SimpleNamespace(model="Qwen/Qwen3-8B", api_key="secret"))

    capabilities = service.capabilities()

    assert capabilities["behavior"]["mode"] == "fallback"
    assert "ultralytics" in capabilities["behavior"]["error"]
    assert capabilities["fatigue"]["mode"] == "partial"
    assert capabilities["fatigue"]["feature_extractor"] == "unavailable"
    assert capabilities["llm"]["configured"] is True
