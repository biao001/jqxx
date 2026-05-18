from backend.app.scoring import compute_driving_stats, normalize_detection


def test_compute_driving_stats_penalizes_high_risk_events():
    stats = compute_driving_stats(
        behavior_risk=70.0,
        fatigue_risk=80.0,
        driver_present=True,
        camera_ok=True,
        detections=[{"severity": "high"}, {"severity": "medium"}],
    )

    assert stats["score"] < 55
    assert stats["status"] in {"危险", "警告"}
    assert stats["fatigue"] == 20
    assert stats["compliance"] < 80


def test_compute_driving_stats_penalizes_missing_driver_and_camera():
    stats = compute_driving_stats(
        behavior_risk=10.0,
        fatigue_risk=0.0,
        driver_present=False,
        camera_ok=False,
        detections=[],
    )

    assert stats["score"] <= 55
    assert stats["focus"] < 80
    assert stats["stability"] < 75


def test_normalize_detection_maps_backend_event_to_ui_shape():
    detection = normalize_detection(
        event={
            "type": "phone_use",
            "label_zh": "驾驶中使用手机",
            "confidence": 0.91,
            "severity": "high",
        },
        timestamp_seconds=12.4,
        prefix="behavior",
        index=2,
    )

    assert detection["id"] == "behavior-2"
    assert detection["type"] == "驾驶中使用手机"
    assert detection["timestamp"] == "00:12"
    assert detection["confidence"] == 0.91
    assert detection["severity"] == "high"
