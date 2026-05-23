from __future__ import annotations

from typing import Any


SEVERITY_PENALTY = {
    "none": 0,
    "low": 4,
    "medium": 9,
    "high": 16,
    "critical": 30,
}


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, float(value)))


def format_timestamp(seconds: float | int | None) -> str:
    value = max(0, int(seconds or 0))
    minutes = value // 60
    remainder = value % 60
    return f"{minutes:02d}:{remainder:02d}"


def normalize_detection(
    event: dict[str, Any],
    timestamp_seconds: float | int | None,
    prefix: str,
    index: int,
) -> dict[str, Any]:
    label = event.get("label_zh") or event.get("label") or event.get("type") or "未知事件"
    confidence = round(float(event.get("confidence", 0.0)), 4)
    normalized = {
        "id": f"{prefix}-{index}",
        "type": str(label),
        "timestamp": format_timestamp(timestamp_seconds),
        "confidence": confidence,
        "severity": str(event.get("severity") or event.get("risk_level") or "low"),
        "source": prefix,
    }
    if event.get("bbox") is not None:
        normalized["bbox"] = [int(value) for value in event["bbox"]]
    return normalized


def risk_level_to_score(risk_level: str | None) -> float:
    if risk_level == "high":
        return 80.0
    if risk_level == "medium":
        return 55.0
    if risk_level == "low":
        return 12.0
    return 0.0


def compute_driving_stats(
    behavior_risk: float,
    fatigue_risk: float,
    driver_present: bool,
    camera_ok: bool,
    detections: list[dict[str, Any]],
) -> dict[str, int | str]:
    severity_penalty = sum(SEVERITY_PENALTY.get(str(item.get("severity", "low")), 4) for item in detections[:8])
    base_penalty = behavior_risk * 0.42 + fatigue_risk * 0.36 + severity_penalty
    if not driver_present:
        base_penalty += 25
    if not camera_ok:
        base_penalty += 20

    score = int(round(clamp(100 - base_penalty)))
    if score >= 80:
        status = "正常"
    elif score >= 60:
        status = "注意"
    elif score >= 40:
        status = "警告"
    else:
        status = "危险"

    focus = int(round(clamp(100 - behavior_risk * 0.65 - (0 if driver_present else 25))))
    reaction = int(round(clamp(100 - fatigue_risk * 0.55 - behavior_risk * 0.15)))
    compliance = int(round(clamp(100 - behavior_risk * 0.55 - severity_penalty * 0.7)))
    fatigue = int(round(clamp(100 - fatigue_risk)))
    stability = int(round(clamp(100 - behavior_risk * 0.25 - fatigue_risk * 0.25 - (0 if camera_ok else 25))))

    # 驾驶行为评分(只看行为风险与违规严重度)；疲劳评分(只看疲劳风险)
    behavior_score = int(round(clamp(
        100 - behavior_risk * 0.7 - severity_penalty - (0 if driver_present else 25) - (0 if camera_ok else 20)
    )))
    fatigue_score = int(round(clamp(100 - fatigue_risk)))

    return {
        "score": score,  # 综合评分(行为+疲劳加权)
        "status": status,
        "behavior_score": behavior_score,
        "fatigue_score": fatigue_score,
        "focus": focus,
        "reaction": reaction,
        "compliance": compliance,
        "fatigue": fatigue,
        "stability": stability,
    }
