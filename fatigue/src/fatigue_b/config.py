"""Shared constants for member 2 and member 6 deliverables."""

PUBLIC_LABELS = [
    "Normal",
    "Yawning",
    "Looking Around",
    "Fatigued Driving",
]

INTERNAL_TARGETS = [
    "is_yawning",
    "is_look_away",
    "is_fatigued",
]

FEATURE_COLUMNS = [
    "ear",
    "mar",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "drowsy_prob",
]

DEFAULT_THRESHOLDS = {
    "yawn": 0.55,
    "look_away": 0.55,
    "fatigue": 0.60,
}

RISK_MAP = {
    "Normal": "low",
    "Yawning": "medium",
    "Looking Around": "medium",
    "Fatigued Driving": "high",
}


def clamp01(value):
    return float(max(0.0, min(1.0, value)))


def scores_to_public_label(yawn_score, look_away_score, fatigue_score, thresholds=None):
    """Map multi-head scores to the public four-class output."""
    return summarize_public_result(
        yawn_score=yawn_score,
        look_away_score=look_away_score,
        fatigue_score=fatigue_score,
        thresholds=thresholds,
    )["label"]


def summarize_public_result(yawn_score, look_away_score, fatigue_score, thresholds=None):
    """Build the public result from internal scores.

    Rules:
    - specific observable events (`Yawning`, `Looking Around`) take priority over the
      aggregate state label `Fatigued Driving`
    - `risk_level` reflects overall danger, not only the displayed label
    - `confidence` is normalized across the four public states to avoid duplicating
      `fatigue_score` when the label is `Fatigued Driving`
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    yawn_score = clamp01(yawn_score)
    look_away_score = clamp01(look_away_score)
    fatigue_score = clamp01(fatigue_score)

    event_candidates = []
    if yawn_score >= thresholds["yawn"]:
        event_candidates.append(("Yawning", yawn_score))
    if look_away_score >= thresholds["look_away"]:
        event_candidates.append(("Looking Around", look_away_score))

    if event_candidates:
        label = max(event_candidates, key=lambda item: item[1])[0]
    elif fatigue_score >= thresholds["fatigue"]:
        label = "Fatigued Driving"
    else:
        label = "Normal"

    if fatigue_score >= thresholds["fatigue"]:
        risk_level = "high"
    elif yawn_score >= thresholds["yawn"] or look_away_score >= thresholds["look_away"]:
        risk_level = "medium"
    else:
        risk_level = "low"

    public_scores = {
        "Normal": clamp01(1.0 - max(yawn_score, look_away_score, fatigue_score)),
        "Yawning": yawn_score,
        "Looking Around": look_away_score,
        "Fatigued Driving": fatigue_score,
    }
    denominator = sum(public_scores.values())
    confidence = public_scores[label] / denominator if denominator > 0 else 0.0

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "risk_level": risk_level,
        "public_scores": {key: round(float(value), 4) for key, value in public_scores.items()},
        "indicators": {
            "yawn_score": round(float(yawn_score), 4),
            "look_away_score": round(float(look_away_score), 4),
            "fatigue_score": round(float(fatigue_score), 4),
        },
    }
