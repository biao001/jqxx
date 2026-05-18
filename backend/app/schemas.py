from __future__ import annotations

BEHAVIOR_LABELS = {
    "no_driver": "驾驶位无人",
    "phone_use": "驾驶中使用手机",
    "calling": "驾驶中打电话",
    "no_seatbelt": "未系安全带",
    "hands_off_wheel": "双手离开方向盘",
    "smoking": "驾驶中吸烟",
    "lens_covered": "摄像头被遮挡",
    "abnormal_posture": "驾驶姿态异常",
}

FATIGUE_LABELS = {
    "Normal": "正常",
    "Yawning": "打哈欠",
    "Looking Around": "视线偏离",
    "Fatigued Driving": "疲劳驾驶",
}

RISK_TO_SEVERITY = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "critical": "critical",
    "safe": "none",
    "attention": "low",
    "warning": "medium",
    "danger": "high",
}
