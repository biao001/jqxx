from __future__ import annotations

BEHAVIOR_LABELS = {
    "no_driver":       "驾驶位无人",
    "phone_use":       "驾驶中使用手机",
    "no_seatbelt":     "未系安全带",
    "hands_off_wheel": "双手离开方向盘",
    "smoking":         "驾驶中吸烟",
    "drinking":        "驾驶中饮水",
    "eating":          "驾驶中进食",
    "lens_covered":    "摄像头被遮挡",
    "hand_on_wheel":   "双手在方向盘",
    "seatbelt":        "已系安全带",
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
