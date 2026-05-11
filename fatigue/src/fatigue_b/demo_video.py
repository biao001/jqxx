"""Render an annotated fatigue demo video from one input video."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.fatigue_b.config import DEFAULT_THRESHOLDS, FEATURE_COLUMNS, RISK_MAP, scores_to_public_label
from src.fatigue_b.config import summarize_public_result

try:
    from src.fatigue_b.infer import FatigueBPredictor
except Exception:  # pragma: no cover - predictor depends on optional torch runtime
    FatigueBPredictor = None


def heuristic_scores(window: np.ndarray) -> tuple[float, float, float]:
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

    yawn_score = float(np.max(yawn_signal).clip(0.0, 1.0))
    look_away_score = float(np.max(look_signal).clip(0.0, 1.0))
    fatigue_score = float(np.clip(fatigue_signal, 0.0, 1.0))
    return yawn_score, look_away_score, fatigue_score


def public_result(yawn_score: float, look_away_score: float, fatigue_score: float, thresholds: dict[str, float]) -> dict:
    summary = summarize_public_result(
        yawn_score=yawn_score,
        look_away_score=look_away_score,
        fatigue_score=fatigue_score,
        thresholds=thresholds,
    )
    return {
        "label": summary["label"],
        "confidence": summary["confidence"],
        "indicators": summary["indicators"],
        "risk_level": summary["risk_level"],
        "public_scores": summary["public_scores"],
    }


def color_for_result(label: str, risk_level: str) -> tuple[int, int, int]:
    if risk_level == "high":
        return (0, 0, 255)
    if label == "Yawning":
        return (0, 165, 255)
    if label == "Looking Around":
        return (0, 255, 255)
    return (0, 200, 0)


def load_feature_rows(features_csv: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(features_csv)
    required = ["frame_id", "timestamp", *FEATURE_COLUMNS]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing feature columns in {features_csv}: {missing}")
    if "face_valid" in dataframe.columns:
        valid_mask = dataframe["face_valid"] == 1
        if valid_mask.any():
            dataframe.loc[~valid_mask, FEATURE_COLUMNS] = np.nan
            dataframe[FEATURE_COLUMNS] = dataframe[FEATURE_COLUMNS].ffill().fillna(0.0)
    return dataframe


def render_demo(
    video_path: Path,
    features_csv: Path,
    output_video: Path,
    output_json: Path,
    window_size: int,
    checkpoint_path: Path | None,
    device: str,
) -> None:
    features = load_feature_rows(features_csv)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    predictor = None
    if checkpoint_path:
        if FatigueBPredictor is None:
            raise RuntimeError("Checkpoint mode requires torch and src.fatigue_b.infer imports to work.")
        predictor = FatigueBPredictor(checkpoint_path, device=device)
        print(f"Using device: {predictor.resolved_device}")

    thresholds = DEFAULT_THRESHOLDS.copy()
    if predictor is not None:
        thresholds.update(predictor.thresholds)

    feature_queue: deque[np.ndarray] = deque(maxlen=window_size)
    predictions = []
    frame_idx = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        row = features.iloc[min(frame_idx, len(features) - 1)]
        feature_row = row[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        feature_queue.append(feature_row)

        if len(feature_queue) < window_size:
            padded = list(feature_queue) + [feature_queue[-1]] * (window_size - len(feature_queue))
            window = np.stack(padded, axis=0)
        else:
            window = np.stack(feature_queue, axis=0)

        if predictor is not None:
            result = predictor.predict_window(window, frame_id=int(row["frame_id"]), timestamp=float(row["timestamp"]))
        else:
            yawn_score, look_away_score, fatigue_score = heuristic_scores(window)
            result = public_result(yawn_score, look_away_score, fatigue_score, thresholds)
            result["module"] = "fatigue"
            result["model_name"] = "fatigue_b_preview"
            result["frame_id"] = int(row["frame_id"])
            result["timestamp"] = round(float(row["timestamp"]), 4)

        predictions.append(result)
        draw_overlay(frame, result)
        writer.write(frame)
        frame_idx += 1

    capture.release()
    writer.release()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2, ensure_ascii=False)


def draw_overlay(frame: np.ndarray, result: dict) -> None:
    label = result["label"]
    color = color_for_result(label, result["risk_level"])
    indicators = result["indicators"]

    cv2.rectangle(frame, (20, 20), (560, 190), (20, 20, 20), thickness=-1)
    cv2.rectangle(frame, (20, 20), (560, 190), color, thickness=2)

    text_lines = [
        f"Label: {label}",
        f"Yawn Score: {indicators['yawn_score']:.2f}",
        f"Look Away Score: {indicators['look_away_score']:.2f}",
        f"Fatigue Score: {indicators['fatigue_score']:.2f}",
        f"Risk: {result['risk_level']}",
    ]
    y = 55
    for line in text_lines:
        cv2.putText(frame, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        y += 24


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an annotated fatigue demo video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--checkpoint", help="Optional trained checkpoint. If omitted, use heuristic preview.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    render_demo(
        video_path=Path(args.video),
        features_csv=Path(args.features_csv),
        output_video=Path(args.output_video),
        output_json=Path(args.output_json),
        window_size=args.window_size,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        device=args.device,
    )
    print(f"Saved annotated video to {args.output_video}")


if __name__ == "__main__":
    main()
