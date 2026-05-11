"""Inference API for fatigue model B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.fatigue_b.config import DEFAULT_THRESHOLDS, FEATURE_COLUMNS, summarize_public_result
from src.fatigue_b.metrics import sigmoid
from src.fatigue_b.model import build_model_from_checkpoint
from src.fatigue_b.runtime import resolve_torch_device


class FatigueBPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "auto"):
        self.resolved_device = resolve_torch_device(device)
        self.device = torch.device(self.resolved_device)
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = build_model_from_checkpoint(self.checkpoint).to(self.device)
        self.model.eval()
        self.thresholds = self.checkpoint.get("thresholds", DEFAULT_THRESHOLDS)

    def predict_scores(self, feature_window: np.ndarray) -> np.ndarray:
        if feature_window.ndim != 2:
            raise ValueError("feature_window must have shape [window_size, feature_dim]")
        tensor = torch.from_numpy(feature_window.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            logit_tensor = torch.stack(
                [logits["yawn"], logits["look_away"], logits["fatigue"]],
                dim=1,
            )
            return sigmoid(logit_tensor.cpu().numpy())[0]

    def predict_window(self, feature_window: np.ndarray, frame_id: int | None = None, timestamp: float | None = None) -> dict:
        yawn_score, look_away_score, fatigue_score = self.predict_scores(feature_window)
        summary = summarize_public_result(
            yawn_score=float(yawn_score),
            look_away_score=float(look_away_score),
            fatigue_score=float(fatigue_score),
            thresholds=self.thresholds,
        )

        return {
            "module": "fatigue",
            "model_name": "fatigue_b",
            "frame_id": int(frame_id) if frame_id is not None else None,
            "timestamp": round(float(timestamp), 4) if timestamp is not None else None,
            "label": summary["label"],
            "confidence": summary["confidence"],
            "indicators": summary["indicators"],
            "risk_level": summary["risk_level"],
            "public_scores": summary["public_scores"],
        }


def sliding_windows(feature_table: pd.DataFrame, window_size: int, stride: int):
    rows = feature_table[FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
    frames = feature_table["frame_id"].to_numpy()
    timestamps = feature_table["timestamp"].to_numpy()
    total = len(rows)
    for start in range(0, max(1, total - window_size + 1), stride):
        end = start + window_size
        if end > total:
            end = total
            start = max(0, end - window_size)
        window = rows[start:end]
        if len(window) < window_size:
            pad = np.repeat(window[-1:], window_size - len(window), axis=0)
            window = np.concatenate([window, pad], axis=0)
        yield window, int(frames[min(end - 1, len(frames) - 1)]), float(timestamps[min(end - 1, len(timestamps) - 1)])
        if end == total:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for fatigue model B.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    predictor = FatigueBPredictor(args.checkpoint, args.device)
    print(f"Using device: {predictor.resolved_device}")
    feature_table = pd.read_csv(args.features_csv)
    predictions = [
        predictor.predict_window(window, frame_id=frame_id, timestamp=timestamp)
        for window, frame_id, timestamp in sliding_windows(feature_table, args.window_size, args.stride)
    ]

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2, ensure_ascii=False)
    print(f"Saved inference results to {output_path}")


if __name__ == "__main__":
    main()
