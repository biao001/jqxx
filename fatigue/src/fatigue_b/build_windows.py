"""Build fixed-length training windows from feature CSV files and label CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.fatigue_b.config import FEATURE_COLUMNS


def load_feature_table(features_dir: Path, video_id: str) -> pd.DataFrame:
    feature_path = features_dir / f"{video_id}.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    return pd.read_csv(feature_path)


def resize_window(values: np.ndarray, window_size: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros((window_size, values.shape[1] if values.ndim == 2 else len(FEATURE_COLUMNS)), dtype=np.float32)
    if len(values) == window_size:
        return values.astype(np.float32)
    if len(values) > window_size:
        indices = np.linspace(0, len(values) - 1, window_size).round().astype(np.int32)
        return values[indices].astype(np.float32)
    pad = np.repeat(values[-1:], window_size - len(values), axis=0)
    return np.concatenate([values, pad], axis=0).astype(np.float32)


def build_split(labels_csv: Path, features_dir: Path, output_npz: Path, output_metadata: Path, window_size: int) -> None:
    labels = pd.read_csv(labels_csv)
    features = []
    targets = []
    metadata_rows = []

    for row in labels.to_dict(orient="records"):
        table = load_feature_table(features_dir, row["video_id"])
        window = table[(table["frame_id"] >= row["start_frame"]) & (table["frame_id"] <= row["end_frame"])].copy()
        if window.empty:
            continue
        feature_values = window[FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
        feature_values = resize_window(feature_values, window_size)

        target = np.array(
            [
                int(row["is_yawning"]),
                int(row["is_look_away"]),
                int(row["is_fatigued"]),
            ],
            dtype=np.float32,
        )
        features.append(feature_values)
        targets.append(target)
        metadata_rows.append(row)

    if not features:
        raise RuntimeError(f"No windows built for {labels_csv}")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        features=np.stack(features),
        targets=np.stack(targets),
    )
    pd.DataFrame(metadata_rows).to_csv(output_metadata, index=False)
    print(f"Saved {len(features)} windows to {output_npz}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed windows for fatigue model B.")
    parser.add_argument("--labels-csv", required=True, help="train.csv, val.csv, or test.csv")
    parser.add_argument("--features-dir", required=True, help="Directory with per-video feature CSV files.")
    parser.add_argument("--output-npz", required=True, help="Output .npz path.")
    parser.add_argument("--output-metadata", required=True, help="Output metadata CSV path.")
    parser.add_argument("--window-size", type=int, default=16)
    args = parser.parse_args()

    build_split(
        labels_csv=Path(args.labels_csv),
        features_dir=Path(args.features_dir),
        output_npz=Path(args.output_npz),
        output_metadata=Path(args.output_metadata),
        window_size=args.window_size,
    )


if __name__ == "__main__":
    main()
