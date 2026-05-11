"""Normalize raw fatigue labels from multiple source datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SOURCE_RULES = {
    "NTHU-DDD": {
        "yawning": {"is_yawning": 1, "is_look_away": 0, "is_fatigued": 0},
        "looking_aside": {"is_yawning": 0, "is_look_away": 1, "is_fatigued": 0},
        "sleepy_eyes": {"is_yawning": 0, "is_look_away": 0, "is_fatigued": 1},
        "nodding": {"is_yawning": 0, "is_look_away": 0, "is_fatigued": 1},
        "normal": {"is_yawning": 0, "is_look_away": 0, "is_fatigued": 0},
    },
    "YawDD": {
        "yawning": {"is_yawning": 1, "is_look_away": 0, "is_fatigued": 0},
        "normal": {"is_yawning": 0, "is_look_away": 0, "is_fatigued": 0},
        "talking_singing": {"is_yawning": 0, "is_look_away": 0, "is_fatigued": 0},
    },
    "UTA-RLDD": {
        "alertness": {"is_yawning": 0, "is_look_away": 0, "is_fatigued": 0},
        "low_vigilance": {"is_yawning": 0, "is_look_away": 0, "is_fatigued": 1},
        "drowsiness": {"is_yawning": 0, "is_look_away": 0, "is_fatigued": 1},
    },
    "self_recorded": {
        "yawning": {"is_yawning": 1, "is_look_away": 0, "is_fatigued": 0},
        "looking_around": {"is_yawning": 0, "is_look_away": 1, "is_fatigued": 0},
        "fatigued_driving": {"is_yawning": 0, "is_look_away": 0, "is_fatigued": 1},
        "normal": {"is_yawning": 0, "is_look_away": 0, "is_fatigued": 0},
    },
}

REQUIRED_COLUMNS = [
    "video_id",
    "start_frame",
    "end_frame",
    "subject_id",
    "source_dataset",
    "source_label",
]


def convert_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []
    for row in manifest.to_dict(orient="records"):
        dataset = row["source_dataset"]
        source_label = row["source_label"]
        rules = SOURCE_RULES.get(dataset, {})
        mapped = rules.get(source_label)
        if mapped is None:
            raise ValueError(f"Unsupported label '{source_label}' for dataset '{dataset}'")
        rows.append(
            {
                "video_id": row["video_id"],
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "subject_id": row["subject_id"],
                "source_dataset": dataset,
                "is_yawning": mapped["is_yawning"],
                "is_look_away": mapped["is_look_away"],
                "is_fatigued": mapped["is_fatigued"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw fatigue annotations to unified labels.")
    parser.add_argument("--input", required=True, help="Input CSV manifest with source labels.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()

    manifest = pd.read_csv(args.input)
    converted = convert_manifest(manifest)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted.to_csv(output_path, index=False)
    print(f"Saved normalized labels to {output_path}")


if __name__ == "__main__":
    main()
