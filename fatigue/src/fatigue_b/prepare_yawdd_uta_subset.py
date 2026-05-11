"""Prepare the selected YawDD and UTA-RLDD subset for fatigue model training."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "fatigue_b" / "raw"
SELECTED_DIR = RAW_ROOT / "selected_videos"
LABEL_DIR = ROOT / "data" / "fatigue_b" / "labels"
SUMMARY_DIR = ROOT / "outputs" / "fatigue_b" / "reports" / "dataset_summary"


@dataclass(frozen=True)
class Sample:
    source_path: Path
    video_id: str
    subject_id: str
    source_dataset: str
    is_yawning: int
    is_look_away: int
    is_fatigued: int
    split: str


def frame_count(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if count <= 0:
        raise RuntimeError(f"Video has no readable frames: {video_path}")
    return count


def build_samples() -> list[Sample]:
    yawdd = RAW_ROOT / "YawDD" / "dataset" / "YawDD dataset" / "Mirror" / "Female_mirror"
    uta = RAW_ROOT / "UTA-RLDD" / "subset" / "Fold1_part1" / "04"

    return [
        Sample(yawdd / "1-FemaleNoGlasses-Normal.avi", "yawdd_s001_normal", "yawdd_s001", "YawDD", 0, 0, 0, "train"),
        Sample(yawdd / "1-FemaleNoGlasses-Yawning.avi", "yawdd_s001_yawning", "yawdd_s001", "YawDD", 1, 0, 0, "train"),
        Sample(yawdd / "2-FemaleNoGlasses-Normal.avi", "yawdd_s002_normal", "yawdd_s002", "YawDD", 0, 0, 0, "train"),
        Sample(yawdd / "2-FemaleNoGlasses-Yawning.avi", "yawdd_s002_yawning", "yawdd_s002", "YawDD", 1, 0, 0, "train"),
        Sample(yawdd / "3-FemaleGlasses-Normal.avi", "yawdd_s003_normal", "yawdd_s003", "YawDD", 0, 0, 0, "val"),
        Sample(yawdd / "3-FemaleGlasses-Yawning.avi", "yawdd_s003_yawning", "yawdd_s003", "YawDD", 1, 0, 0, "val"),
        Sample(yawdd / "4-FemaleGlasses-Normal.avi", "yawdd_s004_normal", "yawdd_s004", "YawDD", 0, 0, 0, "test"),
        Sample(yawdd / "4-FemaleGlasses-Yawning.avi", "yawdd_s004_yawning", "yawdd_s004", "YawDD", 1, 0, 0, "test"),
        Sample(uta / "0.mp4", "uta_s004_alert", "uta_s004", "UTA-RLDD", 0, 0, 0, "train"),
        Sample(uta / "10.mp4", "uta_s004_drowsy", "uta_s004", "UTA-RLDD", 0, 0, 1, "train"),
    ]


def copy_video(sample: Sample) -> Path:
    if not sample.source_path.exists():
        raise FileNotFoundError(f"Missing source video: {sample.source_path}")
    suffix = sample.source_path.suffix.lower()
    target_path = SELECTED_DIR / f"{sample.video_id}{suffix}"
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    if not target_path.exists() or target_path.stat().st_size != sample.source_path.stat().st_size:
        shutil.copy2(sample.source_path, target_path)
    return target_path


def main() -> None:
    rows = []
    for sample in build_samples():
        target_path = copy_video(sample)
        frames = frame_count(target_path)
        rows.append(
            {
                "video_id": sample.video_id,
                "start_frame": 0,
                "end_frame": frames - 1,
                "subject_id": sample.subject_id,
                "source_dataset": sample.source_dataset,
                "is_yawning": sample.is_yawning,
                "is_look_away": sample.is_look_away,
                "is_fatigued": sample.is_fatigued,
                "split": sample.split,
            }
        )

    labels = pd.DataFrame(rows)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    labels.to_csv(LABEL_DIR / "all.csv", index=False)
    for split, split_rows in labels.groupby("split", sort=False):
        split_rows.to_csv(LABEL_DIR / f"{split}.csv", index=False)

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    labels.groupby("split").size().rename("count").reset_index().to_csv(SUMMARY_DIR / "by_split.csv", index=False)
    labels.groupby("source_dataset").size().rename("count").reset_index().to_csv(
        SUMMARY_DIR / "by_source_dataset.csv", index=False
    )
    labels.groupby("subject_id").size().rename("count").reset_index().to_csv(
        SUMMARY_DIR / "by_subject.csv", index=False
    )
    labels[["is_yawning", "is_look_away", "is_fatigued"]].sum().rename("count").reset_index().rename(
        columns={"index": "target"}
    ).to_csv(SUMMARY_DIR / "by_target.csv", index=False)

    print(f"Prepared {len(labels)} labeled videos in {SELECTED_DIR}")
    print(f"Wrote labels to {LABEL_DIR}")


if __name__ == "__main__":
    main()
