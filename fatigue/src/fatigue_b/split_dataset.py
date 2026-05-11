"""Split normalized fatigue labels by subject id."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def split_by_subject(dataframe: pd.DataFrame, seed: int, train_ratio: float, val_ratio: float):
    subjects = sorted(dataframe["subject_id"].unique().tolist())
    random.Random(seed).shuffle(subjects)

    train_cut = max(1, int(len(subjects) * train_ratio))
    val_cut = max(train_cut + 1, int(len(subjects) * (train_ratio + val_ratio)))

    train_subjects = set(subjects[:train_cut])
    val_subjects = set(subjects[train_cut:val_cut])
    test_subjects = set(subjects[val_cut:])

    train_df = dataframe[dataframe["subject_id"].isin(train_subjects)].copy()
    val_df = dataframe[dataframe["subject_id"].isin(val_subjects)].copy()
    test_df = dataframe[dataframe["subject_id"].isin(test_subjects)].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    return train_df, val_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Split fatigue labels by subject id.")
    parser.add_argument("--input", required=True, help="Normalized label CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for train/val/test CSV files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    dataframe = pd.read_csv(args.input)
    train_df, val_df, test_df = split_by_subject(
        dataframe=dataframe,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    print(f"Saved split CSV files to {output_dir}")


if __name__ == "__main__":
    main()
