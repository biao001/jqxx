"""Export dataset counts for member 2 handoff and defense slides."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_summary(dataframe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {
        "by_split": dataframe.groupby("split").size().reset_index(name="count"),
        "by_source_dataset": dataframe.groupby("source_dataset").size().reset_index(name="count"),
        "by_target": pd.DataFrame(
            {
                "target": ["is_yawning", "is_look_away", "is_fatigued"],
                "count": [
                    int(dataframe["is_yawning"].sum()),
                    int(dataframe["is_look_away"].sum()),
                    int(dataframe["is_fatigued"].sum()),
                ],
            }
        ),
        "by_subject": dataframe.groupby("subject_id").size().reset_index(name="count"),
    }
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fatigue dataset summary tables.")
    parser.add_argument("--input", required=True, help="Combined label CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for summary CSV files.")
    args = parser.parse_args()

    dataframe = pd.read_csv(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, table in build_summary(dataframe).items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    print(f"Saved dataset summary tables to {output_dir}")


if __name__ == "__main__":
    main()
