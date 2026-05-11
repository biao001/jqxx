"""Evaluate fatigue model B and optionally search thresholds."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.fatigue_b.config import DEFAULT_THRESHOLDS, PUBLIC_LABELS
from src.fatigue_b.dataset import FatigueWindowDataset
from src.fatigue_b.metrics import (
    confusion_matrix,
    multihead_metrics,
    public_macro_f1,
    save_json,
    scores_to_public_labels,
    sigmoid,
    targets_to_public_labels,
)
from src.fatigue_b.model import build_model_from_checkpoint
from src.fatigue_b.runtime import resolve_torch_device


def collect_scores(model, loader, device):
    targets = []
    scores = []
    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits = model(features)
            logit_tensor = torch.stack(
                [logits["yawn"], logits["look_away"], logits["fatigue"]],
                dim=1,
            )
            scores.append(sigmoid(logit_tensor.cpu().numpy()))
            targets.append(labels.numpy())
    return np.concatenate(targets, axis=0), np.concatenate(scores, axis=0)


def search_thresholds(targets: np.ndarray, scores: np.ndarray) -> tuple[dict[str, float], float]:
    true_public = targets_to_public_labels(targets)
    best_thresholds = DEFAULT_THRESHOLDS.copy()
    best_score = -1.0

    grid = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    for yawn_t, look_t, fatigue_t in itertools.product(grid, grid, grid):
        thresholds = {"yawn": yawn_t, "look_away": look_t, "fatigue": fatigue_t}
        pred_public = scores_to_public_labels(scores, thresholds)
        macro_f1 = public_macro_f1(true_public, pred_public)
        if macro_f1 > best_score:
            best_score = macro_f1
            best_thresholds = thresholds
    return best_thresholds, best_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fatigue model B.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--search-thresholds", action="store_true")
    args = parser.parse_args()

    resolved_device = resolve_torch_device(args.device)
    device = torch.device(resolved_device)
    print(f"Using device: {resolved_device}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint(checkpoint).to(device)
    dataset = FatigueWindowDataset(args.test_npz)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    targets, scores = collect_scores(model, loader, device)
    thresholds = checkpoint.get("thresholds", DEFAULT_THRESHOLDS)
    best_public_f1 = None

    if args.search_thresholds:
        thresholds, best_public_f1 = search_thresholds(targets, scores)

    head_metrics = multihead_metrics(targets, scores, threshold=0.5)
    true_public = targets_to_public_labels(targets)
    pred_public = scores_to_public_labels(scores, thresholds)
    matrix = confusion_matrix(true_public, pred_public, PUBLIC_LABELS)
    public_f1 = public_macro_f1(true_public, pred_public)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=PUBLIC_LABELS, columns=PUBLIC_LABELS).to_csv(output_dir / "confusion_matrix.csv")
    save_json(
        {
            "thresholds": thresholds,
            "head_metrics": head_metrics,
            "public_macro_f1": public_f1,
            "best_public_f1_from_search": best_public_f1,
        },
        output_dir / "report.json",
    )
    print(f"Saved evaluation report to {output_dir}")


if __name__ == "__main__":
    main()
