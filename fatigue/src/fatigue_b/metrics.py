"""Metrics helpers for fatigue model B."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.fatigue_b.config import PUBLIC_LABELS, scores_to_public_label


HEAD_NAMES = ["yawn", "look_away", "fatigue"]


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def multihead_metrics(targets: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, dict[str, float]]:
    metrics = {}
    for index, name in enumerate(HEAD_NAMES):
        predictions = (scores[:, index] >= threshold).astype(np.int32)
        metrics[name] = binary_metrics(targets[:, index], predictions)
    macro_f1 = float(np.mean([metrics[name]["f1"] for name in HEAD_NAMES]))
    metrics["macro_f1"] = {"value": macro_f1}
    return metrics


def targets_to_public_labels(targets: np.ndarray) -> list[str]:
    labels = []
    for yawning, look_away, fatigued in targets:
        if fatigued >= 1:
            labels.append("Fatigued Driving")
        elif yawning >= 1:
            labels.append("Yawning")
        elif look_away >= 1:
            labels.append("Looking Around")
        else:
            labels.append("Normal")
    return labels


def scores_to_public_labels(scores: np.ndarray, thresholds: dict[str, float]) -> list[str]:
    labels = []
    for yawning, look_away, fatigued in scores:
        labels.append(
            scores_to_public_label(
                yawn_score=float(yawning),
                look_away_score=float(look_away),
                fatigue_score=float(fatigued),
                thresholds=thresholds,
            )
        )
    return labels


def confusion_matrix(y_true: list[str], y_pred: list[str], classes: list[str] | None = None) -> np.ndarray:
    classes = classes or PUBLIC_LABELS
    index = {label: idx for idx, label in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int32)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[index[true_label], index[pred_label]] += 1
    return matrix


def public_macro_f1(y_true: list[str], y_pred: list[str], classes: list[str] | None = None) -> float:
    classes = classes or PUBLIC_LABELS
    matrix = confusion_matrix(y_true, y_pred, classes)
    f1_scores = []
    for class_index in range(len(classes)):
        tp = matrix[class_index, class_index]
        fp = matrix[:, class_index].sum() - tp
        fn = matrix[class_index, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores))


def save_json(payload: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
