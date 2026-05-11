"""Train fatigue model B on precomputed window features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.fatigue_b.config import DEFAULT_THRESHOLDS
from src.fatigue_b.dataset import FatigueWindowDataset
from src.fatigue_b.metrics import multihead_metrics, save_json, sigmoid
from src.fatigue_b.model import TemporalFatigueModel
from src.fatigue_b.runtime import resolve_torch_device


def collect_scores(model: nn.Module, loader: DataLoader, device: torch.device):
    targets = []
    scores = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    model.eval()
    with torch.no_grad():
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            logits = model(batch_features)
            logit_tensor = torch.stack(
                [logits["yawn"], logits["look_away"], logits["fatigue"]],
                dim=1,
            )
            loss = criterion(logit_tensor, batch_targets)
            total_loss += loss.item() * batch_features.size(0)

            targets.append(batch_targets.cpu().numpy())
            scores.append(sigmoid(logit_tensor.cpu().numpy()))

    all_targets = np.empty((0, 3), dtype=np.float32) if not targets else np.concatenate(targets, axis=0)
    all_scores = np.empty((0, 3), dtype=np.float32) if not scores else np.concatenate(scores, axis=0)
    average_loss = total_loss / max(1, len(loader.dataset))
    return average_loss, all_targets, all_scores


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: Adam, device: torch.device) -> float:
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    total_loss = 0.0

    for batch_features, batch_targets in loader:
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_features)
        logit_tensor = torch.stack(
            [logits["yawn"], logits["look_away"], logits["fatigue"]],
            dim=1,
        )
        loss = criterion(logit_tensor, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_features.size(0)

    return total_loss / max(1, len(loader.dataset))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fatigue model B.")
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--val-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--rnn-type", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    resolved_device = resolve_torch_device(args.device)
    device = torch.device(resolved_device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {resolved_device}")

    train_dataset = FatigueWindowDataset(args.train_npz)
    val_dataset = FatigueWindowDataset(args.val_npz)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model_config = {
        "input_dim": train_dataset.features.shape[-1],
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": True,
        "rnn_type": args.rnn_type,
    }
    model = TemporalFatigueModel(**model_config).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    history = []
    best_macro_f1 = -1.0
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_targets, val_scores = collect_scores(model, val_loader, device)
        val_metrics = multihead_metrics(val_targets, val_scores, threshold=0.5)
        macro_f1 = val_metrics["macro_f1"]["value"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": macro_f1,
                "val_metrics": val_metrics,
            }
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "thresholds": DEFAULT_THRESHOLDS,
                },
                best_path,
            )

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | val_macro_f1={macro_f1:.4f}"
        )

    save_json({"history": history, "best_macro_f1": best_macro_f1}, output_dir / "history.json")
    print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
