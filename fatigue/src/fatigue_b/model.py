"""Temporal model for fatigue detection algorithm B."""

from __future__ import annotations

import torch
from torch import nn


class TemporalFatigueModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
        rnn_type: str = "lstm",
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.rnn_type = rnn_type.lower()
        recurrent_dropout = dropout if num_layers > 1 else 0.0

        if self.rnn_type == "gru":
            self.encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=recurrent_dropout,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=recurrent_dropout,
                bidirectional=bidirectional,
                batch_first=True,
            )

        encoded_dim = hidden_size * (2 if bidirectional else 1)
        self.projection = nn.Sequential(
            nn.Linear(encoded_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.yawn_head = nn.Linear(hidden_size, 1)
        self.look_away_head = nn.Linear(hidden_size, 1)
        self.fatigue_head = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        normalized = self.input_norm(features)
        encoded, _ = self.encoder(normalized)
        pooled = encoded.mean(dim=1)
        shared = self.projection(pooled)
        return {
            "yawn": self.yawn_head(shared).squeeze(-1),
            "look_away": self.look_away_head(shared).squeeze(-1),
            "fatigue": self.fatigue_head(shared).squeeze(-1),
        }


def build_model_from_checkpoint(checkpoint: dict) -> TemporalFatigueModel:
    config = checkpoint["model_config"]
    model = TemporalFatigueModel(**config)
    model.load_state_dict(checkpoint["model_state"])
    return model
