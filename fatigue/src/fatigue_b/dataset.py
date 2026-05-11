"""Dataset wrappers for fatigue model B."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FatigueWindowDataset(Dataset):
    def __init__(self, npz_path: str | Path):
        payload = np.load(npz_path)
        self.features = payload["features"].astype(np.float32)
        self.targets = payload["targets"].astype(np.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        feature_tensor = torch.from_numpy(self.features[index])
        target_tensor = torch.from_numpy(self.targets[index])
        return feature_tensor, target_tensor
