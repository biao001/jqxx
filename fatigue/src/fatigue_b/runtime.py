"""Runtime helpers for device selection."""

from __future__ import annotations


def resolve_torch_device(requested: str = "auto") -> str:
    """Return the best available torch device.

    Rules:
    - `auto`: prefer CUDA, then MPS, then CPU
    - explicit values such as `cuda`, `cuda:0`, `mps`, `cpu` are respected when available
    - unavailable explicit GPU requests fall back to CPU
    """
    import torch

    requested = (requested or "auto").lower()

    cuda_available = torch.cuda.is_available()
    mps_available = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"

    if requested.startswith("cuda"):
        return requested if cuda_available else "cpu"

    if requested == "mps":
        return "mps" if mps_available else "cpu"

    return "cpu"
