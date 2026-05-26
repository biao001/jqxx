from __future__ import annotations

import sys
from types import SimpleNamespace

from backend.app.analysis import resolve_yolo_device


def test_resolve_yolo_device_prefers_cuda_when_available(monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert resolve_yolo_device("auto") == "cuda"


def test_resolve_yolo_device_falls_back_to_cpu_without_cuda(monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert resolve_yolo_device("auto") == "cpu"


def test_resolve_yolo_device_keeps_explicit_device(monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert resolve_yolo_device("cuda") == "cuda"
