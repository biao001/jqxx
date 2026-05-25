"""一键训练管理：后台子进程跑 train_unified.py，解析进度，完成后供热部署。"""
from __future__ import annotations

import csv as _csv
import shutil
import subprocess
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any


class TrainingManager:
    def __init__(self, repo_root: Path, log_path: Path):
        self.behavior_dir = repo_root / "behavior_algo"
        self.log_path = log_path
        self.run_name = "ui_train"
        self.proc: subprocess.Popen | None = None
        self.started_at = 0.0
        self.total_epochs = 0
        self.deployed = True  # 是否已对完成的训练做过热部署
        self._lock = Lock()

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, cfg: dict[str, Any], data_yaml: Path) -> None:
        with self._lock:
            if self.running():
                raise RuntimeError("已有训练正在进行")
            if not data_yaml.exists():
                raise RuntimeError("数据集不存在，请先构建/上传")
            models = self.behavior_dir / "models"
            cur = models / "unified.pt"
            if cur.exists():
                shutil.copy(cur, models / "unified.prev.backup.pt")  # 训练前备份
            self.total_epochs = int(cfg.get("epochs", 50) or 50)
            cmd = [
                sys.executable, "scripts/train_unified.py",
                "--data", str(data_yaml),
                "--base", str(cfg.get("base", "yolov8s.pt")),
                "--epochs", str(self.total_epochs),
                "--imgsz", str(int(cfg.get("imgsz", 768) or 768)),
                "--batch", str(int(cfg.get("batch", 12) or 12)),
                "--patience", str(int(cfg.get("patience", 30) or 30)),
                "--device", "0", "--name", self.run_name,
            ]
            # 清理上轮 run 以便进度从零统计
            run_dir = self.behavior_dir / "runs" / "unified" / self.run_name
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            logf = open(self.log_path, "w")
            self.proc = subprocess.Popen(cmd, cwd=str(self.behavior_dir), stdout=logf, stderr=subprocess.STDOUT)
            self.started_at = time.time()
            self.deployed = False

    def status(self) -> dict[str, Any]:
        running = self.running()
        rc = self.proc.poll() if self.proc else None
        epoch, best_map = 0, 0.0
        csvp = self.behavior_dir / "runs" / "unified" / self.run_name / "results.csv"
        if csvp.exists():
            try:
                rows = list(_csv.DictReader(open(csvp)))
                epoch = len(rows)
                key = next((k for k in (rows[0] if rows else {}) if "mAP50(B)" in k), None)
                if key and rows:
                    best_map = max(float(r[key] or 0) for r in rows)
            except Exception:
                pass
        log_tail = ""
        if self.log_path.exists():
            lines = [ln for ln in self.log_path.read_text(errors="ignore").splitlines() if ln.strip()]
            log_tail = "\n".join(lines[-12:])
        return {
            "running": running,
            "started": self.started_at > 0,
            "started_at": self.started_at,
            "epoch": epoch,
            "total_epochs": self.total_epochs,
            "best_map": round(best_map, 3),
            "finished": self.proc is not None and not running,
            "success": rc == 0,
            "elapsed": round(time.time() - self.started_at) if self.started_at else 0,
            "log_tail": log_tail,
        }
