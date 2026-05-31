"""一键训练管理：后台子进程跑 train_unified.py，解析进度，完成后供热部署。"""
from __future__ import annotations

import csv as _csv
import json as _json
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
        self.out_model = "unified.pt"   # 本次训练产出的权重文件名
        self.project = ""               # 本次训练的项目名(展示用)
        self.base = "yolov8s.pt"        # 本次训练的基础权重(展示用)
        self._lock = Lock()

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, cfg: dict[str, Any], data_yaml: Path, out_model: str = "unified.pt", project: str = "") -> None:
        with self._lock:
            if self.running():
                raise RuntimeError("已有训练正在进行")
            if not data_yaml.exists():
                raise RuntimeError("数据集不存在，请先构建/上传")
            self.out_model = out_model
            self.project = project
            self.base = str(cfg.get("base", "yolov8s.pt"))
            # 每次训练用带时间戳的独立 run 目录：保留每一轮实验产物(csv/曲线图)供研究页回看/对比，
            # 不再覆盖上一轮(故无需再清理旧目录)。
            self.run_name = f"ui_{project}_{int(time.time())}" if project else f"ui_train_{int(time.time())}"
            models = self.behavior_dir / "models"
            cur = models / out_model
            if cur.exists():  # 训练前备份同名旧权重
                shutil.copy(cur, models / f"{Path(out_model).stem}.prev.backup.pt")
            self.total_epochs = int(cfg.get("epochs", 50) or 50)
            cmd = [
                sys.executable, "scripts/train_unified.py",
                "--data", str(data_yaml),
                "--base", str(cfg.get("base", "yolov8s.pt")),
                "--epochs", str(self.total_epochs),
                "--imgsz", str(int(cfg.get("imgsz", 768) or 768)),
                "--batch", str(int(cfg.get("batch", 12) or 12)),
                "--patience", str(int(cfg.get("patience", 30) or 30)),
                "--freeze", str(int(cfg.get("freeze", 0) or 0)),
                "--lr0", str(float(cfg.get("lr0", 0.001) or 0.001)),
                "--out", out_model,
                "--device", "0", "--name", self.run_name,
            ]
            if cfg.get("full_data"):
                cmd.append("--all-splits")          # 演示：valid/test 并入训练、train 当验证(过拟合)
            if cfg.get("include_incremental"):
                cmd.append("--include-incremental")  # 把 unassigned 增量也纳入训练
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            logf = open(self.log_path, "w")
            self.proc = subprocess.Popen(cmd, cwd=str(self.behavior_dir), stdout=logf, stderr=subprocess.STDOUT)
            self.started_at = time.time()

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
        # 各类别 mAP(逐 epoch 由训练回调写出) —— 让训练时能看到每个标签的精度，而非只有总 mAP50
        per_class: dict = {}
        pcp = self.behavior_dir / "runs" / "unified" / self.run_name / "per_class.json"
        if pcp.exists():
            try:
                per_class = _json.loads(pcp.read_text())
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
            "project": self.project,
            "out_model": self.out_model,
            "base": self.base,
            "run": self.run_name,
            "per_class": per_class,
            "epoch": epoch,
            "total_epochs": self.total_epochs,
            "best_map": round(best_map, 3),
            "finished": self.proc is not None and not running,
            "success": rc == 0,
            "elapsed": round(time.time() - self.started_at) if self.started_at else 0,
            "log_tail": log_tail,
        }
