"""
download_datasets.py — 从 Roboflow 批量下载 DMS 训练数据集

需要环境变量:
  ROBOFLOW_API_KEY=<你的 Roboflow Private API Key>

下载 3 个数据集（各自公开 CC BY 4.0）:
  1. distracted-driving-yolov8 (yolov8-ei4l6)       → 5 类行为
  2. cigarette-wkkgi           (yolov8-jymgm)       → 香烟
  3. seatbelt-detection-lb1ec  (seatbelttraining-7yh0f) → 安全带

下载完成后运行 scripts/merge_datasets.py 合并成统一 8 类数据集。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "data" / "rf_datasets"

DATASETS = [
    # (workspace, project, version, local_subdir)
    ("yolov8-ei4l6",        "distracted-driving-yolov8", 6, "distracted_driving"),
    ("yolov8-jymgm",        "cigarette-wkkgi",           5, "cigarette"),
    ("seatbelttraining-7yh0f", "seatbelt-detection-lb1ec", 4, "seatbelt"),
]


def main():
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        print("[!] 请先设置环境变量 ROBOFLOW_API_KEY")
        print("    Windows PowerShell:  $env:ROBOFLOW_API_KEY='YOUR_KEY'")
        print("    Windows cmd:         set ROBOFLOW_API_KEY=YOUR_KEY")
        print("    Bash/zsh:            export ROBOFLOW_API_KEY=YOUR_KEY")
        print("\n    API Key 获取: https://app.roboflow.com/settings/api")
        sys.exit(2)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("[!] 请先安装: pip install roboflow")
        sys.exit(2)

    rf = Roboflow(api_key=key)
    OUT.mkdir(parents=True, exist_ok=True)

    for i, (ws, proj, ver, sub) in enumerate(DATASETS, 1):
        target = OUT / sub
        if (target / "data.yaml").exists():
            print(f"[{i}/{len(DATASETS)}] {sub} 已存在，跳过")
            continue
        print(f"[{i}/{len(DATASETS)}] {ws}/{proj} v{ver} → {target}")
        try:
            rf.workspace(ws).project(proj).version(ver).download(
                "yolov8", location=str(target))
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            continue
        print(f"  ✓ 完成")

    print("\n[next] 运行 python scripts/merge_datasets.py 合并为统一数据集")


if __name__ == "__main__":
    main()
