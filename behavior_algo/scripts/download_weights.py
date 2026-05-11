"""
download_weights.py — 自动下载/准备所有权重

- yolov8n.pt         (COCO, 6.2MB) — 官方
- yolov8n-pose.pt    (COCO-Pose, 6.5MB) — 官方
- seatbelt.pt        (可选) — 需手动从 HuggingFace / Roboflow 获取
- smoking.pt         (可选) — 需手动从 HuggingFace / Roboflow 获取

公开权重参考地址（若网络通顺可尝试 huggingface_hub 拉取）：
  https://huggingface.co/models?search=seatbelt+yolov8
  https://huggingface.co/models?search=smoking+yolov8
  https://huggingface.co/keremberke (有多个交通相关 yolov8 权重)
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def fetch_official(name: str):
    target = MODELS_DIR / name
    if target.exists():
        print(f"[ok] 已存在: {target}")
        return
    print(f"[..] 下载 {name} ...")
    _ = YOLO(name)  # ultralytics 会缓存到 cwd 或包目录
    # 搜索已下载文件
    for cand in [Path(name), Path.cwd() / name]:
        if cand.exists():
            shutil.copy(cand, target)
            print(f"[ok] 保存到 {target}")
            return
    print(f"[!!] 未找到 {name}，请手动下载后放入 models/")


def fetch_smoking_from_hf():
    """从 HuggingFace 拉取 cigarette 检测器 (Enos-123/smoking-detection)"""
    target = MODELS_DIR / "smoking.pt"
    if target.exists():
        print(f"[ok] 已存在: {target}")
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[!] 缺 huggingface_hub，跳过。安装: pip install huggingface_hub")
        return
    print("[..] 下载 Enos-123/smoking-detection/best.pt (~40MB)")
    p = hf_hub_download(repo_id="Enos-123/smoking-detection",
                        filename="best.pt", local_dir=str(MODELS_DIR))
    src = Path(p)
    if src.exists() and src != target:
        shutil.move(str(src), str(target))
    # 清理 cache
    cache = MODELS_DIR / ".cache"
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)
    print(f"[ok] 保存到 {target}")


def main():
    fetch_official("yolov8n.pt")
    fetch_official("yolov8n-pose.pt")
    fetch_smoking_from_hf()

    # 安全带仍需训练
    if not (MODELS_DIR / "seatbelt.pt").exists():
        print("\n[info] 安全带专用权重 seatbelt.pt 未提供（无稳定的公开 .pt 资源）")
        print("       在未训练前，detector 会启用启发式兜底（置信度 ≤ 0.45）")
        print("       一键训练：")
        print("         pip install roboflow")
        print("         python scripts/train_seatbelt.py --source roboflow \\")
        print("             --rf-api-key <你的Roboflow免费API>")
        print("       或本地数据集：参考 data/seatbelt_example.yaml")


if __name__ == "__main__":
    main()
