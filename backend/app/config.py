from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .llm import LlmSettings


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    runtime_dir: Path
    allowed_origins: list[str]
    allowed_origin_regex: str
    llm: LlmSettings
    frame_stride: int = 10
    max_video_frames: int = 240
    fatigue_yawn_threshold: float = 0.55
    fatigue_look_away_threshold: float = 0.55
    fatigue_fatigue_threshold: float = 0.60


def get_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv_file(repo_root / "backend" / ".env")
    runtime_dir = Path(os.getenv("DMS_RUNTIME_DIR", repo_root / "backend" / "runtime"))
    origins = os.getenv("DMS_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    # 默认放行任意来源的跨域请求(本地/局域网演示工具，手机扫码访问 http://<本机IP>:3000)。
    # 部分网络的内网使用公网网段地址，故不限定私有网段;生产可用 DMS_ALLOWED_ORIGIN_REGEX 收紧。
    origin_regex = os.getenv("DMS_ALLOWED_ORIGIN_REGEX", r"^https?://.*$")
    return Settings(
        repo_root=repo_root,
        runtime_dir=runtime_dir,
        allowed_origins=[item.strip() for item in origins.split(",") if item.strip()],
        allowed_origin_regex=origin_regex,
        llm=LlmSettings(
            base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
            api_key=os.getenv("SILICONFLOW_API_KEY", ""),
            model=os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen3-8B"),
            timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
        ),
        frame_stride=max(1, int(os.getenv("DMS_FRAME_STRIDE", "10"))),
        max_video_frames=max(1, int(os.getenv("DMS_MAX_VIDEO_FRAMES", "240"))),
        fatigue_yawn_threshold=float(os.getenv("DMS_FATIGUE_YAWN_THRESHOLD", "0.55")),
        fatigue_look_away_threshold=float(os.getenv("DMS_FATIGUE_LOOK_AWAY_THRESHOLD", "0.55")),
        fatigue_fatigue_threshold=float(os.getenv("DMS_FATIGUE_FATIGUE_THRESHOLD", "0.60")),
    )
