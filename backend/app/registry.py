"""多数据集/多模型项目注册表。

每个"项目(槽位)"= datasets/<name>/ 一个 7 类数据集 + behavior_algo/models/<model> 一个权重。
另维护一个"当前生效模型"指针(active_model)，供实时检测器加载。
注册表与指针持久化到 runtime/ 下的 JSON，重启不丢。
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

# 全系统统一的 7 类行为 schema(与 datasets/reckless_mapped/data.yaml 保持一致)。
# no_seatbelt 由检测器逻辑从 seatbelt 推断，不单独训练。
STANDARD_NAMES = [
    "phone_use", "smoking", "drinking", "eating",
    "hand_on_wheel", "hands_off_wheel", "seatbelt",
]

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")

# 用户上传/管理的演示数据集统一放在仓库根的 datasets/ 下，与算法代码/源数据集分开。
USER_DATASETS_DIR = "datasets"

# 内置默认项目(首次运行写入注册表)。builtin 项目不可删除/改名。
# dir 为相对仓库根的数据集目录:所有数据集(现役/源/演示)统一放在 datasets/ 下。
_DEFAULT_PROJECTS = [
    {"name": "reckless_mapped", "label": "行为识别(现役)", "model": "unified.pt", "builtin": True,
     "dir": f"{USER_DATASETS_DIR}/reckless_mapped"},
    {"name": "realtime_demo", "label": "实时演示", "model": "realtime.pt", "builtin": False,
     "dir": f"{USER_DATASETS_DIR}/realtime_demo"},
    {"name": "video_demo", "label": "视频演示", "model": "video.pt", "builtin": False,
     "dir": f"{USER_DATASETS_DIR}/video_demo"},
]


def _default_dir(name: str) -> str:
    return f"{USER_DATASETS_DIR}/{name}"


class ProjectRegistry:
    def __init__(self, repo_root: Path, runtime_dir: Path):
        self.repo_root = repo_root
        self.models_root = repo_root / "behavior_algo" / "models"
        self._projects_path = runtime_dir / "projects.json"
        self._active_path = runtime_dir / "active_model.json"
        runtime_dir.mkdir(parents=True, exist_ok=True)

    # ---- 注册表读写 ----
    def load_projects(self) -> list[dict[str, Any]]:
        if self._projects_path.exists():
            try:
                items = json.loads(self._projects_path.read_text())
                if isinstance(items, list) and items:
                    # 旧版注册表回填/迁移 dir：缺失则按约定补全；旧的 behavior_algo/data/ 迁到 datasets/
                    changed = False
                    for it in items:
                        if not it.get("dir"):
                            it["dir"] = _default_dir(it["name"])
                            changed = True
                        elif it["dir"].startswith("behavior_algo/data/"):
                            it["dir"] = "datasets/" + it["dir"].split("/", 2)[-1]
                            changed = True
                    if changed:
                        self._save_projects(items)
                    return items
            except Exception:
                pass
        self._save_projects(_DEFAULT_PROJECTS)
        return [dict(p) for p in _DEFAULT_PROJECTS]

    def _save_projects(self, items: list[dict[str, Any]]) -> None:
        self._projects_path.write_text(json.dumps(items, ensure_ascii=False, indent=2))

    def get(self, name: str) -> dict[str, Any] | None:
        return next((p for p in self.load_projects() if p["name"] == name), None)

    # ---- 路径 ----
    def dataset_dir(self, name: str) -> Path:
        p = self.get(name)
        if p is None:
            raise KeyError(f"未知项目: {name}")
        return self.repo_root / p.get("dir", _default_dir(name))

    def model_path(self, name: str) -> Path:
        p = self.get(name)
        if p is None:
            raise KeyError(f"未知项目: {name}")
        return self.models_root / p["model"]

    # ---- 创建空数据集 ----
    def ensure_dataset_dirs(self, name: str) -> None:
        root = self.dataset_dir(name)
        for split in ("unassigned", "train", "valid", "test"):
            (root / split / "images").mkdir(parents=True, exist_ok=True)
            (root / split / "labels").mkdir(parents=True, exist_ok=True)
        yaml_path = root / "data.yaml"
        if not yaml_path.exists():
            names_inline = "[" + ", ".join(f"'{n}'" for n in STANDARD_NAMES) + "]"
            yaml_path.write_text(
                f"path: {root}\n"
                "train: train/images\n"
                "val: valid/images\n"
                "test: test/images\n"
                f"nc: {len(STANDARD_NAMES)}\n"
                f"names: {names_inline}\n"
            )

    def create_project(self, name: str, label: str) -> dict[str, Any]:
        name = name.strip()
        if not _NAME_RE.match(name):
            raise ValueError("项目名需为小写字母开头，仅含小写字母/数字/下划线，长度2-31")
        if self.get(name) is not None:
            raise ValueError("项目已存在")
        entry = {"name": name, "label": label.strip() or name, "model": f"{name}.pt",
                 "builtin": False, "dir": f"{USER_DATASETS_DIR}/{name}"}
        items = self.load_projects()
        items.append(entry)
        self._save_projects(items)
        self.ensure_dataset_dirs(name)
        return entry

    def delete_project(self, name: str) -> dict[str, Any]:
        items = self.load_projects()
        proj = next((p for p in items if p["name"] == name), None)
        if proj is None:
            raise ValueError("项目不存在")
        if proj.get("builtin"):
            raise ValueError("内置(现役)项目不可删除")
        # 删数据集目录 + 模型文件
        ddir = self.repo_root / proj.get("dir", _default_dir(name))
        if ddir.exists():
            shutil.rmtree(ddir, ignore_errors=True)
        mp = self.models_root / proj["model"]
        if mp.exists():
            mp.unlink()
        reset_active = self.active_model() == proj["model"]
        if reset_active:  # 删的是生效模型 → 回退 unified.pt
            self._active_path.write_text(json.dumps({"model": "unified.pt"}, ensure_ascii=False))
        self._save_projects([p for p in items if p["name"] != name])
        return {"deleted": name, "reset_active": reset_active}

    def rename_project(self, name: str, new_label: str | None = None, new_name: str | None = None) -> dict[str, Any]:
        items = self.load_projects()
        proj = next((p for p in items if p["name"] == name), None)
        if proj is None:
            raise ValueError("项目不存在")
        active_changed = False
        if new_label and new_label.strip():
            proj["label"] = new_label.strip()
        if new_name and new_name != name:
            if proj.get("builtin"):
                raise ValueError("内置项目不可修改标识名")
            new_name = new_name.strip()
            if not _NAME_RE.match(new_name):
                raise ValueError("名称需为小写字母开头，仅含小写字母/数字/下划线，长度2-31")
            if any(p["name"] == new_name for p in items):
                raise ValueError("目标名已存在")
            old_dir = self.repo_root / proj.get("dir", _default_dir(name))
            new_dir = self.repo_root / f"{USER_DATASETS_DIR}/{new_name}"
            if old_dir.exists():
                old_dir.rename(new_dir)
                yml = new_dir / "data.yaml"  # 同步 data.yaml 里的绝对 path
                if yml.exists():
                    lines = yml.read_text().splitlines()
                    lines = [f"path: {new_dir}" if ln.startswith("path:") else ln for ln in lines]
                    yml.write_text("\n".join(lines) + "\n")
            old_model = self.models_root / proj["model"]
            new_model = f"{new_name}.pt"
            if old_model.exists():
                old_model.rename(self.models_root / new_model)
            if self.active_model() == proj["model"]:
                self._active_path.write_text(json.dumps({"model": new_model}, ensure_ascii=False))
                active_changed = True
            proj.update(name=new_name, dir=f"{USER_DATASETS_DIR}/{new_name}", model=new_model)
        self._save_projects(items)
        return {"project": proj, "active_changed": active_changed}

    # ---- 统计概览 ----
    def list_with_stats(self) -> list[dict[str, Any]]:
        active = self.active_model()
        out = []
        for p in self.load_projects():
            ds = self.repo_root / p.get("dir", _default_dir(p["name"]))
            imgs = {}
            for split in ("train", "valid", "test"):
                d = ds / split / "images"
                imgs[split] = sum(1 for _ in d.iterdir()) if d.exists() else 0
            mp = self.models_root / p["model"]
            out.append({
                **p,
                "dataset_exists": (ds / "data.yaml").exists(),
                "images": imgs,
                "model_exists": mp.exists(),
                "model_size_mb": round(mp.stat().st_size / 1e6, 1) if mp.exists() else 0,
                "model_mtime": int(mp.stat().st_mtime) if mp.exists() else 0,
                "active": p["model"] == active,
            })
        return out

    # ---- 当前生效模型 ----
    def active_model(self) -> str:
        if self._active_path.exists():
            try:
                return str(json.loads(self._active_path.read_text()).get("model") or "unified.pt")
            except Exception:
                pass
        return "unified.pt"

    def set_active_model(self, model: str) -> str:
        model = Path(str(model)).name  # 防目录穿越
        known = {p["model"] for p in self.load_projects()}
        if model not in known:
            raise ValueError("未知模型")
        if not (self.models_root / model).exists():
            raise ValueError("该模型尚未训练生成(.pt 不存在)")
        self._active_path.write_text(json.dumps({"model": model}, ensure_ascii=False))
        return model
