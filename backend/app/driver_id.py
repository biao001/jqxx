"""车主识别：基于 OpenCV LBPH 人脸识别(无需额外依赖，opencv-contrib 自带 cv2.face)。

登记时采集人脸灰度裁剪样本并持久化到 runtime 目录，重训 LBPH；
识别时检测人脸→预测身份，距离越小越像，超过阈值判为"未登记"。
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# LBPH 距离阈值：小于即认为是已登记车主(典型 <50 很像, <80 可接受)
MATCH_THRESHOLD = 68.0
FACE_SIZE = (160, 160)


class DriverIdentifier:
    def __init__(self, store_dir: Path):
        self.dir = store_dir
        self.samples_dir = store_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.recognizer: Any | None = None
        self.labels: dict[int, str] = {}
        self.available = hasattr(cv2, "face")
        self._train()

    def _detect_crop(self, bgr: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return cv2.resize(gray[y : y + h, x : x + w], FACE_SIZE)

    def _train(self) -> None:
        if not self.available:
            return
        samples: list[np.ndarray] = []
        labels: list[int] = []
        self.labels = {}
        for idx, person_dir in enumerate(sorted(p for p in self.samples_dir.glob("*") if p.is_dir())):
            self.labels[idx] = person_dir.name
            for f in person_dir.glob("*.png"):
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    samples.append(img)
                    labels.append(idx)
        if samples:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.train(samples, np.array(labels))
        else:
            self.recognizer = None

    def register(self, name: str, images_bgr: list[np.ndarray]) -> int:
        if not self.available:
            raise RuntimeError("cv2.face 不可用(需 opencv-contrib)")
        name = name.strip()[:40] or "driver"
        pdir = self.samples_dir / name
        pdir.mkdir(parents=True, exist_ok=True)
        existing = len(list(pdir.glob("*.png")))
        saved = 0
        for img in images_bgr:
            crop = self._detect_crop(img)
            if crop is not None:
                cv2.imwrite(str(pdir / f"{existing + saved}.png"), crop)
                saved += 1
        if saved == 0 and existing == 0:
            pdir.rmdir()  # 一张都没采到，别留空目录
        self._train()
        return saved

    def identify(self, bgr: np.ndarray) -> dict[str, Any]:
        """检测画面中所有人脸，返回**驾驶员那张脸**的身份与位置(box=xyxy)。

        - 已登记:对每张脸跑 LBPH，取匹配上注册车主且距离最小的那张 → status=known。
        - 未匹配/未登记:取最居中的那张脸作位置兜底 → status=unknown/no_model(box 仍返回，供 ROI 定位)。
        - 无脸:status=no_face，box=None。
        """
        h_img, w_img = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return {"name": None, "status": "no_face", "box": None}

        # 逐张脸做 LBPH，挑匹配上注册车主、距离最小的那张
        best: tuple[float, int, tuple[int, int, int, int]] | None = None
        if self.available and self.recognizer is not None:
            for (x, y, w, h) in faces:
                crop = cv2.resize(gray[y : y + h, x : x + w], FACE_SIZE)
                label, dist = self.recognizer.predict(crop)
                if dist <= MATCH_THRESHOLD and (best is None or dist < best[0]):
                    best = (float(dist), int(label), (int(x), int(y), int(w), int(h)))

        def xyxy(f: tuple[int, int, int, int]) -> list[int]:
            x, y, w, h = f
            return [x, y, x + w, y + h]

        if best is not None:
            d, label, f = best
            return {"name": self.labels.get(label), "status": "known", "distance": round(d, 1), "box": xyxy(f)}

        # 未匹配 → 最居中的脸作位置兜底
        cx0, cy0 = w_img / 2, h_img / 2
        fx, fy, fw, fh = min(faces, key=lambda f: ((f[0] + f[2] / 2) - cx0) ** 2 + ((f[1] + f[3] / 2) - cy0) ** 2)
        status = "unknown" if self.recognizer is not None else "no_model"
        return {"name": None, "status": status, "box": xyxy((int(fx), int(fy), int(fw), int(fh)))}

    def list_drivers(self) -> list[str]:
        return sorted(self.labels.values())

    def delete(self, name: str) -> bool:
        pdir = self.samples_dir / name
        if pdir.exists():
            shutil.rmtree(pdir)
            self._train()
            return True
        return False
