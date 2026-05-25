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
        if not self.available or self.recognizer is None:
            return {"name": None, "status": "no_model"}
        crop = self._detect_crop(bgr)
        if crop is None:
            return {"name": None, "status": "no_face"}
        label, dist = self.recognizer.predict(crop)
        if dist <= MATCH_THRESHOLD:
            return {"name": self.labels.get(int(label)), "status": "known", "distance": round(float(dist), 1)}
        return {"name": None, "status": "unknown", "distance": round(float(dist), 1)}

    def list_drivers(self) -> list[str]:
        return sorted(self.labels.values())

    def delete(self, name: str) -> bool:
        pdir = self.samples_dir / name
        if pdir.exists():
            shutil.rmtree(pdir)
            self._train()
            return True
        return False
