"""Extract frame-level fatigue features from driver videos."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]
POSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

FACE_3D_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def ensure_face_landmarker_model(model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        print(f"Downloading MediaPipe face landmarker model to {model_path}")
        urlretrieve(MODEL_URL, model_path)
    return model_path


def create_landmarker(model_path: Path) -> FaceLandmarker:
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    return FaceLandmarker.create_from_options(options)


def create_fallback_cascades() -> tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    return frontal, profile


def distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


def normalize_pose_angle(angle: float) -> float:
    """Map Euler angles to the nearest equivalent in [-90, 90]."""
    wrapped = ((float(angle) + 180.0) % 360.0) - 180.0
    if wrapped > 90.0:
        wrapped -= 180.0
    elif wrapped < -90.0:
        wrapped += 180.0
    return wrapped


def eye_aspect_ratio(landmarks: np.ndarray, eye_indices: list[int]) -> float:
    p1, p2, p3, p4, p5, p6 = [landmarks[idx] for idx in eye_indices]
    vertical = distance(p2, p6) + distance(p3, p5)
    horizontal = 2.0 * distance(p1, p4)
    return vertical / horizontal if horizontal else 0.0


def mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    p1, p2, p3, p4, p5, p6, p7, p8 = [landmarks[idx] for idx in MOUTH]
    vertical = distance(p2, p8) + distance(p3, p7)
    horizontal = 2.0 * distance(p1, p5)
    return vertical / horizontal if horizontal else 0.0


def estimate_head_pose(landmarks: np.ndarray, width: int, height: int) -> tuple[float, float, float]:
    image_points = np.array([landmarks[idx] for idx in POSE_LANDMARK_IDS], dtype=np.float64)
    focal_length = float(width)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        FACE_3D_MODEL,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0.0

    yaw_deg, pitch_deg, roll_deg = map(math.degrees, (yaw, pitch, roll))
    return (
        normalize_pose_angle(yaw_deg),
        normalize_pose_angle(pitch_deg),
        normalize_pose_angle(roll_deg),
    )


def estimate_drowsy_prob(ear: float, mar: float, head_pitch: float) -> float:
    normalized_pitch = abs(normalize_pose_angle(head_pitch))
    eye_term = max(0.0, 0.23 - ear) * 14.0
    pose_term = max(0.0, normalized_pitch - 18.0) / 12.0
    mouth_term = max(0.0, mar - 0.62) * 1.5
    logit = eye_term + 0.6 * pose_term + 0.2 * mouth_term - 1.4
    return float(1.0 / (1.0 + math.exp(-logit)))


def detect_face_bbox_fallback(frame: np.ndarray, frontal: cv2.CascadeClassifier, profile: cv2.CascadeClassifier):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    if len(faces):
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        return (int(x), int(y), int(w), int(h)), False

    profiles = profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    if len(profiles):
        x, y, w, h = max(profiles, key=lambda item: item[2] * item[3])
        return (int(x), int(y), int(w), int(h)), True

    flipped = cv2.flip(gray, 1)
    profiles = profile.detectMultiScale(flipped, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    if len(profiles):
        x, y, w, h = max(profiles, key=lambda item: item[2] * item[3])
        x = gray.shape[1] - x - w
        return (int(x), int(y), int(w), int(h)), True

    return None, None


def approximate_mouth_aspect_ratio(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x, y, w, h = bbox
    mouth_y0 = y + int(h * 0.52)
    mouth_y1 = y + int(h * 0.88)
    mouth_x0 = x + int(w * 0.18)
    mouth_x1 = x + int(w * 0.82)
    roi = frame[mouth_y0:mouth_y1, mouth_x0:mouth_x1]
    if roi.size == 0:
        return 0.2
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    dark_ratio = float((gray < 55).mean())
    return float(0.18 + dark_ratio * 1.2)


def iter_video_paths(input_dir: Path) -> list[Path]:
    suffixes = {".mp4", ".avi", ".mov", ".mkv"}
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() in suffixes)


def extract_video_features(video_path: Path, output_path: Path, frame_stride: int = 1, model_path: Path | None = None) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    model_path = ensure_face_landmarker_model(model_path or Path("models/mediapipe/face_landmarker.task"))
    face_landmarker = create_landmarker(model_path)
    frontal_cascade, profile_cascade = create_fallback_cascades()

    rows = []
    frame_id = 0
    sampled_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame_id % frame_stride != 0:
            frame_id += 1
            continue

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = face_landmarker.detect_for_video(mp_image, int((frame_id / fps) * 1000.0))
        timestamp = frame_id / fps

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            points = np.array([[point.x * width, point.y * height] for point in landmarks], dtype=np.float32)

            left_ear = eye_aspect_ratio(points, LEFT_EYE)
            right_ear = eye_aspect_ratio(points, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(points)
            head_yaw, head_pitch, head_roll = estimate_head_pose(points, width, height)
            drowsy_prob = estimate_drowsy_prob(ear, mar, head_pitch)
            face_valid = 1
        else:
            bbox, is_profile = detect_face_bbox_fallback(frame, frontal_cascade, profile_cascade)
            if bbox is not None:
                ear = 0.25
                mar = approximate_mouth_aspect_ratio(frame, bbox)
                head_yaw = 25.0 if is_profile else 0.0
                head_pitch = 0.0
                head_roll = 0.0
                drowsy_prob = estimate_drowsy_prob(ear, mar, head_pitch)
                face_valid = 1
            else:
                ear = 0.0
                mar = 0.0
                head_yaw = 0.0
                head_pitch = 0.0
                head_roll = 0.0
                drowsy_prob = 0.0
                face_valid = 0

        rows.append(
            {
                "frame_id": frame_id,
                "timestamp": round(timestamp, 4),
                "ear": round(float(ear), 6),
                "mar": round(float(mar), 6),
                "head_yaw": round(float(head_yaw), 6),
                "head_pitch": round(float(head_pitch), 6),
                "head_roll": round(float(head_roll), 6),
                "drowsy_prob": round(float(drowsy_prob), 6),
                "face_valid": face_valid,
            }
        )

        frame_id += 1
        sampled_index += 1

    face_landmarker.close()
    capture.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {sampled_index} sampled frames to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract fatigue features from one video or a directory.")
    parser.add_argument("--video", help="Single video path.")
    parser.add_argument("--input-dir", help="Directory with videos.")
    parser.add_argument("--output-dir", required=True, help="Directory for output CSV files.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Sample every N frames.")
    parser.add_argument("--model-path", help="Optional local MediaPipe face landmarker model path.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        video_path = Path(args.video)
        extract_video_features(
            video_path,
            output_dir / f"{video_path.stem}.csv",
            args.frame_stride,
            Path(args.model_path) if args.model_path else None,
        )
        return

    if args.input_dir:
        for video_path in iter_video_paths(Path(args.input_dir)):
            extract_video_features(
                video_path,
                output_dir / f"{video_path.stem}.csv",
                args.frame_stride,
                Path(args.model_path) if args.model_path else None,
            )
        return

    raise ValueError("Pass either --video or --input-dir.")


if __name__ == "__main__":
    main()
