"""Shared helpers for SmartPresence."""

from __future__ import annotations

import csv
import os
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.29, 51.69],  # left eye
        [73.53, 51.50],  # right eye
        [56.02, 71.74],  # nose tip
        [41.54, 92.36],  # left mouth corner
        [70.72, 92.20],  # right mouth corner
    ],
    dtype=np.float32,
)


def seed_everything(seed: int) -> None:
    """Seed reproducibility sources used by this project."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch is optional for some project stages.
        pass


def _compute_five_points(landmarks_dict: dict[str, list[tuple[int, int]]]) -> np.ndarray:
    left_eye = np.asarray(landmarks_dict["left_eye"], dtype=np.float32)
    right_eye = np.asarray(landmarks_dict["right_eye"], dtype=np.float32)
    top_lip = np.asarray(landmarks_dict["top_lip"], dtype=np.float32)
    bottom_lip = np.asarray(landmarks_dict["bottom_lip"], dtype=np.float32)
    nose_bridge = np.asarray(landmarks_dict["nose_bridge"], dtype=np.float32)
    nose_tip = np.asarray(landmarks_dict["nose_tip"], dtype=np.float32)

    left_eye_center = left_eye.mean(axis=0)
    right_eye_center = right_eye.mean(axis=0)
    nose = nose_tip[len(nose_tip) // 2] if len(nose_tip) > 0 else nose_bridge[-1]

    mouth_all = np.vstack([top_lip, bottom_lip])
    left_mouth = mouth_all[np.argmin(mouth_all[:, 0])]
    right_mouth = mouth_all[np.argmax(mouth_all[:, 0])]

    return np.vstack([left_eye_center, right_eye_center, nose, left_mouth, right_mouth]).astype(
        np.float32
    )


def align_face(
    rgb_image: np.ndarray,
    face_box: tuple[int, int, int, int],
    landmarks_dict: dict[str, list[tuple[int, int]]],
    out_size: int,
) -> np.ndarray:
    """Align face to ArcFace template and return RGB float image."""
    _ = face_box  # Included to match calling contract.
    src_points = _compute_five_points(landmarks_dict)
    dst_points = ARCFACE_TEMPLATE_112 * (float(out_size) / 112.0)

    transform, _ = cv2.estimateAffinePartial2D(src_points, dst_points, method=cv2.LMEDS)
    if transform is None:
        raise RuntimeError("Could not estimate affine transform for face alignment.")

    aligned = cv2.warpAffine(
        rgb_image,
        transform,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned.astype(np.float32)


class FPSMeter:
    """Exponential-moving-average FPS tracker."""

    def __init__(self, alpha: float = 0.9) -> None:
        self.alpha = alpha
        self._last_time: float | None = None
        self._fps: float = 0.0

    @property
    def fps(self) -> float:
        return self._fps

    def update(self) -> float:
        now = time.perf_counter()
        if self._last_time is None:
            self._last_time = now
            return self._fps

        dt = max(now - self._last_time, 1e-9)
        inst_fps = 1.0 / dt
        if self._fps == 0.0:
            self._fps = inst_fps
        else:
            self._fps = self.alpha * self._fps + (1.0 - self.alpha) * inst_fps
        self._last_time = now
        return self._fps


def draw_label(
    bgr_image: np.ndarray,
    box: tuple[int, int, int, int],
    text: str,
    color: tuple[int, int, int] = (0, 255, 0),
    confidence: float | None = None,
) -> None:
    """Draw face rectangle and filled caption bar in-place."""
    top, right, bottom, left = box
    h, w = bgr_image.shape[:2]

    top = int(np.clip(top, 0, h - 1))
    right = int(np.clip(right, 0, w - 1))
    bottom = int(np.clip(bottom, 0, h - 1))
    left = int(np.clip(left, 0, w - 1))

    cv2.rectangle(bgr_image, (left, top), (right, bottom), color, 2)

    label = text if confidence is None else f"{text}  conf={confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    bar_top = max(0, top - th - baseline - 8)
    bar_bottom = max(th + baseline + 8, top)
    bar_right = min(w - 1, left + tw + 10)
    cv2.rectangle(bgr_image, (left, bar_top), (bar_right, bar_bottom), color, -1)
    cv2.putText(
        bgr_image,
        label,
        (left + 5, bar_bottom - baseline - 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


class CSVAttendanceLogger:
    """CSV logger with same-day deduplication by (name, date)."""

    HEADER = ["name", "roll_no", "date", "time", "confidence"]

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[tuple[str, str]] = set()

        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADER)

        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("name") or "").strip()
                date = (row.get("date") or "").strip()
                if name and date:
                    self._seen.add((name, date))

    def log(self, name: str, roll_no: str, confidence: float) -> bool:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        key = (name, date_str)
        if key in self._seen:
            return False

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([name, roll_no, date_str, time_str, f"{confidence:.4f}"])
        self._seen.add(key)
        return True
