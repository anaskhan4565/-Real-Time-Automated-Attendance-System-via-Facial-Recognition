"""Face detection and landmark helpers using facenet-pytorch's MTCNN.

This module uses MTCNN-based detector that returns top/right/bottom/left boxes and 5-point
landmarks suitable for alignment and embedding.
"""

from __future__ import annotations

import numpy as np
import torch
from facenet_pytorch import MTCNN


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_mtcnn(device: torch.device | None = None) -> MTCNN:
    return MTCNN(
        image_size=None,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        keep_all=True,
        device=device or get_device(),
        post_process=False,
    )


def mtcnn_box_to_face_recognition(box: np.ndarray) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.astype(np.int32).tolist()
    return (y1, x2, y2, x1)


def detect_faces(mtcnn: MTCNN, rgb: np.ndarray) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]]:
    boxes, _, landmarks = mtcnn.detect(rgb, landmarks=True)
    if boxes is None:
        return [], []
    face_boxes = [mtcnn_box_to_face_recognition(box) for box in boxes]
    face_landmarks = [np.asarray(point, dtype=np.float32) for point in landmarks]
    return face_boxes, face_landmarks
