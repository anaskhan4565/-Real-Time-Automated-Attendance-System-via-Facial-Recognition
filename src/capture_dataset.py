"""Webcam-driven dataset capture for SmartPresence.

Usage:
    python -m src.capture_dataset --person ahsan_22K-4176
    python -m src.capture_dataset --person anas_m_22K-4548 --frames 100
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np

from src import config
from src.face import build_mtcnn, detect_faces

INSTRUCTIONS = [
    "Look straight",
    "Slight left",
    "Slight right",
    "Tilt up",
    "Tilt down",
    "Smile",
    "Neutral",
    "Move closer",
    "Move back",
]

FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Capture labeled webcam frames for one person."
    )
    parser.add_argument(
        "--person",
        required=True,
        help="Person label (must exist in src.config.PEOPLE).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=config.FRAMES_PER_PERSON,
        help="Number of valid frames to capture.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Seed RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def person_lookup(label: str) -> tuple[str, str]:
    """Return (human_name, roll_no) for a valid label."""
    mapping = {p_label: (name, roll) for p_label, name, roll in config.PEOPLE}
    if label not in mapping:
        valid = ", ".join(mapping.keys())
        raise ValueError(f"Unknown label '{label}'. Valid labels: {valid}")
    return mapping[label]


def next_index(person_dir: Path, label: str) -> int:
    """Find the next append-safe image index."""
    existing = sorted(person_dir.glob(f"{label}_*.jpg"))
    if not existing:
        return 1
    max_idx = 0
    for file_path in existing:
        stem = file_path.stem
        # Expected: <label>_<4-digit-index>
        idx_str = stem.rsplit("_", 1)[-1]
        if idx_str.isdigit():
            max_idx = max(max_idx, int(idx_str))
    return max_idx + 1


def draw_countdown(frame: np.ndarray, value: int) -> np.ndarray:
    """Draw a large centered countdown value on frame copy."""
    output = frame.copy()
    h, w = output.shape[:2]
    text = str(value)
    (tw, th), _ = cv2.getTextSize(text, FONT, 4.0, 6)
    cv2.putText(
        output,
        text,
        ((w - tw) // 2, (h + th) // 2),
        FONT,
        4.0,
        (0, 255, 255),
        6,
        cv2.LINE_AA,
    )
    return output


def draw_hud(
    frame: np.ndarray,
    person_name: str,
    roll: str,
    saved_count: int,
    target_frames: int,
    instruction: str,
) -> np.ndarray:
    """Overlay capture info and instruction text."""
    output = frame.copy()
    top_text = f"Capturing: {person_name} ({roll})  frame {saved_count} / {target_frames}"
    bottom_text = f"Instruction: {instruction}   (Press q to quit)"

    cv2.putText(output, top_text, (12, 28), FONT, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    h, _ = output.shape[:2]
    cv2.putText(
        output,
        bottom_text,
        (12, h - 18),
        FONT,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def is_single_large_face(frame_bgr: np.ndarray, mtcnn) -> bool:
    """Return True only when exactly one face exists and area >= 10000 px."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes, _ = detect_faces(mtcnn, rgb)
    if len(boxes) != 1:
        return False
    top, right, bottom, left = boxes[0]
    area = max(0, right - left) * max(0, bottom - top)
    return area >= 10000


def run_capture(label: str, target_frames: int) -> int:
    """Capture and save valid frames. Returns number of saved frames."""
    if target_frames <= 0:
        raise ValueError("--frames must be a positive integer.")

    config.ensure_dirs()
    person_name, roll = person_lookup(label)
    person_dir = config.PATHS["raw"] / label
    person_dir.mkdir(parents=True, exist_ok=True)

    save_idx = next_index(person_dir, label)
    saved_count = 0
    display_count = 0
    mtcnn = build_mtcnn()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open default webcam (index 0).")

    try:
        for count in (3, 2, 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame during countdown.")
            preview = draw_countdown(frame, count)
            cv2.imshow("SmartPresence Capture", preview)
            key = cv2.waitKey(1000) & 0xFF
            if key == ord("q"):
                return saved_count

        while saved_count < target_frames:
            ok, frame = cap.read()
            if not ok:
                break

            instruction = INSTRUCTIONS[(saved_count // 15) % len(INSTRUCTIONS)]
            preview = draw_hud(frame, person_name, roll, saved_count, target_frames, instruction)
            cv2.imshow("SmartPresence Capture", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if is_single_large_face(frame, mtcnn):
                filename = f"{label}_{save_idx:04d}.jpg"
                out_path = person_dir / filename
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_count += 1
                save_idx += 1
                display_count += 1

            # Gentle pacing for stability and user repositioning.
            if display_count % 5 == 0:
                time.sleep(0.03)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return saved_count


def main() -> None:
    args = parse_args()
    seed_everything(config.SEED)
    saved = run_capture(label=args.person, target_frames=args.frames)
    target_dir = config.PATHS["raw"] / args.person
    print(f"Saved {saved} frames to {target_dir.as_posix()}")


if __name__ == "__main__":
    main()
