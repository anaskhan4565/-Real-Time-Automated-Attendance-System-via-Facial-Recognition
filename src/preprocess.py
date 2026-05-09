"""Preprocess raw face images: detect, align, resize, and augment.

Usage:
    python -m src.preprocess
    python -m src.preprocess --no-augment
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

from src import config
from src.face import build_mtcnn, detect_faces
from src.utils import align_face


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw face images into aligned 160x160 crops."
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable augmentation and only save aligned base crops.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)




def augment_image(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply randomized augmentation to aligned RGB image."""
    aug = rgb.copy()

    if rng.random() < 0.5:
        aug = np.ascontiguousarray(np.fliplr(aug))

    angle = float(rng.uniform(-15.0, 15.0))
    h, w = aug.shape[:2]
    center = (w * 0.5, h * 0.5)
    rot_mtx = cv2.getRotationMatrix2D(center, angle, 1.0)
    aug = cv2.warpAffine(
        aug,
        rot_mtx,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    brightness = float(rng.uniform(0.8, 1.2))
    aug_f = np.clip(aug.astype(np.float32) / 255.0, 0.0, 1.0)
    aug_f = np.clip(aug_f * brightness, 0.0, 1.0)

    noise = rng.normal(0.0, 0.01, size=aug_f.shape).astype(np.float32)
    aug_f = np.clip(aug_f + noise, 0.0, 1.0)

    return (aug_f * 255.0).round().astype(np.uint8)


def collect_raw_files(label: str) -> list[Path]:
    label_dir = config.PATHS["raw"] / label
    if not label_dir.exists():
        return []
    return sorted(label_dir.glob("*.jpg"))


def process_label(
    label: str,
    do_augment: bool,
    rng: np.random.Generator,
) -> tuple[dict[str, int], list[tuple[str, str, bool]]]:
    """Process one class and return summary + manifest rows."""
    raw_files = collect_raw_files(label)
    out_dir = config.PATHS["processed"] / label
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"raw": len(raw_files), "accepted": 0, "augmented": 0, "total": 0}
    manifest_rows: list[tuple[str, str, bool]] = []
    mtcnn = build_mtcnn()

    # Calculate dynamic augmentation count if raw files < 80
    if len(raw_files) < 80 and len(raw_files) > 0:
        aug_per_image = int(320 / len(raw_files)) - 1
        aug_per_image = max(1, aug_per_image)  # Ensure at least 1 augmentation
    else:
        aug_per_image = config.AUG_PER_IMAGE

    for raw_path in raw_files:
        bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        boxes, landmarks_list = detect_faces(mtcnn, rgb)
        if len(boxes) != 1:
            continue

        try:
            aligned = align_face(rgb, boxes[0], landmarks_list[0], config.IMG_SIZE)
        except Exception:
            continue

        aligned = np.clip(aligned, 0, 255).astype(np.uint8)
        base_name = raw_path.stem
        base_out = out_dir / f"{base_name}.png"
        cv2.imwrite(str(base_out), cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
        summary["accepted"] += 1
        summary["total"] += 1
        manifest_rows.append((base_out.as_posix(), label, False))

        if do_augment:
            for aug_idx in range(1, aug_per_image + 1):
                aug_img = augment_image(aligned, rng)
                aug_out = out_dir / f"{base_name}_aug{aug_idx}.png"
                cv2.imwrite(str(aug_out), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                summary["augmented"] += 1
                summary["total"] += 1
                manifest_rows.append((aug_out.as_posix(), label, True))

    return summary, manifest_rows


def write_manifest(rows: list[tuple[str, str, bool]]) -> None:
    manifest_path = config.PATHS["processed"] / "_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filepath", "label", "is_augmented"])
        for row in rows:
            writer.writerow([row[0], row[1], str(row[2]).lower()])


def print_summary(summaries: dict[str, dict[str, int]]) -> None:
    headers = ["label", "raw", "accepted", "augmented", "total"]
    print(" | ".join(headers))
    print("-" * 64)
    for label, stats in summaries.items():
        print(
            f"{label} | {stats['raw']} | {stats['accepted']} | "
            f"{stats['augmented']} | {stats['total']}"
        )


def main() -> None:
    args = parse_args()
    seed_everything(config.SEED)
    config.ensure_dirs()

    rng = np.random.default_rng(config.SEED)
    do_augment = not args.no_augment

    summaries: dict[str, dict[str, int]] = {}
    all_rows: list[tuple[str, str, bool]] = []

    for label, _, _ in config.PEOPLE:
        stats, rows = process_label(label=label, do_augment=do_augment, rng=rng)
        summaries[label] = stats
        all_rows.extend(rows)

    write_manifest(all_rows)
    print_summary(summaries)


if __name__ == "__main__":
    main()
