"""Generate SmartPresence report figures using only matplotlib."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src import config


def _draw_rounded_box(ax, xy, wh, text, fontsize=9, fc="#f5f5f5", ec="black") -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _draw_arrow(ax, p1, p2) -> None:
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        color="black",
    )
    ax.add_patch(arr)


def make_pipeline_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    labels = [
        "Webcam",
        "HOG Face Detection",
        "5-pt Alignment\n+ Resize 160",
        "Pretrained Encoder\n(ResNet-34, 128-d)",
        "L2-normalize",
        "Trained SVM/MLP\nHead",
        "tau-threshold",
        "Draw label\n+ CSV log",
    ]

    n = len(labels)
    box_w = 0.105
    box_h = 0.36
    y = 0.32
    x0 = 0.02
    gap = (0.98 - x0 - n * box_w) / (n - 1)

    boxes = []
    for i, label in enumerate(labels):
        x = x0 + i * (box_w + gap)
        boxes.append((x, y, box_w, box_h))
        _draw_rounded_box(ax, (x, y), (box_w, box_h), label, fontsize=9)

    for i in range(n - 1):
        x1, y1, w1, h1 = boxes[i]
        x2, y2, w2, h2 = boxes[i + 1]
        _draw_arrow(ax, (x1 + w1, y1 + h1 / 2), (x2, y2 + h2 / 2))

    ax.set_title("SmartPresence Pipeline", fontsize=16, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_architecture_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blocks = [
        ("Input 160x160x3", "~0"),
        ("Conv7x7-32 / s=2", "~4.7K"),
        ("Residual Stage 1\n(32 ch, depth 3)", "~0.20M"),
        ("Residual Stage 2\n(64 ch, depth 4)", "~0.80M"),
        ("Residual Stage 3\n(128 ch, depth 6)", "~4.5M"),
        ("Residual Stage 4\n(256 ch, depth 3)", "~16.0M"),
        ("Global Avg Pool", "0"),
        ("FC -> 128-d Embedding", "~0.26M"),
        ("L2 Normalize", "0"),
        ("Classifier Head\nSVM (RBF, C, gamma)\nOR MLP (hidden=128, ReLU)", "SVM: data-dependent\nMLP: ~16.8K"),
        ("Softmax over K=3", "~387"),
        ("tau-threshold decision\n(Unknown if max_p < tau)", "0"),
    ]

    box_w = 0.72
    box_h = 0.055
    x = 0.14
    y_top = 0.94
    vgap = 0.02
    coords = []

    for i, (name, params) in enumerate(blocks):
        y = y_top - i * (box_h + vgap) - box_h
        text = f"{name}\nparams: {params}"
        fc = "#f2f7ff" if i <= 8 else "#f8fff2"
        _draw_rounded_box(ax, (x, y), (box_w, box_h), text, fontsize=8, fc=fc)
        coords.append((x, y, box_w, box_h))

    for i in range(len(coords) - 1):
        x1, y1, w1, h1 = coords[i]
        x2, y2, w2, h2 = coords[i + 1]
        _draw_arrow(ax, (x1 + w1 / 2, y1), (x2 + w2 / 2, y2 + h2))

    # Dashed separator under frozen encoder portion.
    sep_idx = 8
    _, y_sep, _, _ = coords[sep_idx]
    y_line = y_sep - 0.01
    ax.plot([0.08, 0.92], [y_line, y_line], linestyle="--", linewidth=1.4, color="black")
    ax.text(
        0.5,
        y_line - 0.02,
        "Frozen above this line",
        ha="center",
        va="top",
        fontsize=9,
    )

    ax.set_title("SmartPresence Architecture", fontsize=15, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _augment_hflip(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.fliplr(img))


def _augment_rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _augment_brightness(img: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def _augment_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0
    noisy = img_f + rng.normal(0.0, sigma, size=img_f.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, 1.0)
    return (noisy * 255.0).astype(np.uint8)


def _random_combo(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    if rng.random() < 0.5:
        out = _augment_hflip(out)
    out = _augment_rotate(out, float(rng.uniform(-15, 15)))
    out = _augment_brightness(out, float(rng.uniform(0.8, 1.2)))
    out = _augment_noise(out, 0.01, rng)
    return out


def _pick_one_aligned_face() -> np.ndarray:
    all_pngs = sorted((config.PATHS["processed"]).glob("*/*.png"))
    if not all_pngs:
        raise FileNotFoundError(
            "No aligned face images found in data/processed/<label>/*.png. Run preprocessing first."
        )
    rng = np.random.default_rng(config.SEED)
    pick = all_pngs[int(rng.integers(0, len(all_pngs)))]
    bgr = cv2.imread(str(pick), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {pick}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def make_augmentation_grid(out_path: Path) -> None:
    base = _pick_one_aligned_face()
    rng = np.random.default_rng(config.SEED)

    images = []
    captions = []

    # Row 1
    images.extend(
        [
            base,
            _augment_hflip(base),
            _augment_rotate(base, -15.0),
            _augment_rotate(base, 15.0),
        ]
    )
    captions.extend(["original", "hflip", "rotate -15deg", "rotate +15deg"])

    # Row 2
    images.extend(
        [
            _augment_brightness(base, 0.8),
            _augment_brightness(base, 1.2),
            _augment_noise(base, 0.01, rng),
            _augment_rotate(_augment_hflip(base), 10.0),
        ]
    )
    captions.extend(["brightness 0.8", "brightness 1.2", "noise sigma=0.01", "hflip+rotate"])

    # Rows 3-4: 8 random combos
    for i in range(8):
        images.append(_random_combo(base, rng))
        captions.append(f"random combo {i + 1}")

    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i])
        ax.set_title(captions[i], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    config.ensure_dirs()
    out_dir = config.PATHS["figures"]
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = out_dir / "pipeline.png"
    architecture_path = out_dir / "architecture.png"
    aug_grid_path = out_dir / "augmentation_grid.png"

    make_pipeline_figure(pipeline_path)
    make_architecture_figure(architecture_path)
    make_augmentation_grid(aug_grid_path)

    print(str(pipeline_path))
    print(str(architecture_path))
    print(str(aug_grid_path))


if __name__ == "__main__":
    main()
