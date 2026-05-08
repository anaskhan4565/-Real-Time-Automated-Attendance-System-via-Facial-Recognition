"""Generate evaluation artifacts for SmartPresence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src import config
from src.encode import (
    encode_with_face_recognition,
    encode_with_facenet,
    l2_normalize,
    load_facenet_model,
)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def split_indices(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train_ratio, val_ratio, test_ratio = config.TRAIN_VAL_TEST
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx,
        y,
        train_size=train_ratio,
        random_state=config.SEED,
        stratify=y,
    )
    val_from_temp = val_ratio / (val_ratio + test_ratio)
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp,
        y_temp,
        train_size=val_from_temp,
        random_state=config.SEED,
        stratify=y_temp,
    )
    return idx_train, idx_val, idx_test


def compute_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: list[int],
    class_names: list[str],
    out_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    ax.set_title("Confusion Matrix (Test Split)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: list[int],
    class_names: list[str],
    out_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_ids,
        zero_division=0,
    )
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, precision, width=width, label="Precision")
    ax.bar(x, recall, width=width, label="Recall")
    ax.bar(x + width, f1, width=width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Metrics (Test Split)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return precision, recall, f1


def save_tsne_embeddings(
    x: np.ndarray,
    y: np.ndarray,
    class_ids: list[int],
    class_names: list[str],
    out_path: Path,
) -> None:
    perplexity = min(20, max(5, x.shape[0] - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=config.SEED,
        init="pca",
        learning_rate="auto",
    )
    reduced = tsne.fit_transform(x)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.get_cmap("tab10", len(class_ids))
    for i, class_id in enumerate(class_ids):
        mask = y == class_id
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            s=18,
            alpha=0.8,
            color=cmap(i),
            label=class_names[i],
        )
    ax.set_title("t-SNE of Face Embeddings")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _scramble_blocks(rgb: np.ndarray, rng: np.random.Generator, blocks: int = 8) -> np.ndarray:
    h, w = rgb.shape[:2]
    bh = h // blocks
    bw = w // blocks
    tiles = []
    for r in range(blocks):
        for c in range(blocks):
            y0, y1 = r * bh, (r + 1) * bh if r < blocks - 1 else h
            x0, x1 = c * bw, (c + 1) * bw if c < blocks - 1 else w
            tiles.append(rgb[y0:y1, x0:x1].copy())
    rng.shuffle(tiles)
    out = np.zeros_like(rgb)
    i = 0
    for r in range(blocks):
        for c in range(blocks):
            y0, y1 = r * bh, (r + 1) * bh if r < blocks - 1 else h
            x0, x1 = c * bw, (c + 1) * bw if c < blocks - 1 else w
            tile = cv2.resize(tiles[i], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
            out[y0:y1, x0:x1] = tile
            i += 1
    return out


def _corrupt_image(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = rgb.shape[:2]
    crop_scale = float(rng.uniform(0.4, 0.8))
    ch = max(16, int(h * crop_scale))
    cw = max(16, int(w * crop_scale))
    y0 = int(rng.integers(0, max(1, h - ch + 1)))
    x0 = int(rng.integers(0, max(1, w - cw + 1)))
    crop = rgb[y0 : y0 + ch, x0 : x0 + cw]
    resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    scrambled = _scramble_blocks(resized, rng=rng, blocks=8)
    noisy = np.clip(
        scrambled.astype(np.float32) + rng.normal(0.0, 35.0, size=scrambled.shape),
        0.0,
        255.0,
    ).astype(np.uint8)
    return noisy


def build_impostor_embeddings(
    manifest_rows: list[dict[str, str]],
    idx_test: np.ndarray,
    x_test: np.ndarray,
    backend: str,
) -> np.ndarray:
    rng = np.random.default_rng(config.SEED)
    impostors: list[np.ndarray] = []
    target_n = 200

    model = device = torch_module = None
    if backend == "facenet_pytorch":
        model, device, torch_module = load_facenet_model()

    for _ in range(target_n):
        row = manifest_rows[int(rng.choice(idx_test))]
        bgr = cv2.imread(row["filepath"], cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        corrupted = _corrupt_image(rgb, rng)
        try:
            if backend == "face_recognition":
                emb = encode_with_face_recognition(corrupted)
            else:
                emb = encode_with_facenet(corrupted, model, device, torch_module)
            emb = l2_normalize(emb.astype(np.float32))
            impostors.append(emb)
        except Exception:
            continue

    if len(impostors) < 50:
        # Fallback required by prompt: noisy, re-normalized perturbations of test embeddings.
        fallback = []
        for _ in range(target_n):
            base = x_test[int(rng.integers(0, x_test.shape[0]))]
            noisy = base + rng.normal(0.0, 0.5, size=base.shape).astype(np.float32)
            fallback.append(l2_normalize(noisy))
        return np.vstack(fallback).astype(np.float32)

    return np.vstack(impostors).astype(np.float32)


def save_roc_unknown(
    model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    impostors: np.ndarray,
    out_path: Path,
) -> float:
    probs_test = model.predict_proba(x_test)
    probs_imp = model.predict_proba(impostors)

    pred_test = np.argmax(probs_test, axis=1)
    max_test = np.max(probs_test, axis=1)
    max_imp = np.max(probs_imp, axis=1)

    taus = np.linspace(0.0, 1.0, 101)
    tar = []
    irr = []
    for tau in taus:
        genuine_accept = (max_test >= tau) & (pred_test == y_test)
        impostor_reject = max_imp < tau
        tar.append(float(np.mean(genuine_accept)))
        irr.append(float(np.mean(impostor_reject)))

    tar_arr = np.asarray(tar)
    irr_arr = np.asarray(irr)
    score = (tar_arr + irr_arr) / 2.0
    best_idx = int(np.argmax(score))
    best_tau = float(taus[best_idx])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(taus, tar_arr, label="Genuine Acceptance Rate (TAR)")
    ax.plot(taus, irr_arr, label="Impostor Rejection Rate (IRR)")
    ax.scatter([best_tau], [score[best_idx]], color="red", s=50, zorder=3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Threshold (tau)")
    ax.set_ylabel("Rate")
    ax.set_title("Open-set Threshold Sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return best_tau


def save_sample_predictions(
    manifest_rows: list[dict[str, str]],
    idx_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    id_to_display: dict[str, str],
    out_path: Path,
) -> None:
    count = min(9, len(idx_test))
    selected_idx = idx_test[:count]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    for i in range(9):
        ax = axes[i]
        if i >= count:
            ax.axis("off")
            continue

        sample_idx = int(selected_idx[i])
        row = manifest_rows[sample_idx]
        bgr = cv2.imread(row["filepath"], cv2.IMREAD_COLOR)
        if bgr is None:
            ax.axis("off")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        true_id = int(y_test[i])
        pred_id = int(y_pred[i])
        conf = float(np.max(probs[i]))

        true_name = id_to_display.get(str(true_id), str(true_id))
        pred_name = id_to_display.get(str(pred_id), str(pred_id))
        ok = pred_id == true_id
        color = "green" if ok else "red"
        ax.set_title(
            f"Pred: {pred_name}\nTrue: {true_name}\nConf: {conf:.2f}",
            color=color,
            fontsize=9,
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_metrics_table(
    overall: dict[str, float],
    class_names: list[str],
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    out_path: Path,
) -> None:
    lines = [
        "| Scope | Class | Accuracy | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|---:|",
        (
            "| Overall | Macro | "
            f"{overall['accuracy']:.4f} | {overall['precision_macro']:.4f} | "
            f"{overall['recall_macro']:.4f} | {overall['f1_macro']:.4f} |"
        ),
    ]
    for i, name in enumerate(class_names):
        lines.append(
            f"| Per-class | {name} | - | {precision[i]:.4f} | {recall[i]:.4f} | {f1[i]:.4f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config.ensure_dirs()
    figures_dir = config.PATHS["figures"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_payload = joblib.load(config.PATHS["models"] / "classifier.pkl")
    model = model_payload["model"]
    backend = model_payload.get("backend", config.ENCODER_BACKEND)
    label_map_rel = model_payload.get("label_map_path", "data/encodings/label_map.json")

    with (config.PROJECT_ROOT / label_map_rel).open("r", encoding="utf-8") as f:
        label_map = json.load(f)

    x = np.load(config.PATHS["encodings"] / "X.npy").astype(np.float32)
    y = np.load(config.PATHS["encodings"] / "y.npy").astype(np.int64)
    manifest_rows = load_manifest(config.PATHS["processed"] / "_manifest.csv")
    if len(manifest_rows) != len(x):
        raise RuntimeError("Manifest length and embedding count mismatch.")

    _, _, idx_test = split_indices(y)
    x_test = x[idx_test]
    y_test = y[idx_test]

    probs_test = model.predict_proba(x_test)
    y_pred = np.argmax(probs_test, axis=1)

    class_ids = sorted(int(v) for v in label_map["id_to_label"].keys())
    class_names = [label_map["id_to_display"][str(i)] for i in class_ids]

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_ids=class_ids,
        class_names=class_names,
        out_path=figures_dir / "confusion_matrix.png",
    )
    precision, recall, f1 = save_per_class_metrics(
        y_true=y_test,
        y_pred=y_pred,
        class_ids=class_ids,
        class_names=class_names,
        out_path=figures_dir / "per_class_metrics.png",
    )
    save_tsne_embeddings(
        x=x,
        y=y,
        class_ids=class_ids,
        class_names=class_names,
        out_path=figures_dir / "tsne_embeddings.png",
    )

    impostors = build_impostor_embeddings(
        manifest_rows=manifest_rows,
        idx_test=idx_test,
        x_test=x_test,
        backend=backend,
    )
    best_tau = save_roc_unknown(
        model=model,
        x_test=x_test,
        y_test=y_test,
        impostors=impostors,
        out_path=figures_dir / "roc_unknown.png",
    )
    save_sample_predictions(
        manifest_rows=manifest_rows,
        idx_test=idx_test,
        y_test=y_test,
        y_pred=y_pred,
        probs=probs_test,
        id_to_display=label_map["id_to_display"],
        out_path=figures_dir / "sample_predictions.png",
    )

    overall = compute_overall_metrics(y_test, y_pred)
    save_metrics_table(
        overall=overall,
        class_names=class_names,
        precision=precision,
        recall=recall,
        f1=f1,
        out_path=figures_dir / "metrics_table.md",
    )

    print(f"Recommended TAU = {best_tau:.2f}")


if __name__ == "__main__":
    main()
