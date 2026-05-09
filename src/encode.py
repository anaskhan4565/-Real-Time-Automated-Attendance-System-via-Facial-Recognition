"""Batch face embedding extraction from preprocessed crops.

Usage:
    python -m src.encode
    python -m src.encode --backend facenet_pytorch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode processed faces into embeddings.")
    parser.add_argument(
        "--backend",
        choices=["facenet_pytorch"],
        default=None,
        help="Override encoder backend from src.config.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        if header != ["filepath", "label", "is_augmented"]:
            raise ValueError(
                "Manifest header must be filepath,label,is_augmented"
            )
        for line in f:
            line = line.strip()
            if not line:
                continue
            filepath, label, is_augmented = line.split(",", maxsplit=2)
            rows.append(
                {
                    "filepath": filepath,
                    "label": label,
                    "is_augmented": is_augmented,
                }
            )
    return rows


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / max(norm, eps)


def load_facenet_model():
    import torch
    from facenet_pytorch import InceptionResnetV1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return model, device, torch


def encode_with_facenet(
    rgb: np.ndarray,
    model,
    device,
    torch_module,
) -> np.ndarray:
    if rgb.shape[0] != config.IMG_SIZE or rgb.shape[1] != config.IMG_SIZE:
        rgb = cv2.resize(rgb, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    x = rgb.astype(np.float32)
    x = (x - 127.5) / 128.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    tensor = torch_module.from_numpy(x).to(device)
    with torch_module.no_grad():
        out = model(tensor).detach().cpu().numpy()[0]
    return out.astype(np.float32)


def build_label_map() -> tuple[LabelEncoder, dict[str, dict[str, str]]]:
    labels_in_order = [label for label, _, _ in config.PEOPLE]
    display_map = {label: name for label, name, _ in config.PEOPLE}
    roll_map = {label: roll for label, _, roll in config.PEOPLE}

    le = LabelEncoder()
    le.fit(labels_in_order)

    # Ensure deterministic id mapping follows config.PEOPLE order.
    label_to_id = {label: idx for idx, label in enumerate(labels_in_order)}
    id_to_label = {str(idx): label for label, idx in label_to_id.items()}
    id_to_display = {str(label_to_id[label]): display_map[label] for label in labels_in_order}
    id_to_roll = {str(label_to_id[label]): roll_map[label] for label in labels_in_order}

    label_map = {
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "id_to_display": id_to_display,
        "id_to_roll": id_to_roll,
    }
    return le, label_map


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    config.ensure_dirs()

    manifest_path = config.PATHS["processed"] / "_manifest.csv"
    rows = read_manifest(manifest_path)
    if not rows:
        raise RuntimeError("Manifest is empty; run preprocessing first.")

    backend = args.backend or config.ENCODER_BACKEND
    if backend != "facenet_pytorch":
        raise ValueError(f"Unsupported backend: {backend}")

    labels_order = [label for label, _, _ in config.PEOPLE]
    label_to_idx = {label: idx for idx, label in enumerate(labels_order)}
    for row in rows:
        if row["label"] not in label_to_idx:
            raise ValueError(f"Unknown label in manifest: {row['label']}")

    model, device, torch_module = load_facenet_model()
    encoder_version = "facenet_pytorch_inceptionresnetv1_vggface2"

    embeddings: list[np.ndarray] = []
    y_list: list[int] = []
    per_class_counts = {label: 0 for label in labels_order}

    for row in rows:
        img_path = Path(row["filepath"])
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        emb = encode_with_facenet(rgb, model, device, torch_module)
        emb = l2_normalize(emb).astype(np.float32)
        embeddings.append(emb)
        y_list.append(label_to_idx[row["label"]])
        per_class_counts[row["label"]] += 1

    x = np.vstack(embeddings).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    embed_dim_expected = config.EMBED_DIM
    if x.shape[1] != embed_dim_expected:
        raise RuntimeError(
            f"Embedding dimension mismatch: got {x.shape[1]}, expected {embed_dim_expected}"
        )

    norms = np.linalg.norm(x, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise RuntimeError("L2 normalization check failed: not all rows have norm ~= 1.")

    _, label_map = build_label_map()
    for label in labels_order:
        round_trip = label_map["id_to_label"][str(label_map["label_to_id"][label])]
        if round_trip != label:
            raise RuntimeError(f"Label map round-trip failed for label: {label}")

    np.save(config.PATHS["encodings"] / "X.npy", x)
    np.save(config.PATHS["encodings"] / "y.npy", y)
    save_json(config.PATHS["encodings"] / "label_map.json", label_map)

    meta = {
        "backend": backend,
        "embed_dim": int(x.shape[1]),
        "n_samples": int(x.shape[0]),
        "per_class_counts": per_class_counts,
        "encoder_version": encoder_version,
    }
    save_json(config.PATHS["encodings"] / "meta.json", meta)

    print(f"Encoded {x.shape[0]} samples → X.shape=({x.shape[0]},{x.shape[1]}) using {backend}")


if __name__ == "__main__":
    main()
