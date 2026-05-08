"""Single source of truth for paths, hyperparameters, and class metadata.

Every script in `src/` and `scripts/` reads from this module. Do NOT hard-code
paths or hyperparameters anywhere else.
"""

from __future__ import annotations

from pathlib import Path

SEED: int = 42

PEOPLE: list[tuple[str, str, str]] = [
    ("ahsan_22K-4176", "Ahsan Ali", "22K-4176"),
    ("anas_m_22K-4548", "Mohammad Anas", "22K-4548"),
    ("anas_k_22K-4483", "Anas Khan", "22K-4483"),
]

FRAMES_PER_PERSON: int = 80
IMG_SIZE: int = 160

ENCODER_BACKEND: str = "face_recognition"
EMBED_DIM: int = 128

TAU: float = 0.55

TRAIN_VAL_TEST: tuple[float, float, float] = (0.8, 0.1, 0.1)
CV_FOLDS: int = 5

AUG_PER_IMAGE: int = 3

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

PATHS: dict[str, Path] = {
    "raw": PROJECT_ROOT / "data" / "raw",
    "processed": PROJECT_ROOT / "data" / "processed",
    "encodings": PROJECT_ROOT / "data" / "encodings",
    "models": PROJECT_ROOT / "models",
    "logs": PROJECT_ROOT / "logs",
    "figures": PROJECT_ROOT / "report" / "figures",
    "attendance_csv": PROJECT_ROOT / "logs" / "attendance.csv",
}


def ensure_dirs() -> None:
    """Create every directory in PATHS if it does not exist (idempotent)."""
    for key, p in PATHS.items():
        target = p if key != "attendance_csv" else p.parent
        target.mkdir(parents=True, exist_ok=True)
