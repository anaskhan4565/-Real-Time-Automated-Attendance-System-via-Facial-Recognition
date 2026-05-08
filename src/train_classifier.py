"""Train and select classifier heads on top of face embeddings.

Usage:
    python -m src.train_classifier
    python -m src.train_classifier --quick
    python -m src.train_classifier --no-mlp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from src import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SVM/MLP classifier heads.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip MLP grid-search for faster runs.",
    )
    parser.add_argument(
        "--no-mlp",
        action="store_true",
        help="Disable MLP family.",
    )
    return parser.parse_args()


def load_embeddings() -> tuple[np.ndarray, np.ndarray]:
    x_path = config.PATHS["encodings"] / "X.npy"
    y_path = config.PATHS["encodings"] / "y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("Missing X.npy or y.npy; run src.encode first.")
    x = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError(f"Unexpected shapes: X={x.shape}, y={y.shape}")
    return x, y


def split_data(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_ratio, val_ratio, test_ratio = config.TRAIN_VAL_TEST
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("TRAIN_VAL_TEST must sum to 1.0")

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        train_size=train_ratio,
        random_state=config.SEED,
        stratify=y,
    )
    val_from_temp = val_ratio / (val_ratio + test_ratio)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        train_size=val_from_temp,
        random_state=config.SEED,
        stratify=y_temp,
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


def metrics_for_split(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = sorted(np.unique(y_true).tolist())
    per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_f1 = {str(label): float(score) for label, score in zip(labels, per_class)}

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": per_class_f1,
    }


def to_serializable_param_dict(params: dict) -> dict:
    serializable = {}
    for k, v in params.items():
        if isinstance(v, tuple):
            serializable[k] = list(v)
        else:
            serializable[k] = v
    return serializable


def run_family_grid(
    name: str,
    pipeline: Pipeline,
    param_grid: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[Pipeline, dict]:
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=config.CV_FOLDS,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(x_train, y_train)

    cv_results = grid.cv_results_
    candidates = []
    for idx, params in enumerate(cv_results["params"]):
        candidates.append(
            {
                "params": to_serializable_param_dict(params),
                "mean_test_score": float(cv_results["mean_test_score"][idx]),
                "std_test_score": float(cv_results["std_test_score"][idx]),
                "rank_test_score": int(cv_results["rank_test_score"][idx]),
            }
        )

    family_result = {
        "best_params": to_serializable_param_dict(grid.best_params_),
        "best_cv_score": float(grid.best_score_),
        "candidates": candidates,
    }
    return grid.best_estimator_, family_result


def format_markdown_table(rows: list[tuple[str, float, float]]) -> str:
    lines = [
        "| Family | Val F1 (macro) | Test F1 (macro) |",
        "|---|---:|---:|",
    ]
    for family, val_f1, test_f1 in rows:
        lines.append(f"| {family} | {val_f1:.4f} | {test_f1:.4f} |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config.ensure_dirs()

    x, y = load_embeddings()
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x, y)

    family_specs = []

    linear_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "estimator",
                CalibratedClassifierCV(
                    estimator=LinearSVC(random_state=config.SEED, max_iter=10000),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )
    linear_grid = {"estimator__estimator__C": [0.1, 1, 10, 100]}
    family_specs.append(("linear_svc_calibrated", linear_pipeline, linear_grid))

    rbf_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "estimator",
                SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=config.SEED,
                ),
            ),
        ]
    )
    rbf_grid = {"estimator__C": [0.1, 1, 10, 100], "estimator__gamma": ["scale", 0.01, 0.1]}
    family_specs.append(("rbf_svc", rbf_pipeline, rbf_grid))

    use_mlp = not (args.no_mlp or args.quick)
    if use_mlp:
        mlp_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "estimator",
                    MLPClassifier(
                        max_iter=200,
                        early_stopping=True,
                        random_state=config.SEED,
                    ),
                ),
            ]
        )
        mlp_grid = {
            "estimator__hidden_layer_sizes": [(64,), (128,)],
            "estimator__learning_rate_init": [1e-3, 1e-4],
        }
        family_specs.append(("mlp", mlp_pipeline, mlp_grid))

    family_results: dict[str, dict] = {}
    best_estimators: dict[str, Pipeline] = {}
    comparison_rows: list[tuple[str, float, float]] = []
    best_family = None
    best_val_f1 = -1.0

    for family_name, pipeline, param_grid in family_specs:
        estimator, cv_payload = run_family_grid(
            name=family_name,
            pipeline=pipeline,
            param_grid=param_grid,
            x_train=x_train,
            y_train=y_train,
        )

        y_val_pred = estimator.predict(x_val)
        y_test_pred = estimator.predict(x_test)
        val_metrics = metrics_for_split(y_val, y_val_pred)
        test_metrics = metrics_for_split(y_test, y_test_pred)

        cv_payload["val_metrics"] = val_metrics
        cv_payload["test_metrics"] = test_metrics
        family_results[family_name] = cv_payload
        best_estimators[family_name] = estimator

        val_f1 = val_metrics["f1_macro"]
        test_f1 = test_metrics["f1_macro"]
        comparison_rows.append((family_name, val_f1, test_f1))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_family = family_name

    if best_family is None:
        raise RuntimeError("No classifier family was trained.")

    chosen_estimator = best_estimators[best_family]
    chosen_val_pred = chosen_estimator.predict(x_val)
    chosen_test_pred = chosen_estimator.predict(x_test)
    metrics_payload = {
        "selected_family": best_family,
        "val": metrics_for_split(y_val, chosen_val_pred),
        "test": metrics_for_split(y_test, chosen_test_pred),
    }

    model_payload = {
        "model": chosen_estimator,
        "backend": config.ENCODER_BACKEND,
        "label_map_path": "data/encodings/label_map.json",
    }
    joblib.dump(model_payload, config.PATHS["models"] / "classifier.pkl")

    with (config.PATHS["models"] / "grid_results.json").open("w", encoding="utf-8") as f:
        json.dump(family_results, f, indent=2)

    with (config.PATHS["models"] / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    table = format_markdown_table(comparison_rows)
    log_lines = [
        "SmartPresence training summary",
        f"Selected family (by val macro-F1): {best_family}",
        "",
        table,
        "",
        "Best params by family:",
    ]
    for family, payload in family_results.items():
        log_lines.append(f"- {family}: {payload['best_params']}")
    (config.PATHS["models"] / "training_log.txt").write_text(
        "\n".join(log_lines), encoding="utf-8"
    )

    # Acceptance-style runtime check: loaded model must support predict_proba.
    loaded = joblib.load(config.PATHS["models"] / "classifier.pkl")
    if not hasattr(loaded["model"], "predict_proba"):
        raise RuntimeError("Saved model does not implement predict_proba.")

    print(table)


if __name__ == "__main__":
    main()
