"""Train and compare three supervised models on synthetic, imbalanced data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from fraudlens.data import DATA_DIR, make_transactions
from fraudlens.features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"


def pick_threshold(labels: np.ndarray, scores: np.ndarray, precision_floor: float = 0.50) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        return 0.5
    p, r = precision[:-1], recall[:-1]
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    eligible = np.flatnonzero(p >= precision_floor)
    candidates = eligible if len(eligible) else np.arange(len(thresholds))
    best = candidates[np.argmax(f1[candidates])]
    return float(np.clip(thresholds[best], 0.01, 0.99))


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": round(threshold, 5),
        "average_precision": round(float(average_precision_score(labels, scores)), 5),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 5),
        "precision": round(float(precision_score(labels, predicted, zero_division=0)), 5),
        "recall": round(float(recall_score(labels, predicted, zero_division=0)), 5),
        "f1": round(float(f1_score(labels, predicted, zero_division=0)), 5),
        "confusion": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)},
    }


def train(rows: int = 12_000, seed: int = 42) -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame = make_transactions(rows=rows, seed=seed)
    frame.to_csv(DATA_DIR / "transactions.csv", index=False)
    x, y = frame[FEATURES], frame["is_fraud"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=seed, stratify=y
    )
    x_fit, x_valid, y_fit, y_valid = train_test_split(
        x_train, y_train, test_size=0.25, random_state=seed, stratify=y_train
    )

    candidates = {
        "Logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1_200, random_state=seed),
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=220, max_depth=14, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        ),
        "Histogram gradient boosting": HistGradientBoostingClassifier(
            max_iter=180, learning_rate=0.08, max_leaf_nodes=20,
            l2_regularization=1.0, random_state=seed,
        ),
    }

    fitted, thresholds, validation_scores, results = {}, {}, {}, {}
    weights = compute_sample_weight(class_weight="balanced", y=y_fit)
    for name, model in candidates.items():
        if name == "Histogram gradient boosting":
            model.fit(x_fit, y_fit, sample_weight=weights)
        else:
            model.fit(x_fit, y_fit)
        valid = model.predict_proba(x_valid)[:, 1]
        thresholds[name] = pick_threshold(y_valid.to_numpy(), valid)
        validation_scores[name] = float(average_precision_score(y_valid, valid))
        test_scores = model.predict_proba(x_test)[:, 1]
        results[name] = {
            "validation_average_precision": round(validation_scores[name], 5),
            "test": _metrics(y_test.to_numpy(), test_scores, thresholds[name]),
        }
        fitted[name] = model

    selected_name = max(validation_scores, key=validation_scores.get)
    selected_model = fitted[selected_name]
    reference = frame.loc[frame["is_fraud"] == 0, FEATURES]
    reference_stats = {
        feature: {
            "p10": float(reference[feature].quantile(0.10)),
            "p50": float(reference[feature].quantile(0.50)),
            "p90": float(reference[feature].quantile(0.90)),
            "p95": float(reference[feature].quantile(0.95)),
        }
        for feature in FEATURES
    }
    bundle = {
        "model": selected_model,
        "model_name": selected_name,
        "threshold": thresholds[selected_name],
        "features": FEATURES,
        "reference_stats": reference_stats,
        "trained_rows": int(rows),
        "seed": seed,
    }
    joblib.dump(bundle, ARTIFACT_DIR / "model_bundle.joblib")
    report = {
        "selected_model": selected_name,
        "threshold_selection": "Validation split; maximize F1 among thresholds with precision >= 0.50, otherwise maximize F1.",
        "split": {"train": 0.60, "validation": 0.20, "test": 0.20},
        "fraud_rate": round(float(y.mean()), 5),
        "rows": int(rows),
        "models": results,
        "features": FEATURES,
    }
    (ARTIFACT_DIR / "model_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare FraudLens models.")
    parser.add_argument("--rows", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = train(args.rows, args.seed)
    print(f"Selected model: {report['selected_model']}")
    print(f"Synthetic fraud rate: {report['fraud_rate']:.2%}")
    for name, result in report["models"].items():
        print(f"{name}: test PR-AUC={result['test']['average_precision']:.3f}, "
              f"precision={result['test']['precision']:.3f}, recall={result['test']['recall']:.3f}")
    print(f"Artifacts saved to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
