"""Model loading and risk triage shared by the API and analyst dashboard."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from fraudlens.features import DISPLAY_NAMES, FEATURES, feature_reason
from fraudlens.training import ARTIFACT_DIR


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    path = ARTIFACT_DIR / "model_bundle.joblib"
    if not path.exists():
        raise FileNotFoundError("Model artifacts are missing. Run `python -m fraudlens.training` first.")
    return joblib.load(path)


def score_frame(frame: pd.DataFrame, bundle: dict | None = None, threshold: float | None = None) -> pd.DataFrame:
    bundle = bundle or load_bundle()
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature(s): {', '.join(missing)}")
    values = frame[FEATURES].astype(float)
    if values.isna().any().any():
        raise ValueError("Feature values cannot be blank or null.")
    if (values["amount"] < 0).any():
        raise ValueError("Transaction amount must be zero or greater.")

    scores = bundle["model"].predict_proba(values)[:, 1]
    cutoff = float(bundle["threshold"] if threshold is None else threshold)
    output = frame.copy()
    output["risk_score"] = scores * 100
    output["decision_threshold"] = cutoff * 100
    output["decision"] = ["review" if value >= cutoff else "allow" for value in scores]
    output["risk_indicators"] = [
        [reason for feature, value in row.items()
         if (reason := feature_reason(feature, float(value), bundle["reference_stats"]))]
        for row in values.to_dict(orient="records")
    ]
    output["risk_indicators"] = output["risk_indicators"].apply(
        lambda reasons: reasons or ["No individual indicator crossed its reference cutoff."]
    )
    return output


def explain_one(payload: dict, bundle: dict | None = None, top_k: int = 3) -> list[dict]:
    """Compare a score with one-at-a-time reference-median replacements."""
    bundle = bundle or load_bundle()
    values = pd.DataFrame([{name: payload[name] for name in FEATURES}]).astype(float)
    base_score = float(bundle["model"].predict_proba(values)[0, 1])
    reference = bundle["reference_stats"]
    ablations = []
    for feature in FEATURES:
        typical = float(reference[feature]["p50"])
        counterfactual = values.copy()
        counterfactual.loc[counterfactual.index[0], feature] = typical
        replaced_score = float(bundle["model"].predict_proba(counterfactual)[0, 1])
        change = (base_score - replaced_score) * 100
        if change > 0.01:
            ablations.append({
                "feature": feature,
                "display_name": DISPLAY_NAMES[feature],
                "observed_value": float(values.iloc[0][feature]),
                "reference_value": typical,
                "score_increase_points": round(change, 2),
            })
    ablations.sort(key=lambda item: item["score_increase_points"], reverse=True)
    return ablations[:top_k]


def score_one(payload: dict, threshold: float | None = None) -> dict:
    frame = pd.DataFrame([payload])
    row = score_frame(frame, threshold=threshold).iloc[0]
    bundle = load_bundle()
    return {
        "transaction_id": str(payload.get("transaction_id", "demo-transaction")),
        "risk_score": round(float(row["risk_score"]), 2),
        "decision": str(row["decision"]),
        "threshold": round(float(row["decision_threshold"]), 2),
        "risk_indicators": row["risk_indicators"],
        "feature_attributions": explain_one(payload, bundle=bundle),
        "model": bundle["model_name"],
        "score_note": "Relative model score for triage; it is not a calibrated fraud probability.",
        "explanation_note": "Attributions are one-feature ablations against the legitimate-reference median. They are local sensitivity checks, not causal effects; interactions are not additive.",
    }
