"""Lightweight population stability monitoring for numeric features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calculate PSI using reference quantiles and a small smoothing constant."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_share = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_share = np.clip(ref_share, 1e-5, None)
    cur_share = np.clip(cur_share, 1e-5, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> list[dict]:
    report = []
    for feature in features:
        psi = population_stability_index(reference[feature], current[feature])
        level = "high" if psi >= 0.25 else "moderate" if psi >= 0.10 else "low"
        report.append({"feature": feature, "psi": round(psi, 4), "level": level})
    return sorted(report, key=lambda item: item["psi"], reverse=True)
