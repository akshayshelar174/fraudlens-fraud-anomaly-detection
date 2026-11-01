"""Generate reproducible synthetic transactions for demos and local training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraudlens.features import FEATURES

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def make_transactions(rows: int = 12_000, seed: int = 42) -> pd.DataFrame:
    if rows < 500:
        raise ValueError("Use at least 500 rows to keep train/test splits useful.")
    rng = np.random.default_rng(seed)
    day = rng.integers(0, 120, size=rows)
    amount = rng.lognormal(mean=3.55, sigma=1.05, size=rows).clip(1, 5_000)
    hour = rng.integers(0, 24, size=rows)
    distance = rng.gamma(shape=1.6, scale=16, size=rows).clip(0, 800)
    velocity = rng.poisson(lam=1.4, size=rows)
    failed = rng.poisson(lam=0.16, size=rows).clip(0, 8)
    account_age = rng.lognormal(mean=5.3, sigma=1.15, size=rows).clip(1, 5_000)
    new_device = rng.binomial(1, 0.09, size=rows)
    international = rng.binomial(1, 0.045, size=rows)
    merchant_risk = rng.beta(a=2.1, b=5.0, size=rows)

    # A small, documented population shift in the newest time window lets the
    # dashboard demonstrate drift monitoring without using real transactions.
    recent = day >= 96
    amount[recent] *= rng.lognormal(mean=0.12, sigma=0.08, size=recent.sum())
    new_device[recent] = np.maximum(new_device[recent], rng.binomial(1, 0.13, size=recent.sum()))
    velocity[recent] += rng.binomial(1, 0.18, size=recent.sum())

    # Latent fraud likelihood creates a realistic minority class and several
    # overlapping patterns instead of a single deterministic rule.
    logit = (
        -5.0
        + 0.72 * np.log1p(amount / 90)
        + 0.75 * (distance > 90)
        + 0.72 * (velocity >= 5)
        + 0.95 * (failed >= 2)
        + 1.05 * new_device
        + 1.20 * international
        + 1.10 * merchant_risk
        + 0.55 * ((hour <= 4) | (hour >= 23))
        + 0.65 * (account_age < 35)
    )
    probability = 1 / (1 + np.exp(-logit))
    fraud = rng.binomial(1, probability)

    frame = pd.DataFrame({
        "transaction_id": [f"SYN-{i:07d}" for i in range(rows)],
        "event_day": day,
        "amount": amount.round(2),
        "hour": hour,
        "distance_from_home_km": distance.round(2),
        "transactions_last_hour": velocity,
        "failed_attempts_24h": failed,
        "account_age_days": account_age.round(0).astype(int),
        "new_device": new_device.astype(int),
        "international": international.astype(int),
        "merchant_risk": merchant_risk.round(4),
        "is_fraud": fraud.astype(int),
    })
    return frame[["transaction_id", "event_day", *FEATURES, "is_fraud"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic transaction data.")
    parser.add_argument("--rows", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "transactions.csv")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = make_transactions(args.rows, args.seed)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame):,} synthetic transactions to {args.output}")
    print(f"Fraud rate: {frame['is_fraud'].mean():.2%}")


if __name__ == "__main__":
    main()
