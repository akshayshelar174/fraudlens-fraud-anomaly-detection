# FraudLens model card

## Intended use

Portfolio demonstration of imbalanced binary classification, threshold selection, analyst triage, and feature drift monitoring. It is not a payment authorization system and is not validated for operational fraud decisions.

## Data

Training and dashboard records are generated locally by `fraudlens.data`. They do not represent real customers, merchants, payment accounts, or observed fraud rates. The synthetic generator intentionally makes fraud rare and creates overlapping suspicious patterns. A small time-window shift is included only to demonstrate PSI monitoring.

## Models compared

- Class-weighted logistic regression
- Balanced random forest
- Histogram gradient boosting with balanced training sample weights

The data is split into fit (60%), validation (20%), and held-out test (20%) partitions. The selected model is the highest validation PR-AUC. Its decision threshold is selected on validation data, then reported once on test data. Accuracy is not the primary metric because the positive class is rare.

## Scores and reasons

The API's 0–100 score is a relative model score, not a calibrated fraud probability. `risk_indicators` are reference-range checks. The API also returns one-feature ablation sensitivities: it replaces one observed feature at a time with the legitimate-reference median and measures the score change. These are local sensitivity checks, not causal effects, SHAP values, or additive explanations; they do not capture feature interactions.

## Limitations and safeguards

Synthetic performance does not transfer to real payment streams. Real use would require authorized data, leakage review, calibration, label-delay handling, segment and fairness analysis where appropriate, security and privacy review, workload-aware threshold selection, monitoring ownership, and human review. Do not send card numbers, security codes, names, addresses, or real customer records to the demo.
