# FraudLens

**An explainable fraud and anomaly detection portfolio project.**

FraudLens compares three machine learning models on an imbalanced synthetic transaction dataset, selects an operating threshold on validation data, scores transactions through a FastAPI service, and gives analysts a dashboard for triage, model performance, and drift monitoring.

> **Synthetic demo only:** All transaction records are generated locally. No payment processor or live data source is connected. Scores are for demonstration and analyst triage; they are not calibrated fraud probabilities and must not authorize payments.

## Project features

- Synthetic data generation with a documented class imbalance and a time-window distribution shift
- Logistic regression, random forest, and histogram gradient boosting comparison
- Separate fit, validation, and held-out test splits
- PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix, and validation-selected threshold
- FastAPI scoring endpoint with validated feature-only input
- Streamlit analyst dashboard with adjustable review threshold and CSV export
- Population Stability Index (PSI) drift view
- Plain-language risk indicators based on reference ranges; they are not causal model explanations
- Local one-feature ablation attribution on individual API scores
- Docker and Compose configuration

## Live demo

The Streamlit dashboard is deployed on Render: [Open FraudLens](https://fraudlens-dashboard.onrender.com).

The hosted app uses generated synthetic transactions and is for demonstration only. It is not connected to a payment processor or live customer data, and its scores must not be used to authorize payments. Render's free service may spin down when idle and take about a minute to start on the next visit.

The Render web service uses the repository's Dockerfile and starts the dashboard with:

```bash
streamlit run dashboard.py --server.address=0.0.0.0 --server.port=10000
```

## Run locally

Python 3.10 or later is required. From this directory:

```bash
python -m venv .venv
```

Activate the environment, then install the project:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Train the models and create the synthetic dashboard dataset:

```bash
python -m fraudlens.training --rows 12000
```

Start the API:

```bash
uvicorn fraudlens.api:app --reload
```

Open the API docs at `http://127.0.0.1:8000/docs`. In a second terminal, start the dashboard:

```bash
streamlit run dashboard.py
```

Open the dashboard at `http://localhost:8501`.

## Score one synthetic transaction

```powershell
$payload = @{
  transaction_id = "SYN-DEMO-1"
  amount = 920
  hour = 2
  distance_from_home_km = 280
  transactions_last_hour = 6
  failed_attempts_24h = 2
  account_age_days = 8
  new_device = 1
  international = 1
  merchant_risk = 0.82
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/score" -Method Post `
  -ContentType "application/json" -Body $payload
```

The `/score` response contains a relative 0–100 model score, an `allow` or `review` triage suggestion, the active threshold, reference-range risk indicators, and the top one-feature score sensitivities against typical legitimate values. These ablations are not causal and do not account for feature interactions. The request schema deliberately excludes card numbers and other direct identifiers.

## Docker

```bash
docker compose up --build
```

The dashboard is at `http://localhost:8501`; the API docs are at `http://localhost:8000/docs`.

## Metrics and threshold choice

Fraud is a minority class, so accuracy alone is misleading. FraudLens selects a threshold on the validation partition, preferring the highest F1 threshold with at least 50% precision when one is available. It reports the resulting metrics once on a separate held-out test partition. Actual operations need threshold selection based on investigation capacity, fraud cost, customer friction, and delayed labels.

## Project notes

- [Model card](docs/MODEL_CARD.md)
- [Synthetic data dictionary](docs/DATA_DICTIONARY.md)
- Model artifacts and generated CSVs are local outputs and are excluded from Git.
- The dashboard's `risk_indicators` are simple reference-range signals, not SHAP or a causal explanation.

## Suggested next steps

1. Add an authorized and privacy-reviewed dataset.
2. Validate the split strategy against time and label delay.
3. Calibrate scores and set thresholds with an explicit cost model.
4. Add analyst feedback and monitor precision at a fixed review capacity.
5. Replace heuristic risk indicators with a tested explanation method.
