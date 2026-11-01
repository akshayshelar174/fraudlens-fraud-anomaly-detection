# Synthetic data dictionary

| Field | Meaning | Example range |
|---|---|---|
| `transaction_id` | Synthetic row identifier | `SYN-0000001` |
| `event_day` | Synthetic relative day from 0 to 119 | integer |
| `amount` | Transaction amount in demo currency units | 1–5,000 |
| `hour` | Local transaction hour | 0–23 |
| `distance_from_home_km` | Simulated distance from usual location | 0–800 km |
| `transactions_last_hour` | Simulated transaction count in prior hour | non-negative integer |
| `failed_attempts_24h` | Simulated failed attempts in prior day | non-negative integer |
| `account_age_days` | Simulated account age | 1–5,000 days |
| `new_device` | Whether a new device is used | 0 or 1 |
| `international` | Whether the transaction is international | 0 or 1 |
| `merchant_risk` | Generated merchant risk indicator | 0–1 |
| `is_fraud` | Synthetic target label, training/demo only | 0 or 1 |
