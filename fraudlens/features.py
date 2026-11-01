"""Feature definitions shared by data generation, training, and serving."""

FEATURES = [
    "amount",
    "hour",
    "distance_from_home_km",
    "transactions_last_hour",
    "failed_attempts_24h",
    "account_age_days",
    "new_device",
    "international",
    "merchant_risk",
]

DISPLAY_NAMES = {
    "amount": "Transaction amount",
    "hour": "Transaction hour",
    "distance_from_home_km": "Distance from usual location",
    "transactions_last_hour": "Transactions in the last hour",
    "failed_attempts_24h": "Failed attempts in 24 hours",
    "account_age_days": "Account age",
    "new_device": "New device",
    "international": "International transaction",
    "merchant_risk": "Merchant risk indicator",
}


def feature_reason(feature: str, value: float, reference: dict) -> str | None:
    """Return a plain-language indicator when a value is unusually high/low.

    These rule-based indicators add context to a model score. They are not a
    causal explanation of the model and must not be presented as one.
    """
    q = reference.get(feature, {})
    display = DISPLAY_NAMES.get(feature, feature.replace("_", " ").title())
    if feature in {"new_device", "international"} and value >= 0.5:
        return f"{display}: yes"
    if feature == "account_age_days" and value < q.get("p10", 0):
        return f"{display}: unusually new"
    if value > q.get("p95", float("inf")):
        suffix = " km" if feature == "distance_from_home_km" else ""
        return f"{display}: above the typical range ({value:.1f}{suffix})"
    if feature == "failed_attempts_24h" and value >= 2:
        return f"{display}: {int(value)} recent failures"
    if feature == "transactions_last_hour" and value >= 5:
        return f"{display}: {int(value)} in a short window"
    if feature == "hour" and (value <= 4 or value >= 23):
        return f"{display}: unusual hour ({int(value):02d}:00)"
    return None
