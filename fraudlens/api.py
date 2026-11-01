"""FastAPI scoring service. Accepts feature values only; never card numbers."""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from fraudlens.features import FEATURES
from fraudlens.scoring import load_bundle, score_one

app = FastAPI(
    title="FraudLens API",
    description="Synthetic-data fraud triage demo. Scores are for analyst review, not payment authorization.",
    version="0.1.0",
)


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction_id: str | None = Field(default=None, max_length=80)
    amount: float = Field(ge=0, le=1_000_000)
    hour: int = Field(ge=0, le=23)
    distance_from_home_km: float = Field(ge=0, le=50_000)
    transactions_last_hour: int = Field(ge=0, le=10_000)
    failed_attempts_24h: int = Field(ge=0, le=10_000)
    account_age_days: int = Field(ge=0, le=100_000)
    new_device: int = Field(ge=0, le=1)
    international: int = Field(ge=0, le=1)
    merchant_risk: float = Field(ge=0, le=1)


class ScoreResponse(BaseModel):
    transaction_id: str
    risk_score: float
    decision: str
    threshold: float
    risk_indicators: list[str]
    feature_attributions: list[dict]
    model: str
    score_note: str
    explanation_note: str


@app.get("/health")
def health() -> dict:
    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready", "model": bundle["model_name"], "features": FEATURES}


@app.post("/score", response_model=ScoreResponse)
def score_transaction(
    transaction: Transaction,
    threshold: Annotated[float | None, Query(ge=0.01, le=0.99)] = None,
) -> dict:
    try:
        return score_one(transaction.model_dump(exclude_none=True), threshold=threshold)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
