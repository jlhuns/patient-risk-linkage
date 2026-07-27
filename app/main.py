"""
FastAPI service wrapping the matching engine + risk model, plus a small
HTML dashboard for demos. One deployable unit.
"""

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.monitoring import get_recent_predictions, get_summary_stats, log_prediction
from matching.probabilistic import FEATURE_COLUMNS, extract_features
from models.train import CATEGORICAL_FEATURES, NUMERIC_FEATURES

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="Patient Risk & Identity Resolution API")
# no {{ }} in the template so just serve it as static HTML — hit a
# Jinja2Templates version bug and it wasn't worth chasing for a static page
_DASHBOARD_HTML = (BASE_DIR / "app" / "templates" / "index.html").read_text(encoding="utf-8")

_match_model = joblib.load(BASE_DIR / "matching" / "probabilistic_model.joblib")
_risk_model = joblib.load(BASE_DIR / "models" / "logistic_regression.joblib")


class PatientRecord(BaseModel):
    first: str
    last: str
    dob: str  # YYYY-MM-DD
    zip: str
    ssn: Optional[str] = None


class MatchRequest(BaseModel):
    record_a: PatientRecord
    record_b: PatientRecord


class RiskRequest(BaseModel):
    prior_encounter_count: Optional[float] = None
    prior_inpatient_count: Optional[float] = None
    prior_condition_count: Optional[float] = None
    prior_procedure_count: Optional[float] = None
    prior_medication_count: Optional[float] = None
    chronic_condition_count: Optional[float] = None
    bmi: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    heart_rate: Optional[float] = None
    glucose: Optional[float] = None
    hba1c: Optional[float] = None
    age_at_encounter: Optional[float] = None
    base_cost: Optional[float] = None
    total_claim_cost: Optional[float] = None
    gender: Optional[str] = None
    race: Optional[str] = None
    ethnicity: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return _DASHBOARD_HTML


@app.post("/api/match")
def match(req: MatchRequest):
    a, b = req.record_a.model_dump(), req.record_b.model_dump()
    features = extract_features(a, b)
    row = pd.DataFrame([features])[FEATURE_COLUMNS]
    score = float(_match_model.predict_proba(row)[0, 1])

    log_prediction("match", f"{a['first']} {a['last']} <-> {b['first']} {b['last']}", score)

    return {
        "score": round(score, 4),
        "is_match": score >= 0.5,
        "features": features,
    }


@app.post("/api/predict-risk")
def predict_risk(req: RiskRequest):
    row = pd.DataFrame([req.model_dump()])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    score = float(_risk_model.predict_proba(row)[0, 1])

    log_prediction("risk", f"age={req.age_at_encounter}, prior_inpatient={req.prior_inpatient_count}", score)

    return {"readmission_risk_30d": round(score, 4)}


@app.get("/api/monitoring")
def monitoring():
    return {
        "summary": get_summary_stats(),
        "recent": get_recent_predictions(limit=25),
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}
