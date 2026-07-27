"""
Training table for 30-day inpatient readmission risk. One row per inpatient
encounter, label + features computed only from data available before that
encounter started (no leakage).

Cohort is 'inpatient' encounters only, matching CMS's actual readmission
definition — emergency/urgentcare visits would inflate the cohort but
change what the model's predicting.
"""

from datetime import datetime

import pandas as pd
from sqlalchemy import text

from db.connection import get_engine

# SNOMED-CT codes: diabetes, hypertension, COPD, CHF — standard risk
# factors in readmission models. Not exhaustive, picked deliberately.
CHRONIC_CONDITION_CODES = ("44054006", "38341003", "13645005", "88805009")

# LOINC codes matching the vitals/labs panel loaded in etl/load_synthea.py.
VITAL_CODES = {
    "39156-5": "bmi",
    "8480-6": "systolic_bp",
    "8462-4": "diastolic_bp",
    "8867-4": "heart_rate",
    "2339-0": "glucose",
    "4548-4": "hba1c",
}


def ensure_indexes(engine):
    # without these the correlated subqueries below do a full table scan
    # per cohort row instead of an index seek
    statements = [
        "CREATE INDEX IF NOT EXISTS ix_encounter_patient_start ON fact_encounter(patient_id, start_ts)",
        "CREATE INDEX IF NOT EXISTS ix_condition_patient_start ON fact_condition(patient_id, start_date)",
        "CREATE INDEX IF NOT EXISTS ix_procedure_patient_start ON fact_procedure(patient_id, start_ts)",
        "CREATE INDEX IF NOT EXISTS ix_medication_patient_start ON fact_medication(patient_id, start_ts)",
        "CREATE INDEX IF NOT EXISTS ix_observation_patient_code_date ON fact_observation(patient_id, code, obs_date)",
    ]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


COHORT_AND_LABEL_SQL = """
WITH inpatient_encounters AS (
    SELECT encounter_id, patient_id, start_ts, stop_ts, base_cost, total_claim_cost
    FROM fact_encounter
    WHERE encounter_class = 'inpatient' AND stop_ts IS NOT NULL
)
SELECT
    e1.encounter_id,
    e1.patient_id,
    e1.start_ts,
    e1.stop_ts,
    e1.base_cost,
    e1.total_claim_cost,
    CASE WHEN EXISTS (
        SELECT 1 FROM inpatient_encounters e2
        WHERE e2.patient_id = e1.patient_id
          AND e2.encounter_id != e1.encounter_id
          AND e2.start_ts > e1.stop_ts
          AND e2.start_ts <= datetime(e1.stop_ts, '+30 days')
    ) THEN 1 ELSE 0 END AS readmitted_30d,
    (SELECT COUNT(*) FROM fact_encounter fe
        WHERE fe.patient_id = e1.patient_id AND fe.start_ts < e1.start_ts) AS prior_encounter_count,
    (SELECT COUNT(*) FROM inpatient_encounters ip
        WHERE ip.patient_id = e1.patient_id AND ip.start_ts < e1.start_ts) AS prior_inpatient_count,
    (SELECT COUNT(*) FROM fact_condition fc
        WHERE fc.patient_id = e1.patient_id AND fc.start_date < date(e1.start_ts)) AS prior_condition_count,
    (SELECT COUNT(*) FROM fact_procedure fp
        WHERE fp.patient_id = e1.patient_id AND fp.start_ts < e1.start_ts) AS prior_procedure_count,
    (SELECT COUNT(*) FROM fact_medication fm
        WHERE fm.patient_id = e1.patient_id AND fm.start_ts < e1.start_ts) AS prior_medication_count,
    (SELECT COUNT(*) FROM fact_condition fc
        WHERE fc.patient_id = e1.patient_id AND fc.start_date < date(e1.start_ts)
        AND fc.code IN ({chronic_codes})) AS chronic_condition_count
FROM inpatient_encounters e1
"""


def build_cohort(engine) -> pd.DataFrame:
    placeholders = ",".join(f"'{c}'" for c in CHRONIC_CONDITION_CODES)
    sql = COHORT_AND_LABEL_SQL.format(chronic_codes=placeholders)
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    return df


def attach_prior_vitals(cohort: pd.DataFrame, engine) -> pd.DataFrame:
    # "latest value before this encounter" — merge_asof handles this
    # directly instead of a correlated subquery per row
    codes = tuple(VITAL_CODES.keys())
    with engine.connect() as conn:
        obs = pd.read_sql(
            text(f"SELECT patient_id, code, obs_date, value FROM fact_observation WHERE code IN {codes}"),
            conn,
        )
    obs["obs_date"] = pd.to_datetime(obs["obs_date"], format="mixed", utc=True)
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    cohort = cohort.copy()
    cohort["start_ts"] = pd.to_datetime(cohort["start_ts"], format="mixed", utc=True)

    for code, feature_name in VITAL_CODES.items():
        sub = obs[obs["code"] == code][["patient_id", "obs_date", "value"]].dropna()
        sub = sub.sort_values("obs_date").rename(columns={"value": feature_name})
        cohort_sorted = cohort.sort_values("start_ts")
        merged = pd.merge_asof(
            cohort_sorted,
            sub,
            left_on="start_ts",
            right_on="obs_date",
            by=None,
            left_by="patient_id",
            right_by="patient_id",
            direction="backward",
        )
        cohort = merged.drop(columns=["obs_date"], errors="ignore")

    return cohort


def attach_demographics(cohort: pd.DataFrame, engine) -> pd.DataFrame:
    with engine.connect() as conn:
        patients = pd.read_sql(
            text("SELECT patient_id, birth_date, gender, race, ethnicity FROM dim_patient"), conn
        )
    patients["birth_date"] = pd.to_datetime(patients["birth_date"], utc=True)
    merged = cohort.merge(patients, on="patient_id", how="left")
    merged["age_at_encounter"] = (merged["start_ts"] - merged["birth_date"]).dt.days / 365.25
    return merged.drop(columns=["birth_date"])


def build_training_frame() -> pd.DataFrame:
    engine = get_engine()
    ensure_indexes(engine)
    cohort = build_cohort(engine)
    cohort = attach_prior_vitals(cohort, engine)
    cohort = attach_demographics(cohort, engine)
    return cohort


if __name__ == "__main__":
    df = build_training_frame()
    print(f"Cohort size: {len(df):,} inpatient encounters")
    print(f"Readmitted within 30 days: {df['readmitted_30d'].sum():,} ({df['readmitted_30d'].mean():.1%})")
    print(df.describe(include="all").T)
    df.to_csv("models/training_data.csv", index=False)
    print("Wrote models/training_data.csv")
