"""
Loads a Synthea CSV export into the warehouse (db/schema.py).

    python etl/load_synthea.py [--csv-dir output/csv]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db.connection import get_engine
from db.schema import metadata

# LOINC codes for the vitals/labs I actually use in the risk model.
# Full observations table is ~2.18M rows and mostly irrelevant noise
# (imaging annotations, free-text notes) so filter at load time.
RELEVANT_OBS_CODES = {
    "39156-5",  # BMI
    "8480-6",  # systolic BP
    "8462-4",  # diastolic BP
    "8867-4",  # heart rate
    "2339-0",  # glucose
    "2093-3",  # total cholesterol
    "4548-4",  # hemoglobin A1c
    "718-7",  # hemoglobin
    "6299-2",  # urea nitrogen
    "2160-0",  # creatinine
}

CHUNK_SIZE = 5000


def load_csv(csv_dir: Path, name: str) -> pd.DataFrame:
    path = csv_dir / f"{name}.csv"
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_dim_code(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []

    conditions, procedures = frames["conditions"], frames["procedures"]
    for df in (conditions, procedures):
        parts.append(df[["SYSTEM", "CODE", "DESCRIPTION"]].rename(
            columns={"SYSTEM": "code_system", "CODE": "code", "DESCRIPTION": "description"}
        ))

    encounters = frames["encounters"]
    enc_codes = encounters[["CODE", "DESCRIPTION"]].copy()
    enc_codes["code_system"] = "SNOMED-CT"
    parts.append(enc_codes.rename(columns={"CODE": "code", "DESCRIPTION": "description"})[
        ["code_system", "code", "description"]
    ])

    medications = frames["medications"]
    med_codes = medications[["CODE", "DESCRIPTION"]].copy()
    med_codes["code_system"] = "RxNorm"
    parts.append(med_codes.rename(columns={"CODE": "code", "DESCRIPTION": "description"})[
        ["code_system", "code", "description"]
    ])

    observations = frames["observations"]
    obs = observations[observations["CODE"].isin(RELEVANT_OBS_CODES)]
    obs_codes = obs[["CODE", "DESCRIPTION"]].copy()
    obs_codes["code_system"] = "LOINC"
    parts.append(obs_codes.rename(columns={"CODE": "code", "DESCRIPTION": "description"})[
        ["code_system", "code", "description"]
    ])

    dim = pd.concat(parts, ignore_index=True).dropna(subset=["code_system", "code"])
    dim = dim.drop_duplicates(subset=["code_system", "code"])
    return dim


def to_sql_chunked(df: pd.DataFrame, table_name: str, engine, if_exists="append"):
    if df.empty:
        print(f"  {table_name}: 0 rows, skipping")
        return
    # method="multi" overflows SQLite's bound-variable limit at this row
    # count, so only use it on Redshift/Postgres.
    is_sqlite = engine.dialect.name == "sqlite"
    df.to_sql(
        table_name,
        engine,
        if_exists=if_exists,
        index=False,
        chunksize=CHUNK_SIZE,
        method=None if is_sqlite else "multi",
    )
    print(f"  {table_name}: loaded {len(df):,} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default="output/csv")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    engine = get_engine()

    print(f"Reading Synthea CSVs from {csv_dir} ...")
    frames = {
        name: load_csv(csv_dir, name)
        for name in ("patients", "organizations", "encounters", "conditions", "procedures", "medications", "observations")
    }

    print("Creating schema ...")
    metadata.drop_all(engine)
    metadata.create_all(engine)

    print("Loading dim_organization ...")
    orgs = frames["organizations"][["Id", "NAME", "CITY", "STATE", "ZIP"]].rename(
        columns={"Id": "organization_id", "NAME": "name", "CITY": "city", "STATE": "state", "ZIP": "zip"}
    )
    to_sql_chunked(orgs, "dim_organization", engine)

    print("Loading dim_patient ...")
    patients = frames["patients"][
        ["Id", "BIRTHDATE", "DEATHDATE", "GENDER", "RACE", "ETHNICITY", "MARITAL", "CITY", "STATE", "COUNTY", "ZIP", "LAT", "LON"]
    ].rename(columns={
        "Id": "patient_id", "BIRTHDATE": "birth_date", "DEATHDATE": "death_date",
        "GENDER": "gender", "RACE": "race", "ETHNICITY": "ethnicity", "MARITAL": "marital_status",
        "CITY": "city", "STATE": "state", "COUNTY": "county", "ZIP": "zip", "LAT": "lat", "LON": "lon",
    })
    to_sql_chunked(patients, "dim_patient", engine)

    print("Building + loading dim_code ...")
    dim_code = build_dim_code(frames)
    to_sql_chunked(dim_code, "dim_code", engine)

    print("Loading fact_encounter ...")
    encounters = frames["encounters"][
        ["Id", "PATIENT", "ORGANIZATION", "START", "STOP", "ENCOUNTERCLASS", "CODE", "BASE_ENCOUNTER_COST", "TOTAL_CLAIM_COST", "PAYER_COVERAGE"]
    ].rename(columns={
        "Id": "encounter_id", "PATIENT": "patient_id", "ORGANIZATION": "organization_id",
        "START": "start_ts", "STOP": "stop_ts", "ENCOUNTERCLASS": "encounter_class", "CODE": "code",
        "BASE_ENCOUNTER_COST": "base_cost", "TOTAL_CLAIM_COST": "total_claim_cost", "PAYER_COVERAGE": "payer_coverage",
    })
    encounters["code_system"] = "SNOMED-CT"
    to_sql_chunked(encounters, "fact_encounter", engine)

    print("Loading fact_condition ...")
    conditions = frames["conditions"][["PATIENT", "ENCOUNTER", "START", "STOP", "SYSTEM", "CODE"]].rename(columns={
        "PATIENT": "patient_id", "ENCOUNTER": "encounter_id", "START": "start_date", "STOP": "stop_date",
        "SYSTEM": "code_system", "CODE": "code",
    })
    # IDs assigned here instead of via DB autoincrement — SQLite and
    # Redshift don't agree on how that works, easier to just do it in pandas.
    conditions.insert(0, "condition_id", range(1, len(conditions) + 1))
    to_sql_chunked(conditions, "fact_condition", engine)

    print("Loading fact_procedure ...")
    procedures = frames["procedures"][["PATIENT", "ENCOUNTER", "START", "STOP", "SYSTEM", "CODE", "BASE_COST"]].rename(columns={
        "PATIENT": "patient_id", "ENCOUNTER": "encounter_id", "START": "start_ts", "STOP": "stop_ts",
        "SYSTEM": "code_system", "CODE": "code", "BASE_COST": "base_cost",
    })
    procedures.insert(0, "procedure_id", range(1, len(procedures) + 1))
    to_sql_chunked(procedures, "fact_procedure", engine)

    print("Loading fact_medication ...")
    medications = frames["medications"][["PATIENT", "ENCOUNTER", "START", "STOP", "CODE", "BASE_COST", "DISPENSES", "TOTALCOST"]].rename(columns={
        "PATIENT": "patient_id", "ENCOUNTER": "encounter_id", "START": "start_ts", "STOP": "stop_ts",
        "CODE": "code", "BASE_COST": "base_cost", "DISPENSES": "dispenses", "TOTALCOST": "total_cost",
    })
    medications["code_system"] = "RxNorm"
    medications.insert(0, "medication_id", range(1, len(medications) + 1))
    to_sql_chunked(medications, "fact_medication", engine)

    print("Loading fact_observation (filtered to risk-relevant panel) ...")
    observations = frames["observations"]
    observations = observations[observations["CODE"].isin(RELEVANT_OBS_CODES)]
    observations = observations[["PATIENT", "ENCOUNTER", "DATE", "CATEGORY", "CODE", "VALUE", "UNITS"]].rename(columns={
        "PATIENT": "patient_id", "ENCOUNTER": "encounter_id", "DATE": "obs_date",
        "CATEGORY": "category", "CODE": "code", "VALUE": "value", "UNITS": "units",
    })
    observations["code_system"] = "LOINC"
    observations.insert(0, "observation_id", range(1, len(observations) + 1))
    to_sql_chunked(observations, "fact_observation", engine)

    with engine.connect() as conn:
        counts = {}
        for t in metadata.tables:
            counts[t] = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
    print("\nFinal row counts:")
    for t, c in counts.items():
        print(f"  {t}: {c:,}")


if __name__ == "__main__":
    main()
