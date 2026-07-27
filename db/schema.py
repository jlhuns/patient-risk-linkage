"""
Warehouse schema: fact/dimension split.

dim_patient / dim_organization are entity dimensions, dim_code is a shared
reference table for SNOMED-CT/RxNorm/LOINC so descriptions aren't repeated
on every fact row. The fact_* tables are the clinical events.

SQLAlchemy Core, not raw DDL, so the same schema works against SQLite
(dev) and Redshift (prod) with just a connection string change.
"""

from sqlalchemy import (
    MetaData,
    Table,
    Column,
    String,
    Numeric,
    Integer,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    PrimaryKeyConstraint,
)

metadata = MetaData()

dim_patient = Table(
    "dim_patient",
    metadata,
    Column("patient_id", String(64), primary_key=True),
    Column("birth_date", Date, nullable=False),
    Column("death_date", Date, nullable=True),
    Column("gender", String(8)),
    Column("race", String(32)),
    Column("ethnicity", String(32)),
    Column("marital_status", String(8)),
    Column("city", String(64)),
    Column("state", String(32)),
    Column("county", String(64)),
    Column("zip", String(16)),
    Column("lat", Numeric(9, 6)),
    Column("lon", Numeric(9, 6)),
)

dim_organization = Table(
    "dim_organization",
    metadata,
    Column("organization_id", String(64), primary_key=True),
    Column("name", String(256)),
    Column("city", String(64)),
    Column("state", String(32)),
    Column("zip", String(16)),
)

# Composite key: the same code value can mean different things in different
# coding systems, so (system, code) together identify a concept.
dim_code = Table(
    "dim_code",
    metadata,
    Column("code_system", String(32), nullable=False),
    Column("code", String(32), nullable=False),
    Column("description", String(512)),
    PrimaryKeyConstraint("code_system", "code"),
)

fact_encounter = Table(
    "fact_encounter",
    metadata,
    Column("encounter_id", String(64), primary_key=True),
    Column("patient_id", String(64), ForeignKey("dim_patient.patient_id"), nullable=False),
    Column("organization_id", String(64), ForeignKey("dim_organization.organization_id")),
    Column("start_ts", DateTime, nullable=False),
    Column("stop_ts", DateTime),
    Column("encounter_class", String(32)),
    Column("code_system", String(32)),
    Column("code", String(32)),
    Column("base_cost", Numeric(12, 2)),
    Column("total_claim_cost", Numeric(12, 2)),
    Column("payer_coverage", Numeric(12, 2)),
    ForeignKeyConstraint(["code_system", "code"], ["dim_code.code_system", "dim_code.code"]),
)

fact_condition = Table(
    "fact_condition",
    metadata,
    Column("condition_id", Integer, primary_key=True, autoincrement=True),
    Column("patient_id", String(64), ForeignKey("dim_patient.patient_id"), nullable=False),
    Column("encounter_id", String(64), ForeignKey("fact_encounter.encounter_id")),
    Column("start_date", Date, nullable=False),
    Column("stop_date", Date),
    Column("code_system", String(32)),
    Column("code", String(32)),
    ForeignKeyConstraint(["code_system", "code"], ["dim_code.code_system", "dim_code.code"]),
)

fact_procedure = Table(
    "fact_procedure",
    metadata,
    Column("procedure_id", Integer, primary_key=True, autoincrement=True),
    Column("patient_id", String(64), ForeignKey("dim_patient.patient_id"), nullable=False),
    Column("encounter_id", String(64), ForeignKey("fact_encounter.encounter_id")),
    Column("start_ts", DateTime, nullable=False),
    Column("stop_ts", DateTime),
    Column("code_system", String(32)),
    Column("code", String(32)),
    Column("base_cost", Numeric(12, 2)),
    ForeignKeyConstraint(["code_system", "code"], ["dim_code.code_system", "dim_code.code"]),
)

fact_medication = Table(
    "fact_medication",
    metadata,
    Column("medication_id", Integer, primary_key=True, autoincrement=True),
    Column("patient_id", String(64), ForeignKey("dim_patient.patient_id"), nullable=False),
    Column("encounter_id", String(64), ForeignKey("fact_encounter.encounter_id")),
    Column("start_ts", DateTime, nullable=False),
    Column("stop_ts", DateTime),
    Column("code_system", String(32)),  # RxNorm, by Synthea convention
    Column("code", String(32)),
    Column("base_cost", Numeric(12, 2)),
    Column("dispenses", Integer),
    Column("total_cost", Numeric(12, 2)),
    ForeignKeyConstraint(["code_system", "code"], ["dim_code.code_system", "dim_code.code"]),
)

# Curated, not a full dump of Synthea's 2.18M observation rows: filtered at
# load time to the vital-sign / lab panel relevant to the risk model, since
# most of the raw table is analytically irrelevant to this project's questions.
fact_observation = Table(
    "fact_observation",
    metadata,
    Column("observation_id", Integer, primary_key=True, autoincrement=True),
    Column("patient_id", String(64), ForeignKey("dim_patient.patient_id"), nullable=False),
    Column("encounter_id", String(64), ForeignKey("fact_encounter.encounter_id")),
    Column("obs_date", DateTime, nullable=False),
    Column("category", String(32)),
    Column("code_system", String(32)),  # LOINC, by Synthea convention
    Column("code", String(32)),
    Column("value", String(64)),  # kept as text: Synthea mixes numeric and coded values
    Column("units", String(32)),
    ForeignKeyConstraint(["code_system", "code"], ["dim_code.code_system", "dim_code.code"]),
)
