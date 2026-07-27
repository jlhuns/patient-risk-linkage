# Patient Risk & Identity Resolution

Healthcare ML pipeline on synthetic patient data: patient record matching
(deterministic + probabilistic linkage), a 30-day readmission-risk model,
and a live API + dashboard with basic monitoring.

**Live demo**: https://6uuboxz45h.execute-api.us-west-2.amazonaws.com

Data is 100% synthetic — generated with [Synthea](https://github.com/synthetichealth/synthea),
the open-source synthetic patient generator MITRE built for exactly this
kind of thing. No real PHI anywhere, so the whole thing can just be public.

I built this to get real hands-on depth in patient identity resolution —
reconciling duplicate patient records across systems — which is a real
problem in healthcare data work but barely shows up in typical ML
portfolios.

## Architecture

```
Synthea --> CSV export --> etl/load_synthea.py --> warehouse (db/schema.py)
                                                          |
                    +-------------------------------------+-------------------------------------+
                    |                                                                             |
            matching/ (record linkage)                                              models/ (risk prediction)
            deterministic.py + probabilistic.py                                     features.py + train.py
                    |                                                                             |
                    +-------------------------------------+-------------------------------------+
                                                          |
                                        app/main.py (FastAPI) --> dashboard + monitoring
```

## The data

3,332 synthetic Utah patients (Synthea), loaded into a fact/dimension
warehouse (`db/schema.py`) — 1.2M+ rows across encounters, conditions,
procedures, medications, and a filtered vitals/labs panel, coded with
SNOMED-CT / RxNorm / LOINC.

| table | rows |
|---|---|
| encounters | 180,520 |
| conditions | 108,003 |
| procedures | 487,875 |
| medications | 148,626 |
| observations (filtered panel) | 279,312 |

## Patient matching

Two-stage approach, scored against known ground truth (see
`matching/dirty_data.py`, which simulates a messy second hospital system):

- **Deterministic** (`matching/deterministic.py`): block on DOB, match on
  exact SSN or (last name + first-name prefix + zip). **Precision 1.0,
  recall 0.627** — safe, but misses anything that doesn't agree exactly.
- **Probabilistic** (`matching/probabilistic.py`): relaxed blocking +
  Jaro-Winkler similarity features + logistic regression trained on
  labeled pairs. **Precision 1.0, recall 0.999.**

Recall jump from 0.627 to 0.999 at held precision is the headline number.

## Risk model: 30-day readmission

2,758 inpatient encounters, 11.6% readmitted within 30 days. Features are
built only from data available before the index encounter (`models/features.py`
handles the leakage-avoidance).

Mann-Whitney U test on prior inpatient stays (readmitted vs. not):
**p < 0.000001**, median 13 vs. 1 — real signal.

Trained and compared 4 models, patient-grouped train/test split:

| model | precision | recall | F1 | PR-AUC |
|---|---|---|---|---|
| neural network (Keras) | 0.24 | 0.78 | 0.36 | 0.254 |
| logistic regression | 0.22 | 0.65 | 0.33 | 0.239 |
| random forest | 0.21 | 0.04 | 0.07 | 0.244 |
| gradient boosted trees | 0.12 | 0.08 | 0.09 | 0.214 |

Neural net technically wins on PR-AUC. **Shipped logistic regression to
production anyway** — small margin, and it's directly interpretable plus
doesn't need a TensorFlow runtime in the serving container.

## API + dashboard

`app/main.py`:
- `POST /api/match` — score two patient records
- `POST /api/predict-risk` — 30-day readmission risk
- `GET /api/monitoring` — prediction volume/score log

`app/templates/index.html` — single-page dashboard for all three.

## Running locally

```bash
python -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt

java -jar synthea-with-dependencies.jar -c synthea.properties -p 3000 Utah
python etl/load_synthea.py

python matching/dirty_data.py
python matching/deterministic.py
python -m matching.probabilistic

python -m models.train

python -m uvicorn app.main:app --reload
```

## Deployment

AWS Lambda (container image) + API Gateway, not App Runner — this is a
low-traffic demo and Lambda's scale-to-zero billing fits that better than
App Runner's always-on compute. Tradeoff is cold-start latency on the
first request after idle.

`Dockerfile.lambda` is the deployed image; `Dockerfile` is a normal
long-running-server image kept around as the App Runner/Fargate/local
alternative (`app/lambda_handler.py` is the only difference — a Mangum
adapter for the Lambda front door).

Redshift Serverless was stood up to run the ETL against real AWS infra
(not just SQLite), verified (identical row counts + a join query matching
the SQLite run), then torn down — it's not in the live API's request path
(the API scores from saved model files, not live warehouse queries), so
there's no reason to keep it running. `db/connection.py`'s `DATABASE_URL`
is the only thing that changes to point at Redshift again.

One real bug from that: SQLite autoincrements its integer primary keys for
free, Redshift needs an explicit `IDENTITY` column, and they don't
translate cleanly through SQLAlchemy's generic autoincrement handling.
Fixed by generating surrogate keys in the ETL itself instead.

Errors go to CloudWatch Logs (automatic with Lambda, no setup). Uptime is
checked by a free UptimeRobot ping against `/api/health`. Each deploy is
tagged in ECR by git SHA (not just `latest`) so there's an actual rollback
path.
