# serving image — training happens offline. No TensorFlow, the API just
# loads a saved sklearn pipeline (see requirements-prod.txt).
FROM python:3.12-slim

WORKDIR /srv

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY app/ app/
COPY matching/__init__.py matching/deterministic.py matching/probabilistic.py matching/probabilistic_model.joblib matching/
COPY models/__init__.py models/features.py models/train.py models/logistic_regression.joblib models/
COPY db/ db/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
