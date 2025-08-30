from fastapi import FastAPI
from pydantic import BaseModel
import os, joblib, pandas as pd
from google.cloud import storage

PROJECT_ID   = os.getenv("PROJECT_ID")
MODEL_BUCKET = os.getenv("MODEL_BUCKET")
MODEL_LOCAL  = "/tmp/model.joblib"

FEATURES = ["trip_distance","passenger_count","pickup_hour","pickup_dow","pickup_month"]

app = FastAPI(title="Taxi Fare API")

def download_current_model():
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(MODEL_BUCKET)
    blob = bucket.blob("registry/current/model.joblib")
    blob.download_to_filename(MODEL_LOCAL)
    return joblib.load(MODEL_LOCAL)

# Load at startup
model = download_current_model()

class FarePayload(BaseModel):
    trip_distance: float
    passenger_count: int
    pickup_hour: int
    pickup_dow: int
    pickup_month: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_one(p: FarePayload):
    X = pd.DataFrame([p.dict()])[FEATURES]
    y = float(model.predict(X)[0])
    return {"predicted_fare": y}

@app.post("/predict_batch")
def predict_batch(rows: list[FarePayload]):
    df = pd.DataFrame([r.dict() for r in rows])[FEATURES]
    preds = model.predict(df).tolist()
    return {"predicted_fares": preds}

@app.post("/reload")
def reload_model():
    global model
    model = download_current_model()
    return {"status": "reloaded"}
