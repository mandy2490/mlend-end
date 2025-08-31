import os, json, joblib
import pandas as pd
from datetime import datetime
from google.cloud import bigquery, storage
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
#testing dev branch
# --- Read env vars injected by GitHub Actions ---
PROJECT_ID   = os.getenv("PROJECT_ID")
BQ_TABLE     = os.getenv("BQ_TABLE")        # e.g. "<PROJECT>.ml.taxi_fare_train"
MODEL_BUCKET = os.getenv("MODEL_BUCKET")    # e.g. "ml-models-<PROJECT>"

assert PROJECT_ID, "Missing env var PROJECT_ID"
assert BQ_TABLE,   "Missing env var BQ_TABLE (e.g. <PROJECT>.ml.taxi_fare_train)"
assert MODEL_BUCKET, "Missing env var MODEL_BUCKET (your GCS bucket name)"

# 1) Load data from BigQuery
bq = bigquery.Client(project=PROJECT_ID)
query = f"""
SELECT
  fare_amount,
  trip_distance,
  passenger_count,
  pickup_hour,
  pickup_dow,
  pickup_month
FROM `{BQ_TABLE}`
"""
print("Querying BigQuery…")
df = bq.query(query).to_dataframe()
print(f"Loaded {len(df):,} rows")

if len(df) < 100:
    raise RuntimeError("Not enough rows to train. Check your BigQuery table or filters.")

# 2) Split
X = df[["trip_distance","passenger_count","pickup_hour","pickup_dow","pickup_month"]]
y = df["fare_amount"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# 3) Train
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(Xtr, ytr)

# 4) Evaluate
pred = model.predict(Xte)
mae = float(mean_absolute_error(yte, pred))
print(f"Validation MAE: {mae:.2f}")

# 5) Compare to previous best in GCS
storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(MODEL_BUCKET)
meta_blob = bucket.blob("registry/metadata.json")

best_mae = None
if meta_blob.exists():
    meta = json.loads(meta_blob.download_as_text())
    best_mae = float(meta.get("best_mae", 1e18))
    print("Previous best MAE:", best_mae)
else:
    meta = {}
    print("No previous model metadata found. This will become the first best model.")

# 6) Promote if better
did_promote = False
if (best_mae is None) or (mae < best_mae):
    print("New model is better. Promoting…")
    did_promote = True

    # save model locally in runner
    os.makedirs("/tmp", exist_ok=True)
    joblib.dump(model, "/tmp/model.joblib")

    # upload versioned & current
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    version_path = f"registry/versions/{ts}/model.joblib"
    bucket.blob(version_path).upload_from_filename("/tmp/model.joblib")
    bucket.blob("registry/current/model.joblib").upload_from_filename("/tmp/model.joblib")

    # update metadata
    meta = {"best_mae": mae, "last_promoted": ts, "version_path": version_path}
    meta_blob.upload_from_string(json.dumps(meta))
    print(f"Promoted model. Version: gs://{MODEL_BUCKET}/{version_path}")
else:
    print("New model is NOT better. Keeping current best.")

# 7) Emit a small JSON (later used to decide deployment)
with open("promotion.json", "w") as f:
    json.dump({"did_promote": did_promote, "mae": mae, "best_mae": best_mae}, f)
print("Wrote promotion.json")
