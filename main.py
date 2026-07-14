import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path



MODEL_PATH = Path("model.pkl")


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Please run train_churn.py first."
    )

app = FastAPI(title="Churn Prediction API")

model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
scaler = model_bundle["scaler"]
feature_cols = model_bundle["feature_cols"]
train_medians = model_bundle["train_medians"]

THRESHOLD = 0.5


# the request model for the prediction endpoint
class CustomerFeatures(BaseModel):
    tenure_months: int = Field(..., ge=0)
    monthly_charge: float | None = Field(None, ge=0)  # None -> training median
    support_tickets: int = Field(..., ge=0)
    contract_type: Literal["month-to-month", "one-year", "two-year"]
    age: int | None = Field(None, ge=0 )  # None -> training median

# the response model for the prediction endpoint
class PredictionResponse(BaseModel):
    churn_probability: float
    predicted_label: int


@app.post("/predict")
def predict_churn(customer: CustomerFeatures):
    row = pd.DataFrame([customer.model_dump()])

    # Keep all dummy columns at inference; drop_first=True would remove the only
    # category present in this single-row request. reindex then matches the request
    # to the training columns, drops any extra baseline column, and fills missing
    # dummy columns with 0.
    row = pd.get_dummies(row, columns=["contract_type"], drop_first=False)

    # Match the training columns exactly.
    row = row.reindex(columns=feature_cols, fill_value=0)

    # Use medians learned from training data.
    row = row.fillna(train_medians)

    row_scaled = scaler.transform(row)
    churn_probability = float(model.predict_proba(row_scaled)[0, 1])

    return PredictionResponse(
        churn_probability=round(churn_probability, 4),
        predicted_label=int(churn_probability >= THRESHOLD),
    )
