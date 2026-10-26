
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load the saved ML pipeline
MODEL_PATH = Path(__file__).parent / "models" / "logistic_regression_churn.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Customer Churn Prediction API")


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict_churn(customer: Customer):
    customer_data = pd.DataFrame([customer.model_dump()])

    prediction = int(model.predict(customer_data)[0])
    probability = float(model.predict_proba(customer_data)[0][1])

    return {
        "churn_prediction": prediction,
        "churn_label": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 4),
    }
