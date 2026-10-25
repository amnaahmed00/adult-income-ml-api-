
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.adult_income.data import FEATURE_COLUMNS


# Load the saved models
MODEL_DIR = Path(__file__).resolve().parent / "models"

MODEL_FILES = {
    "dummy": "dummy_baseline.joblib",
    "logistic_regression": "logistic_regression.joblib",
    "random_forest": "random_forest.joblib",
    "lightgbm": "lightgbm.joblib",
}

models = {
    name: joblib.load(MODEL_DIR / filename)
    for name, filename in MODEL_FILES.items()
}

app = FastAPI(
    title="Adult Income Prediction API",
    description="Predict income class using one of four trained ML models.",
    version="1.0.0",
)


class IncomeInput(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


@app.get("/")
def home():
    return {
        "message": "Adult Income Prediction API is running.",
        "available_models": list(MODEL_FILES.keys()),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": list(models.keys())}


@app.post("/predict/{model_name}")
def predict(model_name: str, data: IncomeInput):
    if model_name not in models:
        return {
            "error": "Unknown model.",
            "available_models": list(models.keys()),
        }

    # Convert API field names to the exact feature names expected by the pipeline
    feature_mapping = {
        "age": "age",
        "workclass": "workclass",
        "fnlwgt": "fnlwgt",
        "education": "education",
        "education_num": "education_num",
        "marital_status": "marital_status",
        "occupation": "occupation",
        "relationship": "relationship",
        "race": "race",
        "sex": "sex",
        "capital_gain": "capital_gain",
        "capital_loss": "capital_loss",
        "hours_per_week": "hours_per_week",
        "native_country": "native_country",
    }

    input_data = data.model_dump()
    row = {
        feature_mapping[key]: value
        for key, value in input_data.items()
    }

    # Match the original training feature order
    input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    model = models[model_name]
    prediction = model.predict(input_df)[0]

    result = {
        "model": model_name,
        "prediction": str(prediction),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        result["probabilities"] = {
            str(label): round(float(probability), 4)
            for label, probability in zip(
                model.classes_, probabilities
            )
        }

    return result
