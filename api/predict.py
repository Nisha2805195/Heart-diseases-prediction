
import os
import pickle
import numpy as np

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Heart Disease Prediction")

# Project root folder
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Model file paths
MODEL_PATH = os.path.join(
    BASE_DIR, "model", "heart_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR, "model", "scaler.pkl"
)

# Load trained model and scaler
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)


# Input data format
class HeartData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


# Main dashboard
@app.get("/")
def dashboard():
    return FileResponse(
        os.path.join(BASE_DIR, "index.html")
    )


# Prediction page
@app.get("/predict.html")
def prediction_page():
    return FileResponse(
        os.path.join(BASE_DIR, "predict.html")
    )


# Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# Prediction API
@app.post("/api/predict")
def predict(data: HeartData):

    # Prepare input using the same feature order as training
    input_data = np.array([[
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,
        data.restecg,
        data.thalach,
        data.exang,
        data.oldpeak,
        data.slope,
        data.ca,
        data.thal
    ]])

    # Apply scaler
    input_scaled = scaler.transform(input_data)

    # Predict class
    prediction = int(model.predict(input_scaled)[0])

    # Get class probabilities using model's class labels
    probabilities = model.predict_proba(input_scaled)[0]
    classes = list(model.classes_)

    no_disease_index = classes.index(0)
    disease_index = classes.index(1)

    probability_no_disease = float(
        probabilities[no_disease_index] * 100
    )
    probability_disease = float(
        probabilities[disease_index] * 100
    )

    if prediction == 1:
        result = "Higher likelihood of heart disease"
        probability = probability_disease
    else:
        result = "Lower likelihood of heart disease"
        probability = probability_no_disease

    print("MODEL PROBABILITIES:", probabilities)

    return {
        "prediction": prediction,
        "result": result,
        "probability": round(probability, 2),
        "probability_no_disease": round(
            probability_no_disease, 2
        ),
        "probability_disease": round(
            probability_disease, 2
        )
    }