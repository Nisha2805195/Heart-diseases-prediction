import os
import pickle
import numpy as np

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel


app = FastAPI(title="Heart Disease Prediction")


# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Model files
MODEL_PATH = os.path.join(BASE_DIR, "model", "heart_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")


# Load model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# Load scaler
with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)


# Input data
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


# Show Dashboard
@app.get("/")
def dashboard():
    return FileResponse(
        os.path.join(BASE_DIR, "index.html")
    )


# Prediction
@app.post("/api/predict")
def predict(data: HeartData):

    input_data = np.array([
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
    ]).reshape(1, -1)

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability
    probabilities = model.predict_proba(input_scaled)[0]
    probability = probabilities[int(prediction)] * 100

    if prediction == 1:
        result = "Higher likelihood of heart disease"
    else:
        result = "Lower likelihood of heart disease"

    return {
        "prediction": int(prediction),
        "result": result,
        "probability": round(float(probability), 2)
    }