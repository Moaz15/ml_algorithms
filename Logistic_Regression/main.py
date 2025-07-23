import os
import numpy as np
import joblib

from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

theta = np.load(os.path.join(BASE_DIR, "theta.npy"))
bias = np.load(os.path.join(BASE_DIR, "bias.npy"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

app = FastAPI()

class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict_ctr(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    z = np.dot(X_scaled, theta) + bias
    prob = 1 / (1 + np.exp(-z))
    prediction = int(prob >= 0.5)
    return {
        "probability": float(prob),
        "prediction": prediction
    }