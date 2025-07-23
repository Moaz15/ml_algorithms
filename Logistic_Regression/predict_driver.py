import numpy as np
import joblib

theta = np.load("theta.npy")
bias = np.load("bias.npy")
scaler = joblib.load("scaler.pkl")

def predict_prob(X):
    z = np.dot(X, theta) + bias
    return 1 / (1 + np.exp(-z))

def predict(X, threshold=0.5):
    X_scaled = scaler.transform(X)
    prob = predict_prob(X_scaled)
    return (prob >= threshold).astype(int), prob

