import numpy as np
from tensorflow.keras.models import load_model
import joblib

from preprocessing.windowing import create_windows

def reconstruction_error(model, X):
    preds = model.predict(X, verbose=0)
    return np.mean(np.square(preds - X), axis=(1, 2))

def evaluate_sample(X_scaled, window_size=30):
    model = load_model("models/lstm_autoencoder.h5")
    X_windows = create_windows(X_scaled, window_size)
    errors = reconstruction_error(model, X_windows)
    return errors
