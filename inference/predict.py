import numpy as np
import joblib
from tensorflow.keras.models import load_model

from preprocessing.windowing import create_windows
from inference.anomaly_score import reconstruction_error

class AnomalyPredictor:
    def __init__(
        self,
        model_path="models/lstm_autoencoder.h5",
        scaler_path="models/scaler.pkl",
        window_size=30
    ):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.window_size = window_size

    def preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        X shape: (n_samples, n_features)
        """
        X_scaled = self.scaler.transform(X)
        return X_scaled

    def score(self, X: np.ndarray):
        """
        Returns anomaly errors per window.
        """
        X_scaled = self.preprocess(X)
        X_windows = create_windows(X_scaled, self.window_size)

        preds = self.model.predict(X_windows, verbose=0)
        errors = reconstruction_error(X_windows, preds)

        return errors
