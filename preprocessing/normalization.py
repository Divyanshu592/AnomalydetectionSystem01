import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

class Normalizer:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray):
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray):
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray):
        return self.scaler.fit_transform(X)

    def save(self, path="models/scaler.pkl"):
        joblib.dump(self.scaler, path)

    def load(self, path="models/scaler.pkl"):
        self.scaler = joblib.load(path)
        return self
