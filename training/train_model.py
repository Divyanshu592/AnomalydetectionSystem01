import os
import numpy as np
import pandas as pd

from preprocessing.clean import basic_cleaning
from preprocessing.normalization import Normalizer
from preprocessing.windowing import create_windows
from models.lstm_autoencoder import build_lstm_autoencoder
from config.logger import get_logger

logger = get_logger("training")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_pipeline(csv_path="data/sample/train.csv", window_size=30):
    """
    Basic training pipeline skeleton.
    - read CSV
    - clean
    - normalize
    - create windows
    - train LSTM Autoencoder
    - save model + scaler
    """

    logger.info("📌 Loading dataset...")
    df = pd.read_csv(csv_path)

    logger.info("🧹 Cleaning dataset...")
    df = basic_cleaning(df)

    # Example: selecting numeric columns only
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    X = numeric_df.values.astype(np.float32)

    logger.info("📏 Normalizing dataset...")
    normalizer = Normalizer()
    X_scaled = normalizer.fit_transform(X)
    normalizer.save(os.path.join(MODEL_DIR, "scaler.pkl"))

    logger.info("🪟 Creating windows...")
    X_windows = create_windows(X_scaled, window_size=window_size)

    logger.info(f"✅ Windows shape: {X_windows.shape}")

    window_size, n_features = X_windows.shape[1], X_windows.shape[2]

    logger.info("🧠 Building LSTM Autoencoder...")
    model = build_lstm_autoencoder(window_size=window_size, n_features=n_features)

    logger.info("🏋️ Training model...")
    model.fit(
        X_windows, X_windows,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    logger.info("💾 Saving model...")
    model.save(os.path.join(MODEL_DIR, "lstm_autoencoder.h5"))

    logger.info("🎉 Training completed successfully!")

if __name__ == "__main__":
    train_pipeline()
