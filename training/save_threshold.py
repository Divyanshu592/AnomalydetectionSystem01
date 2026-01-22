import json
import numpy as np
from tensorflow.keras.models import load_model
import joblib

from preprocessing.windowing import create_windows
from inference.anomaly_score import reconstruction_error
from inference.threshold import percentile_threshold
from config.logger import get_logger

logger = get_logger("save-threshold")


def generate_and_save_threshold(
    train_csv_scaled: np.ndarray,
    window_size: int = 30,
    percentile: float = 95,
    output_path: str = "models/threshold.json"
):
    """
    train_csv_scaled : training data already scaled
    """

    logger.info("📌 Loading trained model...")
    model = load_model("models/lstm_autoencoder.h5")

    logger.info("🪟 Creating training windows...")
    X_windows = create_windows(train_csv_scaled, window_size)

    logger.info("🧠 Predicting reconstruction...")
    preds = model.predict(X_windows, verbose=0)

    logger.info("📌 Computing reconstruction errors...")
    errors = reconstruction_error(X_windows, preds)

    thr = percentile_threshold(errors, percentile=percentile)

    data = {
        "threshold": thr,
        "percentile": percentile,
        "window_size": window_size
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    logger.info(f"✅ Saved threshold to {output_path}: {thr:.6f}")
    return thr
