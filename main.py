import numpy as np
from collections import deque
from datetime import datetime

from ingestion.kafka_consumer import consume_messages
from database.influx_writer import InfluxWriter
from inference.predict import AnomalyPredictor
from config.logger import get_logger
from alerts.alert_manager import send_alert

from inference.load_threshold import load_threshold
THRESHOLD = load_threshold()

logger = get_logger("main")

writer = InfluxWriter()

# Load Predictor (model + scaler)
predictor = AnomalyPredictor(window_size=30)

# Store last N data points for window-based inference
buffer = deque(maxlen=60)  # keep some extra points



def process_message(data: dict):
    """
    Expected incoming Kafka data example:
    {
      "device_id": "sensor_01",
      "value": 12.3,
      "timestamp": "2026-01-22T10:00:00Z"
    }
    """

    # 1) Store raw data
    writer.write_sensor_data(measurement="live_data", data=data)

    # 2) Add numeric features into buffer
    # Here only using "value" as a feature.
    # Later can extend to multiple sensor fields.
    if "value" not in data:
        return

    buffer.append([float(data["value"])])

    # 3) Do inference only when enough points exist
    if len(buffer) < predictor.window_size:
        return

    X = np.array(buffer, dtype=np.float32)  # shape: (n_samples, 1)

    errors = predictor.score(X)
    latest_error = float(errors[-1])

    is_anomaly = latest_error > THRESHOLD

    result = {
        "device_id": data.get("device_id", "unknown"),
        "value": float(data["value"]),
        "anomaly_score": latest_error,
        "is_anomaly": bool(is_anomaly),
        "timestamp": data.get("timestamp", datetime.utcnow().isoformat())
    }

    # 4) Store anomaly result
    writer.write_sensor_data(measurement="anomaly_result", data=result)

    if is_anomaly:
        logger.warning(f"🚨 ANOMALY DETECTED: {result}")
        send_alert(result)
    else:
        logger.info(f"✅ Normal: score={latest_error:.5f}")


if __name__ == "__main__":
    logger.info("🚀 Starting Anomaly Detection Pipeline (Real-Time)...")
    consume_messages(on_message=process_message)
