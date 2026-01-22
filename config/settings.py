"""
settings.py - central config loader
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # App
    APP_NAME = os.getenv("APP_NAME", "Anomaly Detection System")
    ENV = os.getenv("ENV", "dev")

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "live_sensor_data")
    KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "anomaly-consumer")

    # InfluxDB
    INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
    INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "")
    INFLUX_ORG = os.getenv("INFLUX_ORG", "")
    INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "")

settings = Settings()
