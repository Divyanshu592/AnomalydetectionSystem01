from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from config.settings import settings
from config.logger import get_logger

logger = get_logger("influx-writer")


class InfluxWriter:
    def __init__(self):
        self.client = InfluxDBClient(
            url=settings.INFLUX_URL,
            token=settings.INFLUX_TOKEN,
            org=settings.INFLUX_ORG
        )
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        logger.info("✅ InfluxDB client initialized")

    def write_sensor_data(self, measurement: str, data: dict):
        """
        Example data expected:
        {
          "device_id": "sensor_01",
          "value": 12.3,
          "timestamp": "2026-01-22T10:00:00Z"
        }
        """

        point = Point(measurement)

        # tags (identifiers)
        if "device_id" in data:
            point = point.tag("device_id", str(data["device_id"]))

        # fields (values)
        for key, value in data.items():
            if key not in ["device_id", "timestamp"]:
                if isinstance(value, (int, float, bool, str)):
                    point = point.field(key, value)

        # optional timestamp handling
        # InfluxDB can use server time if you don't specify
        self.write_api.write(bucket=settings.INFLUX_BUCKET, record=point)

        logger.info(f"📌 Written to Influx: {measurement} -> {data}")
