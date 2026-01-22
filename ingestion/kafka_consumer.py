"""kafka_consumer.py - ingestion module"""

import json
from kafka import KafkaConsumer
from config.settings import settings
from config.logger import get_logger

logger = get_logger("kafka-consumer")

def create_consumer():
    consumer = KafkaConsumer(
        settings.KAFKA_TOPIC,
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id=settings.KAFKA_GROUP_ID,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode("utf-8"))
    )
    logger.info("✅ Kafka consumer created")
    return consumer


def consume_messages(on_message):
    """
    on_message: callback function to process each message.
    """
    consumer = create_consumer()

    logger.info(f"📥 Listening to topic: {settings.KAFKA_TOPIC}")
    for msg in consumer:
        data = msg.value
        logger.info(f"Received: {data}")
        on_message(data)
