from config.logger import get_logger

logger = get_logger("alerts")

def send_alert(result: dict):
    """
    Later we will connect:
    - Email alert
    - Webhook alert
    - SMS alert

    For now, only logging.
    """
    logger.warning(f"🚨 ALERT TRIGGERED: {result}")
