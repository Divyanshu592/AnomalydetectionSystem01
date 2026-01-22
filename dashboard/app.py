import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Anomaly Dashboard", layout="wide")

st.title("📊 Anomaly Detection Dashboard")

st.info("This is a starter dashboard. Next we will connect live InfluxDB queries.")

# Demo table placeholder (later we fetch from InfluxDB)
demo_data = [
    {"time": datetime.utcnow().isoformat(), "device_id": "sensor_01", "value": 12.3, "score": 0.02, "anomaly": False},
    {"time": datetime.utcnow().isoformat(), "device_id": "sensor_01", "value": 95.0, "score": 0.12, "anomaly": True},
]

df = pd.DataFrame(demo_data)

st.subheader("Latest Anomaly Results")
st.dataframe(df, use_container_width=True)
