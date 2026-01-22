import json

def load_threshold(path="models/threshold.json", fallback=0.05):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data["threshold"])
    except Exception:
        return float(fallback)
