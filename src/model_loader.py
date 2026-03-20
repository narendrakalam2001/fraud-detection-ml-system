import json
import os
import joblib
from src.config import MODEL_DIR


def load_latest_model():

    registry_path = os.path.join(MODEL_DIR, "latest_model.json")

    with open(registry_path) as f:
        registry = json.load(f)

    model_path = os.path.join(MODEL_DIR, registry["model_name"])

    model = joblib.load(model_path)

    threshold = registry.get("threshold", 0.8)

    return model, threshold