from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
import time
import json
import os
import random

from src.config import MODEL_DIR
from services.prediction_service import predict_transaction
from src.model_loader import load_latest_model

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection API")

# ==============================
# LOAD MODEL
# ==============================
try:
    model, threshold = load_latest_model()
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    model = None
    threshold = 0.5

metadata_path = os.path.join(MODEL_DIR, "fraud_model_v1_metadata.json")

# ==============================
# INPUT SCHEMA
# ==============================

class Transaction(BaseModel):
    Time: float
    Amount: float
        
# ==============================
# HOME ROUTE (ADD THIS HERE)
# ==============================

@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is live 🚀",
        "docs": "/docs",
        "health": "/health"
    }

# ==============================
# HEALTH CHECK
# ==============================

@app.get("/health")
def health():
    return {"status": "API running"}

# ==============================
# MODEL INFO
# ==============================

@app.get("/model_info")
def model_info():

    with open(metadata_path) as f:
        metadata = json.load(f)

    return metadata

# ==============================
# SINGLE PREDICTION
# ==============================

@app.post("/predict")
def predict(transaction: Transaction):

    start = time.time()

    # ==============================
    # ADD MISSING FEATURES (V1–V28)
    # ==============================

    input_data = transaction.dict()

    for i in range(1, 29):
        input_data[f"V{i}"] = random.uniform(-5, 5)

    # ==============================
    # PREDICTION
    # ==============================

    prob, decision = predict_transaction(
        model,
        input_data,
        threshold
    )
    
    # ==============================
    # PREDICTION LOGGING
    # ==============================

    log_record = {
        "timestamp": time.time(),
        "amount": float(transaction.Amount),
        "fraud_probability": float(prob),
        "decision": decision
    }

    log_path = "logs/prediction_logs.csv"

    if not os.path.exists("logs"):
        os.makedirs("logs")

    log_df = pd.DataFrame([log_record])

    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)

    latency = time.time() - start

    logger.info(f"Fraud probability={prob:.4f}, decision={decision}")

    return {
        "fraud_probability": float(prob),
        "decision": decision,
        "latency_seconds": latency
    }