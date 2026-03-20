import pandas as pd
import numpy as np

from feature_store.fraud_features import build_fraud_features
from graph_detection.fraud_graph_detection import compute_graph_risk


def prepare_features(df):

    df["hour"] = (df["Time"] // 3600) % 24
    df["day"] = df["Time"] // (3600 * 24)

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    df["Amount_original"] = df["Amount"]
    df["Amount"] = np.log1p(df["Amount"])

    df.drop(columns=["Time","hour"], inplace=True)

    df = build_fraud_features(df)

    try:
        df = compute_graph_risk(df)
    except:
        df["graph_risk_score"] = 0
    return df


def predict_transaction(model, transaction, threshold):

    df = pd.DataFrame([transaction])

    # --------------------------------------------------
    # ADD MISSING TRAINING FEATURES (FOR API INFERENCE)
    # --------------------------------------------------

    for i in range(1, 29):
        col = f"V{i}"
        if col not in df.columns:
            df[col] = 0

    if "anomaly_score" not in df.columns:
        df["anomaly_score"] = 0

    # --------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------

    df = prepare_features(df)

    prob = model.predict_proba(df)[0][1]

    decision = "BLOCK" if prob > threshold else "APPROVE"

    return prob, decision