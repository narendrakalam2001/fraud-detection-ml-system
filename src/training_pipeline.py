# ============================================================
# TRAINING PIPELINE — Credit Card Fraud Detection ML System
# ============================================================

import os
import sys
import json
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline        import Pipeline

from src.config                         import RANDOM_STATE, MODEL_DIR
from src.data_loader                    import validate_input_data, load_and_engineer
from src.leakage_check                  import detect_leakage
from src.anomaly_detection              import anomaly_filter
from src.sampling                       import fast_training_sample
from src.preprocessing                  import build_preprocessors
from src.model_tuning                   import scaled_models, unscaled_models, tune_models
from src.neural_net                     import train_mlp_pipeline
from src.evaluation                     import evaluate_models
from src.model_loader                   import run_challenger_comparison
from feature_store.fraud_features       import build_fraud_features
from graph_detection.fraud_graph_detection import compute_graph_risk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training():

    # ── Data path — env variable or relative fallback ─────────
    DATA_PATH = os.getenv(
        "FRAUD_DATA_PATH",
        os.path.join("data", "creditcard.csv")
    )

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'. "
            "Set FRAUD_DATA_PATH environment variable or place "
            "creditcard.csv in the data/ folder."
        )

    logger.info("Loading dataset from: %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # ── Validate + engineer ───────────────────────────────────
    df = validate_input_data(df)
    df = load_and_engineer(df)
    df = build_fraud_features(df)
    df = compute_graph_risk(df)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # ── Train / test split ────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(
        "Split  |  train=%d  test=%d  fraud_rate_train=%.4f",
        len(X_tr), len(X_te), y_tr.mean()
    )

    # ── Leakage check (before any preprocessing) ──────────────
    leakage_warnings = detect_leakage(X_tr, y_tr)
    if leakage_warnings:
        logger.warning("Leakage warnings: %s", leakage_warnings)

    # ── Anomaly filter ────────────────────────────────────────
    X_tr, X_te = anomaly_filter(X_tr, X_te)

    # ─────────────────────────────────────────────────────────
    # STAGE 1 — Fast screening on balanced sample
    # ─────────────────────────────────────────────────────────
    X_fast, y_fast         = fast_training_sample(X_tr, y_tr)
    pre_s_fast, pre_u_fast, _ = build_preprocessors(X_fast)

    scaled_fast,   _ = tune_models(scaled_models,   pre_s_fast, X_fast, y_fast)
    unscaled_fast, _ = tune_models(unscaled_models, pre_u_fast, X_fast, y_fast)

    # ── Select top 3 from each group ──────────────────────────
    top_scaled   = list(scaled_fast.keys())[:3]
    top_unscaled = list(unscaled_fast.keys())[:3]

    # ─────────────────────────────────────────────────────────
    # STAGE 2 — Retrain top models on full training data
    # ─────────────────────────────────────────────────────────
    pre_scaled_full, pre_unscaled_full, _ = build_preprocessors(X_tr)

    scaled_final   = {}
    unscaled_final = {}

    for name in top_scaled:
        clf  = scaled_models[name][0]
        pipe = Pipeline([("pre", pre_scaled_full), ("classifier", clf)])
        pipe.fit(X_tr, y_tr)
        scaled_final[name] = pipe

    for name in top_unscaled:
        clf  = unscaled_models[name][0]
        pipe = Pipeline([("pre", pre_unscaled_full), ("classifier", clf)])
        pipe.fit(X_tr, y_tr)
        unscaled_final[name] = pipe

    mlp_model = train_mlp_pipeline(X_tr, y_tr, pre_scaled_full)

    all_models = {
        **scaled_final,
        **unscaled_final,
        "NeuralNet": mlp_model,
    }

    # ── Evaluate + save model card ────────────────────────────
    best_model, thr, best_name, model_path, metrics = evaluate_models(
        all_models, X_tr, X_te, y_tr, y_te
    )

    # ── Champion vs Challenger ────────────────────────────────
    result = run_challenger_comparison(
        challenger_name       = best_name,
        challenger_f1         = metrics["test_f1"],
        challenger_roc_auc    = metrics["roc_auc"],
        challenger_gap        = metrics["train_test_gap"],
        challenger_model_path = model_path,
        challenger_threshold  = thr,
    )

    logger.info(
        "Challenger result: %s — %s",
        result["decision"], result["reason"]
    )
    logger.info("Training pipeline complete ✅")


if __name__ == "__main__":
    run_training()