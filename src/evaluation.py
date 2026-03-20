import warnings
warnings.filterwarnings("ignore")

import os, time, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from datetime import datetime

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import *
from sklearn.calibration import calibration_curve

from src.metrics import (
    tune_threshold,
    recall_at_k,
    lift_at_k,
    ks_statistic,
    psi,
    cost_sensitive_evaluation
)

from src.config import RANDOM_STATE, CV_FOLDS, N_JOBS, MODEL_DIR
from src.rule_engine import rule_engine

try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except:
    MLFLOW_AVAILABLE = False


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_models(all_models, X_tr, X_te, y_tr, y_te):

    rows = []

    # ============================================================
    # MODEL SUMMARY
    # ============================================================

    for name, pipe in all_models.items():

        y_prob = pipe.predict_proba(X_te)[:,1]
        thr = tune_threshold(y_te,y_prob)
        y_pred = (y_prob>=thr).astype(int)

        cv = cross_val_score(
            pipe,
            X_tr,
            y_tr,
            scoring='average_precision',
            cv=StratifiedKFold(CV_FOLDS,shuffle=True,random_state=RANDOM_STATE),
            n_jobs=N_JOBS
        )

        rows.append({
            "model":name,
            "cv_mean_pr_auc":cv.mean(),
            "cv_std_pr_auc":cv.std(),
            "precision":precision_score(y_te,y_pred),
            "recall":recall_score(y_te,y_pred),
            "f1":f1_score(y_te,y_pred),
            "roc_auc":roc_auc_score(y_te,y_prob),
            "pr_auc":average_precision_score(y_te,y_prob),
            "recall@5%":recall_at_k(y_te,y_prob,0.05),
            "lift@5%":lift_at_k(y_te,y_prob,0.05),
            "ks":ks_statistic(y_te,y_prob)
        })

    summary = pd.DataFrame(rows).sort_values(
        ["cv_mean_pr_auc","recall"],ascending=False
    ).reset_index(drop=True)

    summary.to_csv(os.path.join(MODEL_DIR,"model_experiment_results.csv"), index=False)

    print("\nALL MODELS SUMMARY\n")
    print(summary.head(10))

    # ============================================================
    # BEST MODEL
    # ============================================================

    best_name = summary.loc[0,"model"]
    best_model = all_models[best_name]

    print(f"\nBEST MODEL: {best_name}")

    y_prob = best_model.predict_proba(X_te)[:,1]
    thr = tune_threshold(y_te,y_prob)
    y_pred = (y_prob>=thr).astype(int)

    # ============================================================
    # RULE ENGINE
    # ============================================================

    decisions = rule_engine(X_te, y_prob, thr)

    print("\nRULE ENGINE DECISIONS")
    print(pd.Series(decisions).value_counts())

    # ============================================================
    # COST EVALUATION
    # ============================================================

    fraud_loss, review_cost, total_loss = cost_sensitive_evaluation(
        X_te, y_te, y_pred, y_prob
    )

    print("\nTotal Estimated Loss:", total_loss)
    
    # ============================================================
    # SAVE MONITORING SCORES
    # ============================================================

    monitor_df = pd.DataFrame({
        "score": y_prob,
        "decision": decisions,
        "label": y_te
    })

    monitor_df.to_csv(
        os.path.join(MODEL_DIR,"monitor_scores.csv"),
        index=False
    )

    print("Monitoring scores saved")
    
    # ============================================================
    # FEATURE DRIFT REPORT
    # ============================================================

    print("\nChecking feature drift")

    feature_drift = {}

    for col in X_tr.columns:

        train_mean = X_tr[col].mean()
        test_mean = X_te[col].mean()

        drift = abs(train_mean - test_mean)

        feature_drift[col] = drift

    drift_df = pd.DataFrame({
        "feature": feature_drift.keys(),
        "drift_score": feature_drift.values()
    }).sort_values("drift_score", ascending=False)

    drift_df.to_csv(
        os.path.join(MODEL_DIR,"feature_drift_report.csv"),
        index=False
    )

    print("Feature drift report saved")

    # ============================================================
    # SAVE MODEL
    # ============================================================

    model_version = "v1"

    model_path = os.path.join(MODEL_DIR, f"fraud_model_{model_version}.joblib")

    joblib.dump(best_model, model_path)

    metadata = {

        "model_version": model_version,
        "best_model": best_name,
        "threshold": float(thr),
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    meta_path = os.path.join(MODEL_DIR,f"fraud_model_{model_version}_metadata.json")

    with open(meta_path,"w") as f:
        json.dump(metadata,f,indent=4)

    print("Model saved:", model_path)

    return best_model, thr