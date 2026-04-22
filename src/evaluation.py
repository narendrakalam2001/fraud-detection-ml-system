# ============================================================
# EVALUATION — Credit Card Fraud Detection ML System
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os, time, json, logging
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss
)

from src.metrics import (
    tune_threshold, recall_at_k, lift_at_k,
    ks_statistic, psi, cost_sensitive_evaluation
)
from src.config import (
    RANDOM_STATE, CV_FOLDS, N_JOBS, MODEL_DIR,
    PSI_MODERATE, PSI_HIGH
)
from src.rule_engine import rule_engine

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================
# BUILD MODEL CARD
# ============================================================

def build_model_card(
    selected_name, train_size, test_size, fraud_rate_train,
    metrics, threshold, cost_result, decision_counts,
    version="v1", fi_dict=None, shap_dict=None,
):
    card = {
        "model_version":   str(version),
        "model_name":      str(selected_name),
        "trained_at":      time.strftime("%Y-%m-%d %H:%M:%S"),
        "project":         "Credit Card Fraud Detection System",
        "dataset": {
            "train_size":       int(train_size),
            "test_size":        int(test_size),
            "fraud_rate_train": round(float(fraud_rate_train), 4),
        },
        "metrics": {
            "test_f1":        round(float(metrics.get("test_f1",        0)), 4),
            "test_precision": round(float(metrics.get("precision",      0)), 4),
            "test_recall":    round(float(metrics.get("recall",         0)), 4),
            "roc_auc":        round(float(metrics.get("roc_auc",        0)), 4),
            "pr_auc":         round(float(metrics.get("pr_auc",         0)), 4),
            "ks_statistic":   round(float(metrics.get("ks",             0)), 4),
            "recall_at_5pct": round(float(metrics.get("recall_at_5",   0)), 4),
            "lift_at_5pct":   round(float(metrics.get("lift_at_5",     0)), 4),
            "brier_score":    round(float(metrics.get("brier",          0)), 4),
            "train_test_gap": round(float(metrics.get("train_test_gap", 0)), 4),
        },
        "threshold":       round(float(threshold), 4),
        "risk_decisions":  {str(k): int(v) for k, v in decision_counts.items()},
        "cost_evaluation": {str(k): float(v) for k, v in cost_result.items()},
    }
    if fi_dict   is not None:
        card["feature_importances"] = fi_dict
    if shap_dict is not None:
        card["shap_top_features"] = shap_dict
    return card


# ============================================================
# SHAP — version-safe, ExtraTrees-safe
# ============================================================

def _safe_shap(best_model, X_te):
    """
    Computes mean absolute SHAP values.
    Handles: old shap (list output), new shap (Explanation object),
    sparse matrices, and all edge cases.
    Returns dict {feature_name: mean_abs_shap} or None.
    """
    try:
        steps   = list(best_model.named_steps.keys())
        clf     = best_model.named_steps[steps[-1]]
        pipe_tr = best_model[:-1]

        X_arr = pipe_tr.transform(X_te)
        if hasattr(X_arr, "toarray"):
            X_arr = X_arr.toarray()
        X_arr = np.array(X_arr, dtype=float)

        X_sample  = X_arr[:100]
        explainer = shap.TreeExplainer(clf)

        # Use legacy .shap_values() — consistent across all shap versions
        sv = explainer.shap_values(X_sample)

        # Normalize output to a 2D numpy array (samples x features)
        if isinstance(sv, list):
            # Binary classification: [class0, class1]
            arr = np.array(sv[1], dtype=float)
        elif hasattr(sv, "values"):
            # Explanation object (shap >= 0.40)
            arr = np.array(sv.values, dtype=float)
            if arr.ndim == 3:
                arr = arr[:, :, 1]
        else:
            arr = np.array(sv, dtype=float)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        means = np.abs(arr).mean(axis=0)

        try:
            names = list(pipe_tr.get_feature_names_out())
        except Exception:
            names = [f"f{i}" for i in range(len(means))]

        out = dict(sorted(
            zip(names[:len(means)], [round(float(v), 4) for v in means]),
            key=lambda x: x[1], reverse=True
        )[:15])

        logger.info("SHAP OK — top feature: %s = %.4f",
                    list(out.keys())[0], list(out.values())[0])
        return out

    except Exception as exc:
        logger.warning("SHAP skipped: %s", exc)
        return None


# ============================================================
# EVALUATE MODELS
# ============================================================

def evaluate_models(all_models, X_tr, X_te, y_tr, y_te):

    rows = []

    for name, pipe in all_models.items():
        y_prob_te = pipe.predict_proba(X_te)[:, 1]
        thr       = tune_threshold(y_te, y_prob_te)
        y_pred    = (y_prob_te >= thr).astype(int)

        cv = cross_val_score(
            pipe, X_tr, y_tr,
            scoring = "average_precision",
            cv      = StratifiedKFold(CV_FOLDS, shuffle=True,
                                      random_state=RANDOM_STATE),
            n_jobs  = N_JOBS,
        )

        rows.append({
            "model":          str(name),
            "cv_pr_auc_mean": round(float(cv.mean()), 4),
            "cv_pr_auc_std":  round(float(cv.std()),  4),
            "precision":      round(float(precision_score(y_te, y_pred)),            4),
            "recall":         round(float(recall_score(y_te, y_pred)),                4),
            "test_f1":        round(float(f1_score(y_te, y_pred)),                    4),
            "roc_auc":        round(float(roc_auc_score(y_te, y_prob_te)),            4),
            "pr_auc":         round(float(average_precision_score(y_te, y_prob_te)),  4),
            "ks":             round(float(ks_statistic(y_te, y_prob_te)),             4),
            "recall_at_5":    round(float(recall_at_k(y_te, y_prob_te, 0.05)),       4),
            "lift_at_5":      round(float(lift_at_k(y_te, y_prob_te, 0.05)),         4),
            "brier":          round(float(brier_score_loss(y_te, y_prob_te)),         4),
            "train_test_gap": round(float(abs(pipe.score(X_tr, y_tr)
                                             - pipe.score(X_te, y_te))),              4),
            "threshold":      round(float(thr), 4),
        })

    summary = (
        pd.DataFrame(rows)
        .sort_values(["cv_pr_auc_mean", "recall"], ascending=False)
        .reset_index(drop=True)
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    summary.to_csv(
        os.path.join(MODEL_DIR, "model_experiment_results.csv"), index=False
    )

    print("\n-- ALL MODELS SUMMARY --")
    print(summary.to_string(index=False))

    best_name  = summary.loc[0, "model"]
    best_model = all_models[best_name]
    best_row   = summary.loc[0]
    print(f"\n BEST MODEL: {best_name}")

    y_prob = best_model.predict_proba(X_te)[:, 1]
    thr    = tune_threshold(y_te, y_prob)
    y_pred = (y_prob >= thr).astype(int)

    # Rule engine
    decisions  = rule_engine(X_te, y_prob, thr)
    dec_counts = {str(k): int(v)
                  for k, v in pd.Series(decisions).value_counts().items()}
    print("\nRule Engine Decisions:")
    print(pd.Series(decisions).value_counts().to_string())

    # Cost evaluation
    fraud_loss, review_cost, total_loss = cost_sensitive_evaluation(
        X_te, y_te, y_pred, y_prob
    )
    cost_result = {
        "estimated_fraud_loss": round(float(fraud_loss),  2),
        "review_cost":          round(float(review_cost), 2),
        "total_estimated_loss": round(float(total_loss),  2),
    }
    print(f"\nCost Evaluation:")
    print(f"  Fraud loss   : ${fraud_loss:,.2f}")
    print(f"  Review cost  : ${review_cost:,.2f}")
    print(f"  Total loss   : ${total_loss:,.2f}")

    # Monitoring scores
    pd.DataFrame({
        "score":    y_prob,
        "decision": decisions,
        "label":    y_te.values if hasattr(y_te, "values") else y_te,
    }).to_csv(os.path.join(MODEL_DIR, "monitor_scores.csv"), index=False)
    logger.info("Monitoring scores saved")

    # PSI drift
    logger.info("Computing feature drift (PSI)...")
    drift = []
    for col in X_tr.columns:
        v = float(psi(X_tr[col].values, X_te[col].values))
        drift.append({
            "feature":     str(col),
            "drift_score": round(v, 4),
            "status":      ("CRITICAL" if v >= PSI_HIGH else
                            "MODERATE" if v >= PSI_MODERATE else "OK"),
        })
    (pd.DataFrame(drift)
       .sort_values("drift_score", ascending=False)
       .reset_index(drop=True)
       .to_csv(os.path.join(MODEL_DIR, "feature_drift_report.csv"), index=False))
    logger.info("Feature drift report saved")

    # SHAP
    shap_dict = _safe_shap(best_model, X_te) if SHAP_AVAILABLE else None

    # Feature importances
    fi_dict = None
    try:
        steps = list(best_model.named_steps.keys())
        clf   = best_model.named_steps[steps[-1]]
        if hasattr(clf, "feature_importances_"):
            imps = clf.feature_importances_
            try:
                fnames = list(best_model[:-1].get_feature_names_out())
            except Exception:
                fnames = [f"f{i}" for i in range(len(imps))]
            fi_dict = dict(sorted(
                zip(fnames[:len(imps)], [round(float(v), 4) for v in imps]),
                key=lambda x: x[1], reverse=True
            ))
    except Exception as exc:
        logger.warning("Feature importances failed: %s", exc)

    # Build model card
    metrics_dict = {k: float(best_row[k]) for k in [
        "test_f1", "precision", "recall", "roc_auc", "pr_auc",
        "ks", "recall_at_5", "lift_at_5", "brier", "train_test_gap",
    ]}

    card = build_model_card(
        selected_name    = best_name,
        train_size       = len(X_tr),
        test_size        = len(X_te),
        fraud_rate_train = float(y_tr.mean()),
        metrics          = metrics_dict,
        threshold        = thr,
        cost_result      = cost_result,
        decision_counts  = dec_counts,
        fi_dict          = fi_dict,
        shap_dict        = shap_dict,
    )

    version   = "v1"
    card_path = os.path.join(MODEL_DIR, f"model_card_{best_name}_{version}.json")
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2, default=str)
    logger.info("Model card saved -> %s", card_path)

    # MLflow
    if MLFLOW_AVAILABLE:
        try:
            with mlflow.start_run(run_name=best_name):
                mlflow.log_params({
                    "model":     str(best_name),
                    "threshold": round(float(thr), 4),
                })
                mlflow.log_metrics({k: float(v) for k, v in metrics_dict.items()})
                mlflow.sklearn.log_model(best_model, name="model")
            logger.info("MLflow run logged")
        except Exception as exc:
            logger.warning("MLflow logging failed: %s", exc)

    # Save model
    model_name = f"fraud_model_{best_name}_{version}.joblib"
    model_path = os.path.join(MODEL_DIR, model_name)
    joblib.dump(best_model, model_path)
    logger.info("Model saved -> %s", model_path)

    return best_model, thr, best_name, model_path, metrics_dict