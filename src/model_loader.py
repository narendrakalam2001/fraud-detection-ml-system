# ============================================================
# MODEL LOADER + CHALLENGER SYSTEM
# Credit Card Fraud Detection ML System
# ============================================================

import os
import json
import joblib
import logging
import time

from src.config import (
    MODEL_DIR,
    MIN_F1_IMPROVEMENT,
    MIN_ROCAUC_THRESHOLD,
    MAX_GENERALIZATION_GAP,
)

logger = logging.getLogger(__name__)

CHALLENGER_LOG = os.path.join(MODEL_DIR, "challenger_log.json")


def load_latest_model():
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")
    if not os.path.exists(registry_path):
        raise FileNotFoundError(
            f"Model registry not found at {registry_path}. "
            "Run train_model.py first."
        )
    with open(registry_path) as f:
        registry = json.load(f)
    model_path = os.path.join(MODEL_DIR, registry["model_name"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model     = joblib.load(model_path)
    threshold = float(registry.get("threshold", 0.5))
    logger.info("Champion model loaded: %s  |  threshold=%.4f", model_path, threshold)
    return model, threshold


def _load_champion_metrics() -> dict:
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")
    if not os.path.exists(registry_path):
        return {}
    with open(registry_path) as f:
        registry = json.load(f)
    model_name = registry.get("model_name", "")
    parts      = model_name.replace("fraud_model_", "").replace(".joblib", "")
    card_path  = os.path.join(MODEL_DIR, f"model_card_{parts}.json")
    if not os.path.exists(card_path):
        logger.warning("Champion model card not found: %s", card_path)
        return {}
    with open(card_path) as f:
        card = json.load(f)
    metrics = card.get("metrics", card)
    return {
        "model_name": str(card.get("model_name", "unknown")),
        "f1":         float(metrics.get("test_f1",  0)),
        "roc_auc":    float(metrics.get("roc_auc",  0)),
        "gap":        float(metrics.get("train_test_gap", 0)),
    }


def run_challenger_comparison(
    challenger_name,
    challenger_f1,
    challenger_roc_auc,
    challenger_gap,
    challenger_model_path,
    challenger_threshold,
) -> dict:

    os.makedirs(MODEL_DIR, exist_ok=True)
    champion = _load_champion_metrics()

    if not champion:
        logger.info("No champion found — challenger auto-promoted (first run)")
        _update_registry(challenger_model_path, challenger_threshold)
        result = {
            "decision":           "PROMOTED",
            "reason":             "No existing champion — first model auto-promoted",
            "evaluated_at":       time.strftime("%Y-%m-%d %H:%M:%S"),
            "challenger_name":    str(challenger_name),
            "challenger_f1":      round(float(challenger_f1),      4),
            "challenger_roc_auc": round(float(challenger_roc_auc), 4),
            "champion_name":      None,
            "champion_f1":        None,
            "gates": {
                "f1_improvement_passed": True,
                "roc_auc_passed":        True,
                "gap_passed":            True,
            },
        }
        _save_challenger_log(result)
        return result

    champion_f1   = float(champion.get("f1",       0.0))
    champion_roc  = float(champion.get("roc_auc",  0.0))
    champion_name = str(champion.get("model_name", "unknown"))

    logger.info("=" * 55)
    logger.info("CHAMPION vs CHALLENGER")
    logger.info("  Champion  : %-20s  F1=%.4f  ROC=%.4f", champion_name, champion_f1, champion_roc)
    logger.info("  Challenger: %-20s  F1=%.4f  ROC=%.4f", challenger_name, challenger_f1, challenger_roc_auc)
    logger.info("=" * 55)

    # Explicitly cast to Python bool — numpy bool_ is NOT JSON serializable
    f1_diff = float(challenger_f1) - float(champion_f1)
    gate1   = True if f1_diff >= float(MIN_F1_IMPROVEMENT)       else False
    gate2   = True if float(challenger_roc_auc) >= float(MIN_ROCAUC_THRESHOLD)    else False
    gate3   = True if float(challenger_gap)      <= float(MAX_GENERALIZATION_GAP) else False

    if gate1 and gate2 and gate3:
        decision = "PROMOTED"
        reason   = (
            f"All gates passed — F1 {champion_f1:.4f} -> {float(challenger_f1):.4f} "
            f"(+{f1_diff:.4f})"
        )
        logger.info("CHALLENGER PROMOTED -> new champion: %s", challenger_name)
        _update_registry(challenger_model_path, challenger_threshold)
    else:
        decision = "REJECTED"
        failed   = []
        if not gate1:
            failed.append(f"F1 improvement {f1_diff:+.4f} < {MIN_F1_IMPROVEMENT}")
        if not gate2:
            failed.append(f"ROC-AUC {float(challenger_roc_auc):.4f} < {MIN_ROCAUC_THRESHOLD}")
        if not gate3:
            failed.append(f"train-test gap {float(challenger_gap):.4f} > {MAX_GENERALIZATION_GAP}")
        reason = "Gates failed: " + " | ".join(failed)
        logger.info("CHALLENGER REJECTED — champion '%s' retained", champion_name)
        logger.info("   Reason: %s", reason)

    result = {
        "decision":           str(decision),
        "reason":             str(reason),
        "evaluated_at":       time.strftime("%Y-%m-%d %H:%M:%S"),
        "challenger_name":    str(challenger_name),
        "challenger_f1":      round(float(challenger_f1),      4),
        "challenger_roc_auc": round(float(challenger_roc_auc), 4),
        "challenger_gap":     round(float(challenger_gap),     4),
        "champion_name":      str(champion_name),
        "champion_f1":        round(float(champion_f1),        4),
        "champion_roc_auc":   round(float(champion_roc),       4),
        "gates": {
            "f1_improvement_passed": gate1,
            "roc_auc_passed":        gate2,
            "gap_passed":            gate3,
        },
    }

    _save_challenger_log(result)
    return result


def _update_registry(model_path, threshold):
    registry = {
        "model_name": str(os.path.basename(model_path)),
        "threshold":  round(float(threshold), 4),
    }
    with open(os.path.join(MODEL_DIR, "latest_model.json"), "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registry updated -> %s", registry["model_name"])


def _save_challenger_log(result: dict):
    history = []
    if os.path.exists(CHALLENGER_LOG):
        try:
            with open(CHALLENGER_LOG) as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(result)
    # default=str is the FINAL safety net for any remaining non-serializable types
    with open(CHALLENGER_LOG, "w") as f:
        json.dump(history, f, indent=2, default=str)
    logger.info("Challenger log saved -> %s", CHALLENGER_LOG)