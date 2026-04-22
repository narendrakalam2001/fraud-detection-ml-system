# ============================================================
# METRICS — Credit Card Fraud Detection ML System
# ============================================================

import numpy as np
import pandas as pd
import logging

from sklearn.metrics import precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)


# ============================================================
# THRESHOLD TUNING — maximise F1
# ============================================================

def tune_threshold(y_true, y_prob):
    """
    Finds the decision threshold that maximises F1-score
    on the precision-recall curve.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
    best_idx  = np.nanargmax(f1_scores)
    best_thr  = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    logger.info(
        "Best threshold: %.4f  |  precision=%.4f  recall=%.4f  f1=%.4f",
        best_thr, precision[best_idx], recall[best_idx], f1_scores[best_idx]
    )
    return best_thr


# ============================================================
# PSI — Population Stability Index  (edge-based, correct)
# ============================================================

def psi(expected, actual, buckets: int = 10) -> float:
    """
    Population Stability Index — measures distribution shift.

    PSI < 0.10  → no significant shift (stable)
    PSI 0.10–0.20 → moderate shift   (monitor closely)
    PSI > 0.20  → major shift         (retrain recommended)

    Correct implementation:
      1. Compute quantile bin EDGES from `expected` (reference/train)
      2. Bin BOTH distributions using those SAME edges
      3. Compare proportions via the PSI formula

    Common bug (present in old fraud metrics.py):
      Ranking both arrays independently → both become uniform
      → PSI is always ~0 regardless of actual shift. WRONG.
    """
    try:
        expected = np.asarray(expected, dtype=float)
        actual   = np.asarray(actual,   dtype=float)

        # Step 1: bin edges from reference (training) distribution only
        quantiles  = np.linspace(0, 100, buckets + 1)
        bin_edges  = np.percentile(expected, quantiles)
        bin_edges  = np.unique(bin_edges)

        if len(bin_edges) < 2:
            return 0.0

        # Extend edges to cover full range of actual distribution
        bin_edges[0]  = min(bin_edges[0],  actual.min()) - 1e-9
        bin_edges[-1] = max(bin_edges[-1], actual.max()) + 1e-9

        # Step 2: bin BOTH using the SAME edges
        exp_hist, _ = np.histogram(expected, bins=bin_edges)
        act_hist, _ = np.histogram(actual,   bins=bin_edges)

        # Step 3: proportions, avoid div-by-zero
        exp_pct = exp_hist / (exp_hist.sum() + 1e-9)
        act_pct = act_hist / (act_hist.sum() + 1e-9)
        exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
        act_pct = np.where(act_pct == 0, 1e-6, act_pct)

        # Step 4: PSI formula
        psi_val = float(np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct)))
        return psi_val

    except Exception as e:
        logger.warning("PSI computation failed: %s", e)
        return float("nan")


# ============================================================
# RECALL @ K
# ============================================================

def recall_at_k(y_true, y_prob, k: float = 0.05) -> float:
    df = (
        pd.DataFrame({"y": y_true, "p": y_prob})
        .sort_values("p", ascending=False)
    )
    top_n = int(len(df) * k)
    return float(df.iloc[:top_n]["y"].sum() / (df["y"].sum() + 1e-9))


# ============================================================
# LIFT @ K
# ============================================================

def lift_at_k(y_true, y_prob, k: float = 0.05) -> float:
    base = np.mean(y_true)
    return float(recall_at_k(y_true, y_prob, k) / (base + 1e-9))


# ============================================================
# KS STATISTIC
# ============================================================

def ks_statistic(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


# ============================================================
# COST-SENSITIVE EVALUATION
# ============================================================

def cost_sensitive_evaluation(
    X_test,
    y_true,
    y_pred,
    y_prob,
    review_cost_per_txn: float = 5.0
):
    """
    Fintech-grade cost evaluation:
      - False Negative loss  = actual dollar amount of missed fraud
      - False Positive cost  = analyst review cost per incorrectly flagged txn

    Returns: (fraud_loss, review_cost, total_loss)
    """
    df_eval = X_test.copy()
    df_eval["y_true"] = y_true.values if hasattr(y_true, "values") else y_true
    df_eval["y_pred"] = y_pred

    # Missed fraud (FN) → actual financial loss
    fn_mask   = (df_eval["y_true"] == 1) & (df_eval["y_pred"] == 0)
    amount_col = "Amount_original" if "Amount_original" in df_eval.columns else "Amount"
    fraud_loss = float(df_eval.loc[fn_mask, amount_col].sum()) if amount_col in df_eval.columns \
                 else float(fn_mask.sum())

    # False positives (FP) → analyst review cost
    fp_mask     = (df_eval["y_true"] == 0) & (df_eval["y_pred"] == 1)
    review_cost = float(fp_mask.sum() * review_cost_per_txn)

    total_loss = fraud_loss + review_cost

    logger.info(
        "Cost eval  |  FN=%d  FP=%d  fraud_loss=%.2f  review_cost=%.2f  total=%.2f",
        fn_mask.sum(), fp_mask.sum(), fraud_loss, review_cost, total_loss
    )

    return fraud_loss, review_cost, total_loss