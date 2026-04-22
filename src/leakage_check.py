# ============================================================
# LEAKAGE CHECK — Credit Card Fraud Detection ML System
# ============================================================
# Detects potential data leakage before model training.
# Two checks:
#   1. Exact match  — feature column identical to target
#   2. Near-perfect correlation — corr >= threshold with target
# ============================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def detect_leakage(
    X_train:        pd.DataFrame,
    y_train:        pd.Series,
    threshold_corr: float = 0.99
) -> list:
    """
    Runs leakage heuristics on the training set.

    Args:
        X_train        : feature DataFrame (after split, before preprocessing)
        y_train        : target Series (Class column)
        threshold_corr : correlation threshold above which a feature is flagged

    Returns:
        List of warning strings. Empty list = no leakage detected.

    Why this matters for fraud detection:
        The fraud dataset has PCA features that are already partially
        derived from the target signal. A near-perfect correlation (>0.99)
        would indicate a feature that almost perfectly predicts the label —
        meaning it likely encodes target information and should be removed.
    """
    warnings_list = []

    for col in X_train.columns:
        try:
            # ── Check 1: exact match with target ─────────────
            if X_train[col].equals(y_train.astype(X_train[col].dtype)):
                warnings_list.append(
                    f"[LEAKAGE] '{col}' is identical to target → remove"
                )
                continue

            # ── Check 2: near-perfect correlation ────────────
            if np.issubdtype(X_train[col].dtype, np.number):
                corr = abs(np.corrcoef(
                    X_train[col].fillna(0),
                    y_train.fillna(0)
                )[0, 1])

                if corr >= threshold_corr:
                    warnings_list.append(
                        f"[LEAKAGE] '{col}' corr={corr:.4f} with target "
                        f"(>= {threshold_corr}) → possible leakage"
                    )

        except Exception as e:
            logger.warning("Leakage check failed for '%s': %s", col, e)

    if warnings_list:
        for w in warnings_list:
            logger.warning(w)
    else:
        logger.info("Leakage check passed — no leakage detected")

    return warnings_list