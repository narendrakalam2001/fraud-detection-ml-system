# ============================================================
# CONFIGURATION — Credit Card Fraud Detection ML System
# ============================================================

import os

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE         = 42
N_JOBS               = -1

# ── Cross-validation ─────────────────────────────────────────
CV_FOLDS             = 5
RANDOM_SEARCH_ITERS  = 5
SELECT_K_MAX         = 20

# ── Outlier clipping ─────────────────────────────────────────
CLIP_FOLD            = 1.5   # IQR multiplier for Clipper transformer

# ── Risk / Fraud bands (probability → tier) ──────────────────
FRAUD_BANDS = {
    "LOW":    (0.00, 0.30),
    "MEDIUM": (0.30, 0.60),
    "HIGH":   (0.60, 1.01),
}

# ── Business rule thresholds ─────────────────────────────────
HIGH_AMOUNT_RULE     = 5000    # Amount > this → BLOCK_RULE (hard rule)
LOW_AMOUNT_RULE      = 10      # Amount < this → potential card-testing flag
REVIEW_PROB_RATIO    = 0.6     # prob >= threshold * this ratio → REVIEW

# ── PSI drift thresholds ─────────────────────────────────────
PSI_MODERATE         = 0.10    # PSI >= 0.10 → moderate drift, monitor
PSI_HIGH             = 0.20    # PSI >= 0.20 → critical drift, retrain

# ── Score monitoring alert ───────────────────────────────────
SCORE_MEAN_ALERT     = 0.10    # avg fraud score > this → alert

# ── Challenger promotion gates ───────────────────────────────
MIN_F1_IMPROVEMENT      = 0.005   # challenger must beat champion by >= 0.5%
MIN_ROCAUC_THRESHOLD    = 0.95    # challenger must have ROC-AUC >= 0.95
MAX_GENERALIZATION_GAP  = 0.10    # train-test gap must be <= 10%

# ── Paths ─────────────────────────────────────────────────────
MODEL_DIR   = "fraud_models"
LOGS_DIR    = "logs"
DATA_DIR    = "data"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,  exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)