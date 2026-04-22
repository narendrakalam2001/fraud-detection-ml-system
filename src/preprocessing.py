# ============================================================
# PREPROCESSING — Credit Card Fraud Detection ML System
# ============================================================

import numpy as np
import pandas as pd
import logging

from sklearn.base          import BaseEstimator, TransformerMixin
from sklearn.compose       import ColumnTransformer
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer

from src.config import CLIP_FOLD

logger = logging.getLogger(__name__)


# ============================================================
# CLIPPER — IQR-based outlier clipping transformer
# ============================================================

class Clipper(BaseEstimator, TransformerMixin):
    """
    Clips values to [Q1 - fold*IQR, Q3 + fold*IQR].
    Fitted on train set only — applied to train + test using
    training statistics to prevent leakage.

    Why needed for fraud:
      Fraud transactions often contain extreme PCA values and
      extreme amounts. Raw outliers distort StandardScaler's
      mean/std estimates, shrinking all normal values toward zero.
      Clipping preserves the fraud signal while bounding the range.

    get_feature_names_out() implemented so ColumnTransformer
    can propagate clean feature names (prevents f0, f1... warnings).
    """

    def __init__(self, fold: float = CLIP_FOLD):
        self.fold = fold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        q1  = np.quantile(X, 0.25, axis=0)
        q3  = np.quantile(X, 0.75, axis=0)
        iqr = q3 - q1

        self.lower_ = q1 - self.fold * iqr
        self.upper_ = q3 + self.fold * iqr

        # Avoid zero-width clip range
        eps          = 1e-9
        self.upper_  = np.where(
            self.upper_ == self.lower_,
            self.upper_ + eps,
            self.upper_
        )

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features, dtype=object)
        n = getattr(self, "n_features_in_", 1)
        return np.array([f"x{i}" for i in range(n)], dtype=object)


# ============================================================
# PREPROCESSOR BUILDER
# ============================================================

def build_preprocessors(X_train):
    """
    Builds two ColumnTransformers:
      pre_scaled   — Clipper → PowerTransformer → StandardScaler
                     for distance / linear models (LR, SGD, KNN, NB)
      pre_unscaled — Clipper only (tree models are scale-invariant
                     but still benefit from bounded outlier range)

    Returns:
        pre_scaled, pre_unscaled, cont_cols
    """
    cont_cols = X_train.columns.tolist()

    skewed = [c for c in cont_cols if abs(X_train[c].skew()) > 1.0]
    normal = [c for c in cont_cols if c not in skewed]

    logger.info("Skewed cols (%d): %s", len(skewed), skewed)
    logger.info("Normal cols (%d): %s", len(normal),  normal)

    # ── Scaled preprocessor ───────────────────────────────────
    scaled_transformers = []

    if skewed:
        scaled_transformers.append((
            "skewed",
            Pipeline([
                ("clip",  Clipper(fold=CLIP_FOLD)),
                ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
                ("scale", StandardScaler()),
            ]),
            skewed
        ))

    if normal:
        scaled_transformers.append((
            "normal",
            Pipeline([
                ("clip",  Clipper(fold=CLIP_FOLD)),
                ("scale", StandardScaler()),
            ]),
            normal
        ))

    pre_scaled = ColumnTransformer(
        transformers=scaled_transformers,
        remainder="drop"
    )

    # ── Unscaled preprocessor (tree models) ───────────────────
    unscaled_transformers = []

    if skewed:
        unscaled_transformers.append((
            "skewed",
            Pipeline([("clip", Clipper(fold=CLIP_FOLD))]),
            skewed
        ))

    if normal:
        unscaled_transformers.append((
            "normal",
            Pipeline([("clip", Clipper(fold=CLIP_FOLD))]),
            normal
        ))

    pre_unscaled = ColumnTransformer(
        transformers=unscaled_transformers,
        remainder="drop"
    )

    logger.info(
        "Preprocessors built | skewed=%d  normal=%d  total=%d",
        len(skewed), len(normal), len(cont_cols)
    )

    return pre_scaled, pre_unscaled, cont_cols