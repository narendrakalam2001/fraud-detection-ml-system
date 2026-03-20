
# ============================================================
#  ANOMALY DETECTION
# ============================================================

from sklearn.ensemble import IsolationForest
from src.config import RANDOM_STATE

import logging
logger = logging.getLogger(__name__)

def anomaly_filter(X_train,X_test):

    logger.info("Anomaly Detection")

    iso = IsolationForest(
        n_estimators=100,
        contamination=0.002,
        random_state=RANDOM_STATE
    )

    iso.fit(X_train)

    train_scores = iso.decision_function(X_train)
    test_scores = iso.decision_function(X_test)

    X_train["anomaly_score"] = train_scores
    X_test["anomaly_score"] = test_scores

    return X_train, X_test
