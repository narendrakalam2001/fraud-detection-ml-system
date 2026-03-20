import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve
)

# ============================================================
# METRICS
# ============================================================

def tune_threshold(y_true, y_prob):

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

    best_idx = np.argmax(f1_scores)

    best_threshold = thresholds[best_idx]

    print("Best threshold:", round(best_threshold,4))
    print("Precision:", round(precision[best_idx],4))
    print("Recall:", round(recall[best_idx],4))
    print("F1:", round(f1_scores[best_idx],4))

    return best_threshold

def recall_at_k(y_true, y_prob, k=0.05):
    df = pd.DataFrame({'y': y_true, 'p': y_prob}).sort_values('p', ascending=False)
    top_k = int(len(df)*k)
    return df.iloc[:top_k]['y'].sum() / df['y'].sum()

def lift_at_k(y_true, y_prob, k=0.05):
    base = y_true.mean()
    return recall_at_k(y_true,y_prob,k)/base

def ks_statistic(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return max(tpr-fpr)

def psi(expected, actual, buckets=10):
    expected, actual = pd.Series(expected), pd.Series(actual)
    qs = np.linspace(0,1,buckets+1)
    e = np.percentile(expected,qs*100)
    a = np.percentile(actual,qs*100)
    return np.sum((e-a)*np.log((e+1e-6)/(a+1e-6)))

def cost_sensitive_evaluation(X_test, y_true, y_pred, y_prob, review_cost_per_txn=5):

    df_eval = X_test.copy()
    df_eval["y_true"] = y_true.values
    df_eval["y_pred"] = y_pred

    # Missed fraud (False Negative)
    fn = df_eval[(df_eval["y_true"]==1) & (df_eval["y_pred"]==0)]
    fraud_loss = fn["Amount_original"].sum()

    # False Positive review cost
    fp = df_eval[(df_eval["y_true"]==0) & (df_eval["y_pred"]==1)]
    review_cost = len(fp) * review_cost_per_txn

    total_loss = fraud_loss + review_cost

    return fraud_loss, review_cost, total_loss
