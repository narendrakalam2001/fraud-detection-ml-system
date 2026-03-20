import pandas as pd

from src.config import RANDOM_STATE


# ============================================================
# Train on Balanced Sample Instead of Full Dataset
# ============================================================

def fast_training_sample(X, y, majority_ratio=20):

    df = X.copy()
    df["target"] = y

    fraud = df[df["target"] == 1]
    normal = df[df["target"] == 0]

    normal_sample = normal.sample(
        n=min(len(normal), majority_ratio * len(fraud)),
        random_state=RANDOM_STATE
    )

    balanced = pd.concat([fraud, normal_sample])

    X_new = balanced.drop(columns=["target"])
    y_new = balanced["target"]

    return X_new, y_new