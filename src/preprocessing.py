from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer

# ============================================================
# PREPROCESSORS
# ============================================================

def build_preprocessors(X_train):

    cont_cols = X_train.columns.tolist()

    skewed = [c for c in cont_cols if abs(X_train[c].skew())>1]
    normal = [c for c in cont_cols if c not in skewed]

    pre_scaled = ColumnTransformer([
        ('skew', Pipeline([
            ('power', PowerTransformer()),
            ('scaler', StandardScaler())
        ]), skewed),
        ('norm', StandardScaler(), normal)
    ])

    pre_unscaled = ColumnTransformer([
        ('num','passthrough',cont_cols)
    ])

    return pre_scaled, pre_unscaled, cont_cols