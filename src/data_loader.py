import numpy as np
import pandas as pd

# ============================================================
# DATA VALIDATION
# ============================================================

def validate_input_data(df):

    required_columns = ["Time", "Amount", "Class"]

    # Column existence
    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Data type checks
    if not np.issubdtype(df["Amount"].dtype, np.number):
        raise ValueError("Amount column must be numeric")

    if not np.issubdtype(df["Time"].dtype, np.number):
        raise ValueError("Time column must be numeric")

    # Target validation
    if not set(df["Class"].unique()).issubset({0,1}):
        raise ValueError("Class column must contain only 0 and 1")

    # Missing values warning
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        print(f"Warning: dataset contains {nulls} missing values")

    # Dataset size check
    if df.shape[0] < 100:
        raise ValueError("Dataset too small for training")

    print("Data validation passed")

    return df

# ============================================================
# DATA + FEATURE ENGINEERING
# ============================================================

def load_and_engineer(df):

    df = df.drop_duplicates()

    df["hour"] = (df["Time"] // 3600) % 24
    df["day"] = df["Time"] // (3600*24)

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    df.drop(columns=["Time","hour"], inplace=True)

    df["Amount_original"] = df["Amount"]

    df["Amount"] = np.log1p(df["Amount"])

    return df