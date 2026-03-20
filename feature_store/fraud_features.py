import pandas as pd


def build_fraud_features(df):

    df = df.sort_values("day")

    df["txn_count_day"] = df.groupby("day")["Amount_original"].transform("count")

    df["avg_amount_day"] = df.groupby("day")["Amount_original"].transform("mean")

    df["amount_deviation"] = df["Amount_original"] - df["avg_amount_day"]

    return df


def save_feature_store(df):

    features = build_fraud_features(df)

    features.to_parquet("feature_store/fraud_feature_table.parquet")

    print("Fraud feature store saved")