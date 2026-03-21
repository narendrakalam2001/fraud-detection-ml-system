import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os

st.title("Fraud Detection Monitoring Dashboard")

# ==============================
# REAL-TIME FRAUD PREDICTION
# ==============================

st.subheader("🔮 Real-Time Fraud Prediction")

API_URL = "https://fraud-detection-ml-system.onrender.com/predict"

time_input = st.number_input("Transaction Time", value=50000.0)
amount_input = st.number_input("Transaction Amount", value=100.0)

if st.button("Predict Fraud"):

    payload = {
        "Time": time_input,
        "Amount": amount_input
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            st.success(f"Fraud Probability: {result['fraud_probability']:.4f}")
            st.warning(f"Decision: {result['decision']}")
        else:
            st.error("API Error")

    except Exception as e:
        st.error(f"Connection Error: {e}")

df = pd.read_csv("fraud_models/monitor_scores.csv")

# ==============================
# FRAUD RATE
# ==============================

if "label" in df.columns:

    fraud_rate = df["label"].mean()

    st.metric(
        "Fraud Rate (%)",
        round(fraud_rate*100,3),
        help="Percentage of transactions labeled as fraud"
    )

# ==============================
# SCORE DISTRIBUTION
# ==============================

score_col = None

for col in df.columns:
    if "score" in col.lower():
        score_col = col

if score_col:

    fig, ax = plt.subplots()

    ax.hist(df[score_col], bins=50, alpha=0.7)

    ax.set_title("Fraud Score Distribution")

    st.pyplot(fig)

else:

    st.error("No score column found in monitoring file")

# ==============================
# SCORE STATISTICS
# ==============================

st.subheader("Score Statistics")

st.write(df.describe())

# ==============================
# DECISION DISTRIBUTION
# ==============================

if "decision" in df.columns:

    st.subheader("Decision Distribution")

    decision_counts = df["decision"].value_counts()

    st.bar_chart(decision_counts)

# ==============================
# RECENT PREDICTIONS
# ==============================

log_path = "logs/prediction_logs.csv"

if os.path.exists(log_path):

    log_df = pd.read_csv(log_path)

    st.subheader("Recent Transactions")

    st.dataframe(log_df.tail(20))

else:

    st.warning("No prediction logs found yet.")