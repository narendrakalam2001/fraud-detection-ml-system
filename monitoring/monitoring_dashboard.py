# ============================================================
# MONITORING DASHBOARD — Credit Card Fraud Detection ML System
# ============================================================

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Credit Card Fraud Detection — Monitoring Dashboard")

# ── API URL ───────────────────────────────────────────────────
API_URL = os.getenv(
    "FRAUD_API_URL",
    "https://fraud-detection-ml-system.onrender.com"
) + "/predict"

# ── PSI thresholds (from config) ──────────────────────────────
PSI_MODERATE     = 0.10
PSI_HIGH         = 0.20
SCORE_MEAN_ALERT = 0.10

# ── Path resolution ───────────────────────────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR    = os.path.dirname(_SCRIPT_DIR)
except Exception:
    BASE_DIR = os.getcwd()

MONITOR_PATH    = os.path.join(BASE_DIR, "fraud_models", "monitor_scores.csv")
LOG_PATH        = os.path.join(BASE_DIR, "logs",         "prediction_logs.csv")
PSI_PATH        = os.path.join(BASE_DIR, "fraud_models", "feature_drift_report.csv")
CHALLENGER_PATH = os.path.join(BASE_DIR, "fraud_models", "challenger_log.json")


# ============================================================
# SIDEBAR — LIVE PREDICTION
# ============================================================

st.sidebar.header("🔮 Predict Transaction Fraud")

time_input   = st.sidebar.number_input("Transaction Time (seconds)", value=50000.0, min_value=0.0)
amount_input = st.sidebar.number_input("Transaction Amount ($)",     value=100.0,   min_value=0.0)

if st.sidebar.button("Predict Fraud"):
    payload = {"Time": time_input, "Amount": amount_input}
    with st.sidebar:
        with st.spinner("Calling API... First request may take 30–60s (Render cold start)"):
            try:
                response = requests.post(API_URL, json=payload, timeout=90)
                if response.status_code == 200:
                    result   = response.json()
                    decision = result["decision"]
                    color    = {"APPROVE": "green", "REVIEW": "orange",
                                "BLOCK": "red", "BLOCK_RULE": "red",
                                "BLOCK_MODEL": "red"}.get(decision, "gray")
                    st.success("Prediction received!")
                    st.markdown(f"**Fraud Probability:** `{result['fraud_probability']}`")
                    st.markdown(
                        f"<h3 style='color:{color}'>Decision: {decision}</h3>",
                        unsafe_allow_html=True
                    )
                    if result.get("rule_triggered"):
                        st.warning(f"Rule: {result['rule_triggered']}")
                else:
                    st.error(f"API error: HTTP {response.status_code}")
                    st.code(response.text[:300])
            except requests.exceptions.Timeout:
                st.warning("Request timed out. Render is waking up — wait 30s and try again.")
            except Exception as e:
                st.error(f"Connection error: {e}")


# ============================================================
# SECTION 1 — REAL-TIME MONITORING ALERTS
# ============================================================

st.markdown("---")
st.subheader("🚨 Real-Time Monitoring Alerts")

alerts_found = False

if os.path.exists(MONITOR_PATH):
    df_monitor = pd.read_csv(MONITOR_PATH)

    if "score" in df_monitor.columns:
        avg_score = df_monitor["score"].mean()
        if avg_score > SCORE_MEAN_ALERT:
            st.error(
                f"🔴 HIGH FRAUD SCORE ALERT: avg={avg_score:.4f} "
                f"(threshold {SCORE_MEAN_ALERT})"
            )
            alerts_found = True

    if "decision" in df_monitor.columns:
        block_rate = df_monitor["decision"].str.contains("BLOCK").mean()
        if block_rate > 0.10:
            st.error(f"🔴 HIGH BLOCK RATE: {block_rate:.1%} (expected < 10%)")
            alerts_found = True

        review_rate = df_monitor["decision"].str.contains("REVIEW").mean()
        if review_rate > 0.15:
            st.warning(f"🟡 HIGH REVIEW QUEUE: {review_rate:.1%} (expected < 15%)")
            alerts_found = True

if os.path.exists(PSI_PATH):
    df_psi_alert = pd.read_csv(PSI_PATH)
    if "drift_score" in df_psi_alert.columns:
        max_psi     = df_psi_alert["drift_score"].max()
        top_feature = (
            df_psi_alert.iloc[0]["feature"]
            if "feature" in df_psi_alert.columns else "unknown"
        )
        if max_psi >= PSI_HIGH:
            st.error(
                f"🔴 CRITICAL DRIFT: PSI={max_psi:.4f} on '{top_feature}'. "
                "Retrain recommended."
            )
            alerts_found = True
        elif max_psi >= PSI_MODERATE:
            st.warning(
                f"🟡 MODERATE DRIFT: PSI={max_psi:.4f} on '{top_feature}'. "
                "Monitor closely."
            )
            alerts_found = True

if not alerts_found:
    st.success("✅ All systems normal — no alerts triggered")


# ============================================================
# SECTION 2 — CHAMPION vs CHALLENGER HISTORY
# ============================================================

st.markdown("---")
st.subheader("🏆 Champion vs Challenger History")

if os.path.exists(CHALLENGER_PATH):
    with open(CHALLENGER_PATH) as f:
        challenger_log = json.load(f)

    if challenger_log:
        latest         = challenger_log[-1]
        decision_color = "green" if latest["decision"] == "PROMOTED" else "red"
        icon           = "✅" if latest["decision"] == "PROMOTED" else "❌"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Latest Challenger", latest.get("challenger_name", "—"))
        with col2:
            st.metric("Challenger F1",     latest.get("challenger_f1",  "—"))
        with col3:
            st.metric("Champion F1",       latest.get("champion_f1",    "—") or "First Run")

        st.markdown(
            f"<h4 style='color:{decision_color}'>"
            f"{icon} Decision: {latest['decision']} — {latest.get('reason','')}"
            f"</h4>",
            unsafe_allow_html=True
        )

        if latest.get("gates"):
            g = latest["gates"]
            gcol1, gcol2, gcol3 = st.columns(3)
            gcol1.metric("F1 Gate",    "✅ Pass" if g.get("f1_improvement_passed") else "❌ Fail")
            gcol2.metric("ROC-AUC Gate", "✅ Pass" if g.get("roc_auc_passed") else "❌ Fail")
            gcol3.metric("Gap Gate",   "✅ Pass" if g.get("gap_passed") else "❌ Fail")

        if len(challenger_log) > 1:
            with st.expander("View full challenger history"):
                hist_df      = pd.DataFrame(challenger_log)
                display_cols = [c for c in [
                    "evaluated_at", "challenger_name", "challenger_f1",
                    "champion_name", "champion_f1", "decision", "reason"
                ] if c in hist_df.columns]
                st.dataframe(hist_df[display_cols])
else:
    st.info("No challenger log found. Run train_model.py to populate.")


# ============================================================
# SECTION 3 — KPI METRICS + CHARTS
# ============================================================

st.markdown("---")
st.subheader("📊 Model Performance KPIs")

if os.path.exists(MONITOR_PATH):
    df = pd.read_csv(MONITOR_PATH)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if "label" in df.columns:
            st.metric("Fraud Rate",     f"{df['label'].mean():.3%}")
    with col2:
        if "decision" in df.columns:
            st.metric("Auto-Approve Rate", f"{(df['decision']=='APPROVE').mean():.1%}")
    with col3:
        if "decision" in df.columns:
            st.metric("Review Rate",    f"{df['decision'].str.contains('REVIEW').mean():.1%}")
    with col4:
        if "decision" in df.columns:
            st.metric("Block Rate",     f"{df['decision'].str.contains('BLOCK').mean():.1%}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Fraud Score Distribution")
        if "score" in df.columns:
            fig, ax = plt.subplots()
            ax.hist(df["score"], bins=60, alpha=0.7, color="steelblue", edgecolor="white")
            ax.axvline(0.30, color="orange", linestyle="--", label="LOW/MED (0.30)")
            ax.axvline(0.60, color="red",    linestyle="--", label="MED/HIGH (0.60)")
            ax.set_xlabel("Fraud Probability")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close(fig)

    with col_b:
        st.subheader("Decision Distribution")
        if "decision" in df.columns:
            st.bar_chart(df["decision"].value_counts())

    st.subheader("Score Statistics")
    if "score" in df.columns:
        st.write(df["score"].describe().to_frame().T.round(4))
else:
    st.warning("No monitor scores found. Run train_model.py first.")


# ============================================================
# SECTION 4 — PSI DRIFT REPORT
# ============================================================

st.markdown("---")
st.subheader("📉 Feature Drift Report (PSI)")

if os.path.exists(PSI_PATH):
    df_psi = pd.read_csv(PSI_PATH)
    if "drift_score" in df_psi.columns:
        def _psi_flag(val):
            if val >= PSI_HIGH:       return "🔴 CRITICAL"
            elif val >= PSI_MODERATE: return "🟡 MODERATE"
            return "🟢 OK"
        df_psi["status_flag"] = df_psi["drift_score"].apply(_psi_flag)
        st.dataframe(df_psi.head(15), use_container_width=True)

        fig, ax = plt.subplots(figsize=(12, 4))
        colors  = [
            "#E74C3C" if v >= PSI_HIGH else "#F39C12" if v >= PSI_MODERATE else "#2ECC71"
            for v in df_psi["drift_score"].head(15)
        ]
        ax.barh(df_psi["feature"].head(15)[::-1],
                df_psi["drift_score"].head(15)[::-1],
                color=colors[::-1])
        ax.axvline(PSI_MODERATE, color="orange", linestyle="--", label=f"Moderate ({PSI_MODERATE})")
        ax.axvline(PSI_HIGH,     color="red",    linestyle="--", label=f"Critical ({PSI_HIGH})")
        ax.set_title("Feature PSI Drift Scores (Top 15)")
        ax.set_xlabel("PSI Score")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("PSI file found but 'drift_score' column missing.")
else:
    st.warning("Feature drift report not found. Run train_model.py first.")


# ============================================================
# SECTION 5 — RECENT PREDICTIONS
# ============================================================

st.markdown("---")
st.subheader("📋 Recent Predictions")

if os.path.exists(LOG_PATH):
    log_df = pd.read_csv(LOG_PATH)
    st.dataframe(log_df.tail(20), use_container_width=True)
else:
    st.info(
        "Prediction logs are written by the Render API — not available on Streamlit Cloud. "
        "Run locally to see live logs."
    )