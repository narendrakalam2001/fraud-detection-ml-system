# 💳 Credit Card Fraud Detection ML System

⭐ **If you find this project useful, please give it a star!**

![Python](https://img.shields.io/badge/Python-3.9-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Production-orange)
![Tests](https://img.shields.io/badge/Tests-35%20passing-brightgreen)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-blue)
![CI](https://github.com/narendrakalam2001/fraud-detection-ml-system/actions/workflows/ci.yml/badge.svg)

Production-grade end-to-end machine learning system for **real-time credit card fraud detection** with a
**3-tier decision engine (APPROVE / REVIEW / BLOCK)**, IQR-based outlier clipping, correct PSI drift monitoring,
Champion vs Challenger promotion system, model card, and a 5-section live monitoring dashboard.

---

## 🚀 Project Overview

This project builds a complete **Fintech-grade Fraud Detection System** with:

* Automated ML training pipeline — 11 models, hyperparameter tuning
* Rich feature engineering — temporal (sin/cos), feature store, graph-based, anomaly score
* IQR-based `Clipper` transformer — prevents outlier distortion of StandardScaler
* SMOTE + `fast_training_sample` — handles 578:1 class imbalance
* Dual ColumnTransformer — scaled preprocessor for linear models, unscaled for tree models
* Leakage detection — exact match + near-perfect correlation check before training
* 3-tier rule engine — BLOCK_RULE / BLOCK_MODEL / REVIEW / APPROVE
* Champion vs Challenger model promotion system (3 promotion gates)
* Correct edge-based PSI drift monitoring (fixed from rank-based approach)
* Model card — structured JSON with metrics, cost eval, feature importances
* MLflow experiment tracking
* 35 pytest unit tests with coverage report
* Real-time FastAPI + Transaction Simulator
* Streamlit monitoring dashboard — 5 sections with real-time alerts

---

## 💡 Why This Project Matters

Real fraud detection requires more than a trained classifier.
This system combines ML + business rules + monitoring + governance to simulate
real-world payment system decision pipelines.

---

## 🏗 System Architecture

![Architecture](docs/architecture/system_architecture.png)

---
## 🌐 Live Demo

📊 Real-time fraud monitoring with interactive visualizations

🚀 **Fraud Monitoring Dashboard (Live)**
👉 [fraud-detection-ml-system.streamlit.app](https://fraud-detection-ml-system.streamlit.app)

⚡ **Fraud Detection API**
👉 [fraud-detection-ml-system.onrender.com](https://fraud-detection-ml-system.onrender.com)

📄 **API Docs:**
👉 [fraud-detection-ml-system.onrender.com/docs](https://fraud-detection-ml-system.onrender.com/docs)

---

## 📊 Monitoring Dashboard

The system includes a real-time monitoring dashboard built using **Streamlit** to track fraud predictions and system behavior.

---

### 🎬System Demo (End-to-End Flow)

![Dashboard Demo](docs/gifs/system_demo.gif)

---

### 🖥️ Full Dashboard UI

Real-time applicant fraud scoring + Champion vs Challenger history.

![Dashboard](docs/screenshots/dashboard_full_ui.png)

---

### 📈 Fraud Score and Decision Distribution

score plot shows how predicted fraud probabilities are distributed across transactions.
and decision shows how many transactions are approved, blocked, or sent for review.

![Fraud Score](docs/screenshots/fraud_score_decision_distribution.png)

---

### 📊 Score Statistics

Statistical summary of fraud scores and labels to monitor distribution shifts.

![Score Stats](docs/screenshots/score_statistics.png)

---

### 📉 Feature Drift Report (PSI) and score

PSI drift monitoring with 🔴🟡🟢 status flags

![Drift Report](docs/screenshots/drift_report.png)

![Drift Score](docs/screenshots/drift_score.png)

---

### 📋 Recent Predictions

Displays recent predictions with amount, fraud probability, and decision.

![Transactions](docs/screenshots/recent_prediction.png)

---

### 🔍 What This Dashboard Helps With

* Detect abnormal fraud score patterns
* Monitor model prediction behavior in real-time
* Track approval vs block vs review rates
* Identify potential model drift via PSI
* Compare champion vs challenger model versions
* Debug real-time predictions with rule trigger details

---

## 📈 Model Results

![Model Results](docs/reports/model_results.png)

**Best Model:** Extra Trees Classifier

| Metric | Value |
|--------|-------|
| F1 Score | 0.8962 |
| ROC-AUC | 0.9669 |
| PR-AUC | 0.8817 |
| KS Statistic | 0.9090 |
| Precision | 0.9318 |
| Recall | 0.8632 |
| Brier Score | 0.0003 |
| Train-Test Gap | 0.0004 |
| Estimated Fraud Loss | $834.69 |

---

## 🆚 Champion vs Challenger System

Every new training run is compared against the production champion using 3 promotion gates:

| Gate | Condition |
|------|-----------|
| F1 Improvement | Challenger must beat champion by ≥ 0.5% |
| ROC-AUC | Challenger must have ROC-AUC ≥ 0.95 |
| Generalization Gap | Train-test gap must be ≤ 10% |

Results logged to `fraud_models/challenger_log.json` and visible in dashboard.

---

## 🎯 Risk Decision Engine

Unlike simple classifiers, this system uses a **3-tier decision engine**:

| Decision | Trigger |
|----------|---------|
| `APPROVE` | Low probability + no rule flags |
| `REVIEW` | Borderline score (prob ≥ threshold × 0.6) |
| `BLOCK_MODEL` | High probability (prob ≥ threshold) |
| `BLOCK_RULE` | Hard rule — Amount > $5,000 |

Rules are checked **before** ML — matching real payment system architecture.

---

## 🧪 Test Coverage

![Tests](docs/reports/test_coverage.png)

---

## ⚡ Real-Time Prediction API

![API Demo](docs/screenshots/api_demo.png)

---

### Run API

```bash
python scripts/run_api.py
```

### Endpoint

```
POST /predict
```

### Example Request

```json
{
  "Time": 50000,
  "Amount": 120.5
}
```

### Example Response

```json
{
  "fraud_probability": 0.0312,
  "decision": "APPROVE",
  "rule_triggered": null,
  "latency_seconds": 0.043
}
```

---

## 🔁 Transaction Simulator

```bash
python scripts/run_simulation.py
```

Supports 3 scenarios: `random`, `risky`, `safe`

---

## 📊 Data Visualization (EDA)

* `notebooks/fraud_detection_eda.ipynb` — 25-step professional EDA
* `notebooks/fraud_detection_eda.html` — rendered HTML version

Note: The sample dataset provided is a subset of the original raw dataset.
All preprocessing and feature engineering are handled in the pipeline.

---

## ⚙ Machine Learning Pipeline

* Data validation + leakage detection
* IQR-based outlier clipping (Clipper transformer)
* Dual preprocessor — scaled (linear models) + unscaled (tree models)
* Feature engineering + feature store + graph features + anomaly score
* SMOTE oversampling + fast screening sample
* 11 models × RandomizedSearchCV (PR-AUC scoring)
* Threshold tuning (max F1)
* Cost-sensitive evaluation (FN loss + FP review cost)
* Model card + MLflow tracking
* Champion vs Challenger comparison

---

## 🤖 Models Used

LR · SGD · GaussianNB · DecisionTree · RandomForest · ExtraTrees ·
GradientBoosting · AdaBoost · XGBoost · LightGBM · CatBoost · MLP (NeuralNet)

---

## 📈 Evaluation Metrics

* Precision / Recall / F1 Score
* ROC-AUC / PR-AUC
* KS Statistic
* Recall@K / Lift@K
* Brier Score
* Train-Test Gap
* Cost Evaluation (Expected Credit Loss)

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

Tests cover: `Clipper`, `build_preprocessors`, `detect_leakage`,
`tune_threshold`, `psi`, `recall_at_k`, `lift_at_k`, `ks_statistic`,
`rule_engine`, `config` thresholds

---

## 📂 Project Structure

```
fraud-detection-ml-system/
│
├── src/
│   ├── config.py              ← constants, thresholds, PSI limits, challenger gates
│   ├── data_loader.py         ← validation + feature engineering
│   ├── preprocessing.py       ← Clipper + dual ColumnTransformer
│   ├── model_tuning.py        ← 11 model grids + RandomizedSearchCV
│   ├── metrics.py             ← PSI (edge-based fix), KS, cost eval, threshold
│   ├── rule_engine.py         ← 3-tier decision engine
│   ├── evaluation.py          ← model card, feature importances, MLflow, PSI drift
│   ├── leakage_check.py       ← pre-training leakage detection
│   ├── model_loader.py        ← champion load + 3-gate challenger comparison
│   ├── anomaly_detection.py   ← IsolationForest anomaly score
│   ├── neural_net.py          ← MLP pipeline
│   ├── sampling.py            ← fast_training_sample
│   └── training_pipeline.py  ← full orchestration
│
├── serving/
│   └── fraud_api.py          ← FastAPI endpoints
│
├── services/
│   └── prediction_service.py
│
├── monitoring/
│   └── monitoring_dashboard.py  ← 5-section Streamlit dashboard
│
├── simulation/
│   └── transaction_simulator.py
│
├── feature_store/
│   └── fraud_features.py
│
├── graph_detection/
│   └── fraud_graph_detection.py
│
├── scripts/
│   ├── train_model.py
│   ├── run_api.py
│   ├── run_dashboard.py
│   └── run_simulation.py
│
├── tests/
│   └── test_pipeline_core.py ← 35 pytest unit tests
│
├── notebooks/
│   └── fraud_detection_eda.ipynb  ← 25-step professional EDA
│
├── fraud_models/
│   ├── challenger_log.json
│   ├── feature_drift_report.csv
│   ├── fraud_model_ExtraTrees_v1.joblib
│   ├── latest_model.json
│   ├── model_card_ExtraTrees_v1.json
│   ├── fraud_model_v1_metadata.json
│   ├── monitor_scores.csv
│   └── model_experiment_results.csv
│
├── logs/
│   └── prediction_logs.csv
│
├── docs/
│   ├── architecture/
│   │   └── system_architecture.svg    
│   ├── gifs/
│   │   └── system_demo.gif                  
│   ├── reports/
│   │   ├── model_results.png               
│   │   └── test_coverage.png                        
│   └── screenshots/
│       ├── dashboard_full_ui.png                        
│       ├── fraud_score_decision_distribution.png 
│       ├── score_statistics.png ← revenue panel
│       ├── drift_report.png                              
│       ├── drift_score.png                             
│       ├── recent_prediction.png        
│       └── api_demo
│
├── .streamlit/
│   └── config.toml
│
├── Dockerfile                 ← API Docker image
├── Dockerfile.dashboard       ← Dashboard Docker image
├── docker-compose.yml         ← Local dev (API + Dashboard)
├── .dockerignore
├── .github/
│   └── workflows/
│       └── ci.yml             ← GitHub Actions CI (pytest on push)
├── requirements.txt
├── requirements_api.txt
├── requirements_dashboard.txt
├── runtime.txt
├── render.yaml
├── README.md
└── .gitignore
```

---

## 🐳 Docker

> Run the full system locally using Docker — no manual environment setup needed.

```bash
# API only
docker build -t fraud-detection-api .
docker run -p 8000:8000 -v ./fraud_models:/app/fraud_models fraud-detection-api
```

```bash
# API + Dashboard together
docker compose up --build
```

API will be available at `http://localhost:8000`
Dashboard will be available at `http://localhost:8501`

---

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Model

```bash
# Windows PowerShell
$env:FRAUD_DATA_PATH = "path\to\creditcard.csv"
python scripts/train_model.py

# Mac / Linux
FRAUD_DATA_PATH=path/to/creditcard.csv python scripts/train_model.py
```

### 3. Run Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 4. Start API

```bash
python scripts/run_api.py
```

### 5. Run Simulator

```bash
python scripts/run_simulation.py
```

### 6. Start Dashboard

```bash
streamlit run monitoring/monitoring_dashboard.py
```

---

## 🛠 Tech Stack

Python · Scikit-Learn · XGBoost · LightGBM · CatBoost · imbalanced-learn ·
FastAPI · Streamlit · NetworkX · MLflow · Pytest · Pandas · NumPy · Seaborn ·
Docker · GitHub Actions CI/CD · Render · Streamlit Cloud

---

## 📌 Future Improvements

* Kafka streaming for real-time transaction ingestion
* Online learning with concept drift adaptation
* SHAP explainability (version compatibility fix pending)
* A/B traffic splitting for live champion/challenger testing

---

## 👤 Author

**Narendra Kalam**
Machine Learning & Data Science

📧 kalamnarendra2001@gmail.com

🔗 https://www.linkedin.com/in/narendra-kalam

🌐 Portfolio: ...