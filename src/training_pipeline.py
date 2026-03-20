import pandas as pd

from src.data_loader import validate_input_data,load_and_engineer
from src.anomaly_detection import anomaly_filter
from src.sampling import fast_training_sample
from src.preprocessing import build_preprocessors
from src.model_tuning import scaled_models,unscaled_models,tune_models
from src.neural_net import train_mlp_pipeline
from src.evaluation import evaluate_models

from feature_store.fraud_features import build_fraud_features
from graph_detection.fraud_graph_detection import compute_graph_risk
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import logging

import os
import json
from src.config import RANDOM_STATE, MODEL_DIR
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training():

    DATA_PATH=r"D:\\Data Science Datasets\\Credit Card Fraud Detection.csv"

    df=pd.read_csv(DATA_PATH)

    df=validate_input_data(df)

    df=load_and_engineer(df)

    df=build_fraud_features(df)
    df=compute_graph_risk(df)

    X=df.drop(columns=["Class"])
    y=df["Class"]

    X_tr,X_te,y_tr,y_te=train_test_split(
        X,y,test_size=0.2,stratify=y,random_state=RANDOM_STATE
    )

    # =====================================
    # ANOMALY FILTER
    # =====================================

    X_tr, X_te = anomaly_filter(X_tr, X_te)

    # =====================================
    # STAGE 1: FAST MODEL SCREENING
    # =====================================

    X_fast, y_fast = fast_training_sample(X_tr, y_tr)

    pre_scaled_fast, pre_unscaled_fast, _ = build_preprocessors(X_fast)

    scaled_fast, _ = tune_models(
        scaled_models,
        pre_scaled_fast,
        X_fast,
        y_fast
    )

    unscaled_fast, _ = tune_models(
        unscaled_models,
        pre_unscaled_fast,
        X_fast,
        y_fast
    )

    # =====================================
    # SELECT TOP MODELS
    # =====================================

    top_scaled = list(scaled_fast.keys())[:3]
    top_unscaled = list(unscaled_fast.keys())[:3]

    # =====================================
    # BUILD PREPROCESSORS FOR FULL DATA
    # =====================================

    pre_scaled_full, pre_unscaled_full, _ = build_preprocessors(X_tr)
    
    # =====================================
    # STAGE 2: FULL DATA TRAINING
    # =====================================

    scaled_final = {}
    unscaled_final = {}

    for name in top_scaled:

        model = scaled_models[name][0]

        pipe = Pipeline([
            ("pre", pre_scaled_full),
            ("classifier", model)
        ])

        pipe.fit(X_tr, y_tr)

        scaled_final[name] = pipe


    for name in top_unscaled:

        model = unscaled_models[name][0]

        pipe = Pipeline([
            ("pre", pre_unscaled_full),
            ("classifier", model)
        ])

        pipe.fit(X_tr, y_tr)

        unscaled_final[name] = pipe
    
    mlp_model = train_mlp_pipeline(X_tr, y_tr, pre_scaled_full)

    all_models = {
        **scaled_final,
        **unscaled_final,
        "NeuralNet": mlp_model
    }

    # ===================================
    # MODEL EVALUATION
    # ===================================

    best_model, thr = evaluate_models(all_models, X_tr, X_te, y_tr, y_te)

    # ===================================
    # SAVE BEST MODEL
    # ===================================

    model_name = "fraud_model_v1.joblib"

    model_path = os.path.join(MODEL_DIR, model_name)

    # ===================================
    # UPDATE MODEL REGISTRY
    # ===================================

    registry = {
        "model_name": model_name,
        "threshold": thr
    }

    with open(os.path.join(MODEL_DIR,"latest_model.json"),"w") as f:
        json.dump(registry,f)