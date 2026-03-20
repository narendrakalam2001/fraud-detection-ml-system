# config

import os

RANDOM_STATE = 42
N_JOBS = -1
CV_FOLDS = 5
RANDOM_SEARCH_ITERS = 5
SELECT_K_MAX = 20
MODEL_DIR = "fraud_models"

os.makedirs(MODEL_DIR, exist_ok=True)