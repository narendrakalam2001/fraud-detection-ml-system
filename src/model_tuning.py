import numpy as np
import logging

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.pipeline import Pipeline as ImbPipeline

# Models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
except:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except:
    CatBoostClassifier = None

from src.config import (
    RANDOM_STATE,
    CV_FOLDS,
    RANDOM_SEARCH_ITERS,
    SELECT_K_MAX,
    N_JOBS
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# MODEL GRIDS
# ============================================================

scaled_models = {
    'LogisticRegression': (
        LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        { 'classifier__penalty':['l1','l2','elasticnet'],
          'classifier__C':np.logspace(-4,2,12),
          'classifier__solver':['saga'],
          'classifier__l1_ratio':[0,0.5,1],
          'classifier__class_weight':[None,'balanced']}
    
    ),
    'SGD': (
        SGDClassifier(loss='log_loss', random_state=RANDOM_STATE),
        {'classifier__alpha':[1e-4,1e-3,1e-2],
         'classifier__class_weight':[None,'balanced']}
    ),
    'GaussianNB': (GaussianNB(), {})
}

unscaled_models = {
    'DecisionTree': (DecisionTreeClassifier(random_state=RANDOM_STATE), {'classifier__max_depth':[5,10,None], 
                                                                         'classifier__class_weight':['balanced']}),
    'RandomForest': (RandomForestClassifier(n_jobs=N_JOBS, random_state=RANDOM_STATE),
                     {'classifier__n_estimators':[200,300],
                      'classifier__max_depth':[None,20],
                      'classifier__min_samples_leaf':[1,2,5],
                      'classifier__class_weight':['balanced']}),
    'ExtraTrees': (ExtraTreesClassifier(n_jobs=N_JOBS, random_state=RANDOM_STATE),
                   {'classifier__n_estimators':[200,300], 'classifier__class_weight':['balanced']}),
    'GradientBoosting': (GradientBoostingClassifier(random_state=RANDOM_STATE),
                         {'classifier__n_estimators':[200]}),
    'AdaBoost': (AdaBoostClassifier(random_state=RANDOM_STATE),
                 {'classifier__n_estimators':[100,200]}),
    'XGBoost': (XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE),
                {'classifier__n_estimators':[200,300],
                 'classifier__learning_rate':[0.05,0.1],
                 'classifier__max_depth':[3,5],
                 'classifier__scale_pos_weight':[100,300,500],
                 'classifier__subsample':[0.8,1.0],
                 'classifier__colsample_bytree':[0.8,1.0]}),
    'BernoulliNB': (BernoulliNB(), {})
}

if LGBMClassifier:
    unscaled_models['LightGBM'] = (
        LGBMClassifier(random_state=RANDOM_STATE, force_col_wise=True, verbose=-1),
        {
            'classifier__n_estimators':[200,300],
            'classifier__learning_rate':[0.05,0.1],
            'classifier__num_leaves':[31,63],
            'classifier__scale_pos_weight':[100,300,500]
        }
    )

if CatBoostClassifier:
    unscaled_models['CatBoost'] = (
        CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
        {
            'classifier__iterations':[200],
            'classifier__depth':[4,6],
            'classifier__auto_class_weights':['Balanced']
        }
    )

# ============================================================
# TUNING
# ============================================================

def tune_models(models, preprocessor, X, y):

    pipelines = {}
    feature_records = {}

    for name,(clf,params) in models.items():

        logger.info(f"Tuning {name}")

        pipe = ImbPipeline([
            ('pre',preprocessor),
            ('skb',SelectKBest(mutual_info_classif,k=min(SELECT_K_MAX,X.shape[1]))),
            ('classifier',clf)
        ])

        search = RandomizedSearchCV(
            pipe,
            params,
            n_iter=RANDOM_SEARCH_ITERS,
            scoring='average_precision',
            cv=StratifiedKFold(CV_FOLDS,shuffle=True,random_state=RANDOM_STATE),
            n_jobs=N_JOBS,
            verbose=0
        )

        search.fit(X,y)
        best = search.best_estimator_
        pipelines[name] = best

        try:
            selected = best.named_steps['skb'].get_support()
            feature_records[name] = selected
        except:
            feature_records[name] = None

    return pipelines, feature_records

