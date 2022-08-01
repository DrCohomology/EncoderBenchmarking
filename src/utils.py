# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 10:29:35 2022

@author: federicom
"""

import glob
import numpy as np
import os
import pandas as pd
import time
import warnings

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets import fetch_openml
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    r2_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from tqdm import tqdm

# ----


DATASET_FOLDER = "C:/Data"
RESULT_FOLDER = "C:/Data/EncoderBenchmarking_results"


def cat2idx_dicts(domain) -> tuple:
    c2i, i2c = {}, {}
    idx = 0
    for cat in domain:
        c2i[cat] = idx
        i2c[idx] = cat
        idx += 1
    return c2i, i2c


def pre2process(df):

    df2 = df[~df.target.isna()].dropna(axis=1, how="any")

    if len(df2.columns) <= 1:
        df2 = df.dropna(axis=0, how="any").reset_index(drop=True)
    X = df2.drop(columns="target")
    y = df2.target.astype(int)

    return X, y


def mean_squared_score(x):
    return np.exp(-x)


def cohen_kappa(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return 2 * (tp * tn - fn * fp) / ((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn))


def get_models_metrics(dtype):
    dtype = str(dtype)
    if dtype == "bool":
        return (
            [RandomForestClassifier, DecisionTreeClassifier, ],
            [balanced_accuracy_score, r2_score],
        )
    elif dtype.startswith(("int", "float")):
        return (
            [DecisionTreeRegressor, RandomForestRegressor],
            [r2_score, mean_squared_score],
        )
    else:
        raise ValueError(f"{dtype} is not a valid target type")


def get_catnum(X):
    catcols = [col for col in X if "cat" in col]
    Xcat = X[catcols]
    Xnum = X[[col for col in X if col not in catcols and "target" not in col]]

    return Xcat, Xnum

def get_pipe_search_space_one_encoder(model, encoder):
    out = {}

    # https://arxiv.org/pdf/1802.09596.pdf
    # https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1301
    # bootstrap [sklearn] ~= sample.fraction [ranger]
    if "RandomForest" in repr(model):
        out = {
            # "model": Categorical([model()]),
            "model__max_features": Real(0.5, 1, prior="uniform"),
            "model__bootstrap": Categorical([True, False]),
            "model__max_depth": Categorical([2, 10, None]),
        }
    # https://arxiv.org/pdf/1802.09596.pdf
    # https://cran.r-project.org/web/packages/xgboost/xgboost.pdf
    # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
    elif "LGBM" in repr(model):
        out = {
            # "model": Categorical([model()]),
            "model__min_child_weight": Integer(3, 8),
            "model__colsample_bytree": Real(0.1, 1),
            "model__learning_rate": Real(0.01, 0.5),
            "model__max_depth": Integer(-1, 6)
        }
    elif "DecisionTree" in repr(model):
        out = {
            # "model": Categorical([model()]),
            "model__max_depth": Categorical(list(range(1, 5)) + [None]),
        }
    elif "Logistic" in repr(model):
        out = {
            "model__C": Real(0.2, 5)
        }
    # CatBoost paper: they tune just the number of trees
    elif "CatBoost" in repr(model):
        out = {
            "model__iterations": Integer(100, 1000)    
        }
    else:
        raise ValueError(
            f"Model with representation {repr(model)} is not valid")

    if "SmoothedTE" in repr(encoder):
        out.update({
            "preproc__encoder__w": Real(0, 20)
        })
    elif "CVTargetEncoder" in repr(encoder):
        out.update({
            "preproc__encoder__n_splits": Integer(2, 10)
        })
    return out
    
    


def get_pipe_search_space(model, encoders=[]):
    out = {}

    # https://arxiv.org/pdf/1802.09596.pdf
    # https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1301
    # bootstrap [sklearn] ~= sample.fraction [ranger]
    if "RandomForest" in repr(model):
        out = {
            # "model": Categorical([model()]),
            "model__max_features": Real(0.5, 1, prior="uniform"),
            "model__bootstrap": Categorical([True, False]),
            "model__max_depth": Categorical([2, 10, None]),
        }
    # https://arxiv.org/pdf/1802.09596.pdf
    # https://cran.r-project.org/web/packages/xgboost/xgboost.pdf
    # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
    elif "LGBM" in repr(model):
        out = {
            # "model": Categorical([model()]),
            "model__min_child_weight": Integer(3, 8),
            "model__colsample_bytree": Real(0.1, 1),
            "model__learning_rate": Real(0.01, 0.5),
            "model__max_depth": Integer(-1, 6)
        }
    elif "DecisionTree" in repr(model):
        out = {
            # "model": Categorical([model()]),
            "model__max_depth": Categorical(list(range(1, 5)) + [None]),
        }
    elif "Logistic" in repr(model):
        out = {
            "model__C": Real(0.2, 5)
        }
    # CatBoost paper: they tune just the number of trees
    elif "CatBoost" in repr(model):
        out = {
            "model__iterations": Integer(100, 1000)    
            
        }
        
    else:
        raise ValueError(
            f"Model with representation {repr(model)} is not valid")

    for encoder in encoders:
        if "SmoothedTE" in repr(encoder):
            out.update({
                "preproc__encoder__w": Real(0, 20)
            })
        elif "CVTargetEncoder" in repr(encoder):
            out.update({
                "preproc__encoder__n_splits": Integer(2, 10)
            })
    return out


def tune_pipe(pipe, X, y, search_space, score, random_state=1444, n_jobs=-1, n_iter=20, n_splits=5, verbose=0):
    start = time.time()
    cv = StratifiedKFold(
        n_splits=n_splits, random_state=random_state, shuffle=True
    )
    BS = BayesSearchCV(
        pipe,
        search_spaces=search_space,
        n_jobs=n_jobs,
        cv=cv,
        verbose=verbose,
        n_iter=n_iter,
        random_state=random_state+1,
        scoring=score,
        refit=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        BS.fit(X, y)
    return (
        {
            "best_score": BS.best_score_,
            "best_params": BS.best_params_,
            "time": time.time() - start,
        },
        BS,
    )


def get_acronym(string, underscore=True):
    out = ""
    for c in string.split("(")[0]:
        if c.isupper() or c.isdigit():
            out += c
    return out + "_" if underscore else out



