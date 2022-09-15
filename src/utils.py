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

from functools import reduce
from itertools import product
from scipy.stats import t
from sklearn.impute import SimpleImputer
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
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from tqdm import tqdm

# models
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ----


DATASET_FOLDER = "C:/Data"
RESULT_FOLDER = "C:/Data/EncoderBenchmarking_results"

DATASETS = {
    'kr-vs-kp': 3,
    'credit-approval': 29,
    'credit-g': 31,
    'sick': 38,
    'tictactoe': 50,
    'heart-h': 51,
    'vote': 56,
    'monks-problems-1': 333,
    'monks-problems-2': 334,
    'irish': 451,
    'profb': 470,
    'mv': 881,
    'molecular_biology_promoters': 956,
    'nursery': 959,
    'kdd_internet_usage': 981,
    'ada_prior': 1037,
    'KDDCup09_appetency': 1111,
    'KDDCup09_churn': 1112,
    'KDDCup09_upselling': 1114,
    'airlines': 1169,
    'Agrawal1': 1235,
    'bank_marketing': 1461,
    'blogger': 1463,
    'nomao': 1486,
    'thoracic-surgery': 1506,
    'wholesale-customers': 1511,
    'adult': 1590,
    'cylinder-bands': 6332,
    'dresses-sales': 23381,
    'SpeedDating': 40536,
    'titanic': 40945,
    'Australian': 40981,
    'jungle_chess_2pcs_endgame_elephant_elephant': 40999,
    'jungle_chess_2pcs_endgame_rat_rat': 41005,
    'jungle_chess_2pcs_endgame_lion_lion': 41007,
    'kick': 41162,
    'porto_seguro': 41224,
    'churn': 42178,
    'KDD98': 42343,
    'sf-police-incidents': 42344,
    'open_payments': 42738,
    'Census-Income-KDD': 42750,
    'students_scores': 43098,
    'WMO-Hurricane-Survival-Dataset': 43607,
    'law-school-admission-bianry': 43890,
    'national-longitudinal-survey-binary': 43892,
    'ibm-employee-attrition': 43896,
    'ibm-employee-performance': 43897,
    'amazon_employee_access': 43900,
    'mushroom': 43922
}


def t_test(v1, v2, alpha=0.05, corrected=True):
    """
    Test whether one of v1 or v2 is statistically greater than the other.
    Assume v1 and v2 are results from a cross validation
    Add Bengio correction term (2003)
    """
    if len(v1) != len(v2):
        raise ValueError("The inputs should have equal length.")

    n = len(v1)

    diff = v1 - v2
    avg = diff.mean()
    std = diff.std()
    if std == 0:
        return 0, 0, 0, 0

    # test training ratio
    ttr = 1 / (n - 1)

    adjstd = np.sqrt(1 / n + ttr) * std if corrected else np.sqrt(1 / n) * std
    tstat = avg / adjstd

    df = n - 1
    crit = t.ppf(1.0 - alpha, df)
    p = (1.0 - t.cdf(np.abs(tstat), df)) * 2.0
    return p, tstat, df, crit
def compare_with_ttest(v1, v2, alpha=0.05, corrected=True):
    """
    returns 0 if the two are indistinguishable
    returns 1 if v1 is "greater than" v2
    returns 2 if v2 is "greater than" v1

    """

    if (v1 == v2).all():
        return 0, 1

    p, s1, s2, s3 = t_test(v1, v2, alpha=alpha, corrected=corrected)

    if p >= alpha:
        return 0, p
    else:
        if (v1 - v2).mean() > 0:
            return 1, p
        else:
            return 2, p
def get_dominating(R):
    """
    Retrieves the indices of non-dominated elements and "removes" them from the matrix.
    At the end of the recursive process, the result is a list of nested sets.
    """
    if np.linalg.norm(R - np.ones_like(R)) == 0:
        return [set(range(len(R)))]

    non_dominated = []
    R = R.copy()
    for i in range(len(R)):
        if R[i, :].sum() == len(R):
            non_dominated.append(i)
    # having found the index of non-dominated rows, we can make the corresponding columns ininfluent
    # ie make the other indices dominate also the dominating indices
    for i in non_dominated:
        R[i, :] = np.ones_like(R[i, :])
        R[:, i] = np.ones_like(R[:, i])
    final_rank = [set(non_dominated)]
    final_rank.extend(get_dominating(R))
    return final_rank
def filter_rank(r):
    """
    Transforms a list of nested sets into a list of increments
    """
    return [r[0]] + [r[i]-r[i-1] for i in range(1, len(r))]
def test_totality(R):
    """
    Tests for totality of a relation. Bonus come riflexivity.
    """
    if len(R.shape) >= 3 or R.shape[0] != R.shape[1]:
        raise ValueError(f"The input should be a square matrix but has shape {R.shape}.")
    return reduce(
        lambda x, y: x and y,
        [
            R[i, j] + R[j, i] >= 1
            for i, j in product(range(len(R)), repeat=2)
        ]
    )
def test_transitivity(R):
    if len(R.shape) >= 3 or R.shape[0] != R.shape[1]:
        raise ValueError(f"The input should be a square matrix but has shape {R.shape}.")
    return reduce(
        lambda x, y: x and y,
        [
            R[i, j] + R[j, k] - R[i, k] <= 1
            for i, j, k in product(range(len(R)), repeat=3)
        ]
    )
def get_rank_from_matrix(R, e2i):
    """
    If the matrix satisfies totality, riflexivity, and transitivity properties (ie weak order), extract the rank
    of every element that is key in e2i.
    Two procedures are possible:
        best first: take rows with maximum sum (dominating everything)
        worst frst: take columns with maximum sum (dominated by everything)
    """

    # test for totality
    if not test_totality(R):
        raise ValueError("Input matrix is not a total relation, hence not a weak order.")
    elif not test_transitivity(R):
        raise ValueError("Input matrix is not a transitive relation, hence not a weak order.")

    rank = {}
    for E, i in e2i.items():
        for j, dom in enumerate(filter_rank(get_dominating(R))):
            if i in dom:
                rank[E] = j
                continue
    return rank
def get_lgbm_scoring(scoring):
    def lgbm_scoring(y_true, y_pred):
        return scoring.__name__, scoring(y_true, np.round(y_pred)), True

    return lgbm_scoring
def cat2idx_dicts(domain) -> tuple:
    c2i, i2c = {}, {}
    idx = 0
    for cat in domain:
        c2i[cat] = idx
        i2c[idx] = cat
        idx += 1
    return c2i, i2c
def mean_squared_score(x):
    return np.exp(-x)
def cohen_kappa(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return 2 * (tp * tn - fn * fp) / ((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn))
def get_pipe_search_space_one_encoder(model, encoder):
    out = {}

    # https://arxiv.org/pdf/1802.09596.pdf
    # https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1301
    # bootstrap [sklearn] ~= sample.fraction [ranger]
    if isinstance(model, RandomForestClassifier):
        out = {
            # "model": Categorical([model()]),
            "model__max_features": Real(0.5, 1, prior="uniform"),
            "model__bootstrap": Categorical([True, False]),
            "model__max_depth": Categorical([2, 10, None]),
        }
    # https://arxiv.org/pdf/1802.09596.pdf
    # https://cran.r-project.org/web/packages/xgboost/xgboost.pdf
    # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
    elif isinstance(model, LGBMClassifier):
        out = {
            # "model": Categorical([model()]),
            # "model__min_child_weight": Integer(3, 8),
            # "model__colsample_bytree": Real(0.1, 1),
            # "model__learning_rate": Real(0.01, 0.5),
            # "model__max_depth": Integer(-1, 6)
            "model__n_estimators": Integer(1, 1000, prior="log-uniform")
        }
    elif isinstance(model, DecisionTreeClassifier):
        out = {
            # "model": Categorical([model()]),
            "model__max_depth": Integer(2, 5),
        }
    elif isinstance(model, LogisticRegression):
        out = {
            "model__C": Real(0.2, 5)
        }
    # CatBoost paper: they tune just the number of trees
    elif isinstance(model, CatBoostClassifier):
        out = {
            "model__iterations": Integer(1, 1000, prior="log-uniform")
        }
    elif isinstance(model, SVC):
        out = {
            "model__C": Real(0.1, 2),
            "model__gamma": Real(0.1, 100, prior="log-uniform")
        }
    elif isinstance(model, KNeighborsClassifier):
        out = {
            "model__n_neighbors": Integer(2, 10)
        }

    else:
        raise ValueError(
            f"Model with representation {repr(model)} is not valid")

    # if "SmoothedTE" in repr(encoder):
    #     out.update({
    #         "preproc__encoder__w": Real(0, 20)
    #     })
    # elif "CVTargetEncoder" in repr(encoder):
    #     out.update({
    #         "preproc__encoder__n_splits": Integer(2, 10)
    #     })
    return out
def tune_pipe(pipe, X, y, search_space, score, random_state=1444, n_jobs=-1, max_iter=20, n_splits=5, verbose=0):
    start = time.time()
    cv = StratifiedKFold(
        n_splits=n_splits, random_state=random_state, shuffle=True
    )

    n_iter = 1
    for par_space in search_space.values():
        if isinstance(par_space, Integer):
            bounds = par_space.bounds
            par_iter = bounds[1] - bounds[0] + 1
            n_iter *= par_iter
        else:
            n_iter *= 3
    n_iter = min(n_iter, max_iter)

    BS = BayesSearchCV(
        pipe,
        search_spaces=search_space,
        n_jobs=n_jobs,
        cv=cv,
        verbose=verbose,
        n_iter=n_iter,
        random_state=random_state + 1,
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