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
from numpy.random import default_rng
from openml.datasets import get_dataset
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    make_scorer,
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
from sklearn.model_selection import StratifiedKFold, GridSearchCV
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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ----

RESULT_FOLDER = "C:/Data/EncoderBenchmarking_results/ExperimentalResults"

DATASETS = {
    'kr-vs-kp': 3,
    'credit-approval': 29,
    'credit-g': 31,
    'sick': 38,
    'tic-tac-toe': 50,
    'heart-h': 51,
    'vote': 56,
    'monks-problems-1': 333,
    'monks-problems-2': 334,
    'irish': 451,
    'profb': 470,
    'mv': 881,
    'molecular-biology_promoters': 956,
    'nursery': 959,
    'kdd_internet_usage': 981,
    'ada_prior': 1037,
    'KDDCup09_appetency': 1111,
    'KDDCup09_churn': 1112,
    'KDDCup09_upselling': 1114,
    'airlines': 1169,
    'Agrawal1': 1235,
    'bank-marketing': 1461,
    'blogger': 1463,
    'nomao': 1486,
    'thoracic-surgery': 1506,
    'wholesale-customers': 1511,
    'adult': 1590,
    'cylinder-bands': 6332,
    'dresses-sales': 23381,
    'SpeedDating': 40536,
    'Titanic': 40945,
    'Australian': 40981,
    'jungle_chess_2pcs_endgame_elephant_elephant': 40999,
    'jungle_chess_2pcs_endgame_rat_rat': 41005,
    'jungle_chess_2pcs_endgame_lion_lion': 41007,
    'kick': 41162,
    'porto-seguro': 41224,
    'telco-customer-churn': 42178,
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

DATASETS_SMALL = {
    'kr-vs-kp': 3,
    'credit-approval': 29,
    'credit-g': 31,
    'sick': 38,
    'tic-tac-toe': 50,
    'vote': 56,
    'monks-problems-1': 333,
    'monks-problems-2': 334,
    'irish': 451,
    'profb': 470,
    'mv': 881,
    'molecular-biology_promoters': 956,
    'kdd_internet_usage': 981,
    'ada_prior': 1037,
    'blogger': 1463,
    'thoracic-surgery': 1506,
    'wholesale-customers': 1511,
    'adult': 1590,
    'cylinder-bands': 6332,
    'dresses-sales': 23381,
    'SpeedDating': 40536,
    'Australian': 40981,
    'jungle_chess_2pcs_endgame_elephant_elephant': 40999,
    'jungle_chess_2pcs_endgame_rat_rat': 41005,
    'jungle_chess_2pcs_endgame_lion_lion': 41007,
    'students_scores': 43098,
    'national-longitudinal-survey-binary': 43892,
    'ibm-employee-attrition': 43896,
    'ibm-employee-performance': 43897,
    'mushroom': 43922
}


LEFT_DATASETS = {
    # 'kr-vs-kp': 3,
    # 'credit-approval': 29,
    # 'credit-g': 31,
    # 'sick': 38,
    # 'tictactoe': 50,
    # 'heart-h': 51, PROBLEMATIC
    # 'vote': 56,
    # 'monks-problems-1': 333,
    # 'monks-problems-2': 334,
    # 'irish': 451,
    # 'profb': 470,
    # 'mv': 881,
    # 'molecular_biology_promoters': 956,
    # 'nursery': 959, PROBLEMATIC
    # 'kdd_internet_usage': 981,
    # 'ada_prior': 1037,
    'KDDCup09_appetency': 1111,
    'KDDCup09_churn': 1112,
    'KDDCup09_upselling': 1114,
    'airlines': 1169,
    'Agrawal1': 1235,
    'bank_marketing': 1461,
    # 'blogger': 1463,
    'nomao': 1486,
    # 'thoracic-surgery': 1506,
    # 'wholesale-customers': 1511,
    'adult': 1590,
    # 'cylinder-bands': 6332,
    # 'dresses-sales': 23381,
    # 'SpeedDating': 40536,
    # 'titanic': 40945, PROBLEMATIC
    # 'Australian': 40981,
    # 'jungle_chess_2pcs_endgame_elephant_elephant': 40999,
    # 'jungle_chess_2pcs_endgame_rat_rat': 41005,
    # 'jungle_chess_2pcs_endgame_lion_lion': 41007,
    'kick': 41162,
    'porto_seguro': 41224,
    'churn': 42178,
    'KDD98': 42343,
    'sf-police-incidents': 42344,
    'open_payments': 42738,
    'Census-Income-KDD': 42750,
    # 'students_scores': 43098,
    'WMO-Hurricane-Survival-Dataset': 43607,
    # 'law-school-admission-bianry': 43890, PROBLEMATIC
    # 'national-longitudinal-survey-binary': 43892,
    # 'ibm-employee-attrition': 43896,
    # 'ibm-employee-performance': 43897,
    'amazon_employee_access': 43900,
    # 'mushroom': 43922
}


def smart_sort(experiments, num_small_per_large=100, random=False):
    """
    orders the experiments according to dataset size, so that big iterations are not condensed together
    less likely segmentation
    """

    if random:
        return list(default_rng().permutation(tuple(experiments)))

    lengths = dict(
        sorted({ie: len(ee[0].get_data()[0]) for ie, ee in enumerate(experiments)}.items(), key=lambda x: x[1])
    )
    presorted_indices = list(lengths.keys())
    # take 1 from last, kk from first positions
    num_small_per_large = 100
    indices = []
    i=0
    while len(presorted_indices) > 0:
        print(i)
        indices.append(presorted_indices.pop())
        indices.extend(presorted_indices[:num_small_per_large])
        presorted_indices = presorted_indices[num_small_per_large:]
        i+=1

    return [experiments[i] for i in indices]


def remove_failed_main8(experiments, result_folder):
    """
    logs are in the form "success_dataset_encoder"
    """

    print("Removing failed runs.")

    # --- Find the failed logs
    failed_runs = [os.path.split(x)[-1] for x in glob.glob(os.path.join(result_folder, "logs", "2*.json"))]

    return [x for x in experiments if f"2_{x[0]}_{get_acronym(x[1].__str__(), underscore=False)}.json" not in failed_runs]


def remove_failed_main9(experiments, result_folder):
    """
    logs are in the form "success_dataset_encoder"
    """

    print("Removing failed runs.")

    # --- Find the failed logs
    failed_runs = [os.path.split(x)[-1] for x in glob.glob(os.path.join(result_folder, "logs", "2*.json"))]

    return [x for x in experiments if f"2_{x[0]}_{get_acronym(x[1].__str__(), underscore=False)}.json" not in failed_runs]

def remove_concluded_main8(experiments, result_folder, model=None):
    print("Removing completed runs.")

    # --- Load df_concatenated
    try:
        df_concatenated = pd.read_csv(os.path.join(result_folder, "main8_final.csv"))
    except FileNotFoundError:
        df_concatenated = pd.read_csv(os.path.join(result_folder, "_concatenated.csv"))

    if model is not None:
        df_concatenated = df_concatenated.query("model == @model")

    # --- Mask of non completed runs
    groups = set(df_concatenated.groupby("dataset encoder".split()).groups)
    return [x for x in experiments if (x[0], get_acronym(x[1].__str__(), underscore=False)) not in groups]

def remove_concluded_main9(experiments, result_folder, model=None):
    print("Removing completed runs.")

    # --- Load df_concatenated
    try:
        df_concatenated = pd.read_csv(os.path.join(result_folder, "main9_final.csv"))
    except FileNotFoundError:
        df_concatenated = pd.read_csv(os.path.join(result_folder, "_concatenated.csv"))

    if model is not None:
        df_concatenated = df_concatenated.query("model == @model")

    # --- Mask of non completed runs
    groups = set(df_concatenated.groupby("dataset encoder".split()).groups)
    return [x for x in experiments if (x[0], get_acronym(x[1].__str__(), underscore=False)) not in groups]



def remove_concluded_runs(all_experiments, result_folder, repeat_unsuccessful=False):
    """
    Checks for every experiment in all_experiments whether it was already run or not.
    Already run is checked in the result_folder: if the experiment
    is already in df_concatenated, and if the name is in the logs.
    It does NOT check if a file with the name already exists (relies on the logs for that).
    If repeat_unsuccessful, the runs with outcome 2 will be inclded in experiments.

    Automaticaly handles the different versions.
    main6 requires scoring as experimental parameter, while main7 does not.
    """

    # -- preload df_concatenated
    try:
        df_concatenated = pd.read_csv(os.path.join(result_folder, "_concatenated.csv"))
    except FileNotFoundError:
        print(f"df_concatenated not found in {result_folder}")
        df_concatenated = None
    experiments = []

    print("Checking experiments")
    for experiment in tqdm(all_experiments):
        """
        main6 experiments are identified with scoring (among the rest), while main7 experiments are not. 
        Naming styles are also different.
        This function handles both cases.
        """

        if len(experiment) == 7:  # main6 style output files. Handles only the new format.
            main_version = 6
            # warnings.WarningMessage("Only new (acronym) style files are checked")
            dataset, encoder, scaler, cat_imputer, num_imputer, model, scoring = experiment
            exec_log = {
                "exit_status": 0,  # see doc
                "dataset": dataset.name,
                "encoder": get_acronym(encoder.__str__(), underscore=False),
                "model": get_acronym(model.__class__.__name__, underscore=False),
                "scoring": scoring.__name__,
                "error_message": ""
            }
            exp_name = "{}_{}_{}_{}".format(exec_log["dataset"], exec_log["encoder"], exec_log["model"],
                                            exec_log["scoring"])

        elif len(experiment) == 6:  # main7 style output files
            main_version = 7
            dataset, encoder, scaler, cat_imputer, num_imputer, model = experiment
            exec_log = {
                "exit_status": 0,  # 0 = no computation; 1 = success; 2 = fail
                "dataset": dataset.name,
                "encoder": get_acronym(encoder.__str__(), underscore=False),
                "model": get_acronym(model.__class__.__name__, underscore=False),
                "error_message": ""
            }
            exp_name = "{}_{}_{}".format(exec_log["dataset"], exec_log["encoder"], exec_log["model"])
        elif len(experiment) == 5:  # main8 and main9 style output files
            main_version = 8
            dataset, encoder, scaler, cat_imputer, num_imputer = experiment
            exec_log = {
                "exit_status": 0,  # 0 = no computation; 1 = success; 2 = fail
                "dataset": dataset.name,
                "encoder": get_acronym(encoder.__str__(), underscore=False),
                "error_message": ""
            }
            exp_name = "{}_{}".format(exec_log["dataset"], exec_log["encoder"])
        else:
            raise(ValueError("Parameter experiment has wrong number of entries."))

        # -- check in the files
        saveset_name = exp_name + ".csv"
        if os.path.join(result_folder, saveset_name) in glob.glob(os.path.join(result_folder, "*.csv")):
            continue

        # -- check in the logs
        flag = False
        skip_outcomes = [1, ] if repeat_unsuccessful else [1, 2]
        for success in skip_outcomes:
            tmp_log_name = f"{success}_{exp_name}.json"
            if os.path.join(result_folder, "logs", tmp_log_name) in glob.glob(
                    os.path.join(result_folder, "logs", "*.json")):
                flag = True
        if flag:
            continue

        # -- check in concatenated.csv
        if df_concatenated is not None:

            # conditions
            cdat = (df_concatenated.dataset == exec_log["dataset"])
            cenc = (df_concatenated.encoder == get_acronym(exec_log["encoder"], underscore=False))

            ccc = cdat & cenc
            if main_version in (6, 7):
                ccc = ccc & (df_concatenated.model == exec_log["model"])
            if main_version in (6, ):
                ccc = ccc & (df_concatenated.scoring == exec_log["scoring"])

            # if one entry with the required fields exists, the experiment should not be repeated
            if len(df_concatenated.loc[ccc]) > 0:
                continue
        experiments.append(experiment)

    return experiments


def find_bins(universe, weights: np.ndarray, thr):
    """
    Group elements of the univese set into the maximum number of bins so that the sum of 'shares' on each bin exceeds
    threshold
    """
    csum = weights.cumsum()
    inds = len(universe) - 1

    if inds == 0 or csum[inds] < 2 * thr:
        return [universe]

    maxsum = csum[inds]
    cursum = weights[inds]
    i = inds - 1

    inds = np.array((inds, ))
    while (cursum + 10 ** -10) < thr:
        if i == 0:
            inds = np.append(inds, i)
            cursum += weights[i]
        elif (cursum + weights[i] + 10 ** -10) < thr or (cursum + csum[i - 1] + 10 ** -10) < thr:
            inds = np.append(inds, i)
            cursum += weights[i]
        i = i - 1

    if maxsum - cursum < thr:
        return [universe]

    tmp = find_bins(np.delete(universe, inds), np.delete(weights, inds), thr)
    tmp.append(universe[inds])
    return tmp


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


def get_grid_search_space(model):
    out = {}
    if isinstance(model, DecisionTreeClassifier):
        out = {
            "max_depth": [2, 5, None],
        }
    elif isinstance(model, SVC):
        out = {
            "C": [0.1, 1, 10],
            "gamma": [0.01, 0.1, 1, 10]
        }
    elif isinstance(model, KNeighborsClassifier):
        out = {
            "n_neighbors": [2, 5, 10]
        }
    elif isinstance(model, LogisticRegression):
        out = {
            "C": [0.1, 1, 10]
        }
    else:
        raise ValueError(f"Model {repr(model)} is not valid")

    return out


def tune_model(model, X, y, scoring, random_state=1444, n_splits=5, verbose=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GS = GridSearchCV(
            model,
            param_grid=get_grid_search_space(model=model),
            n_jobs=1,
            cv=StratifiedKFold(n_splits=n_splits, random_state=random_state+1, shuffle=True),
            scoring=make_scorer(scoring),
            refit=True,
            verbose=verbose,
            error_score="raise"
        )
        GS.fit(X, y)
    return GS


def get_acronym(string, underscore=True):
    out = ""
    for c in string.split("(")[0]:
        if c.isupper() or c.isdigit():
            out += c
    return out + "_" if underscore else out


if __name__ == "__main__":
    print("Test datasets")
    try:
        shapes
    except NameError:
        shapes = {}
        for k, v in tqdm(DATASETS.items()):
            dataset = get_dataset(v)
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=dataset.default_target_attribute, dataset_format="dataframe"
            )
            shapes[k] = X.shape

    for k, v in sorted(shapes.items(), key=lambda x:x[1][0]):
        print(f"{k:50s}{v}")