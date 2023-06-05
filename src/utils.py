# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 10:29:35 2022

@author: federicom
"""

import contextlib
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import warnings

from functools import reduce
from itertools import product
from numpy.random import default_rng
from openml.datasets import get_dataset
from pathlib import Path
from scipy.stats import kendalltau
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    make_scorer,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from tqdm import tqdm
from types import MappingProxyType
from typing import Tuple, Callable, Iterable, List, Union, Literal, Sized

# models
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ----

# --- Directories
BASE_DIR = Path(r"C:\Data\EncoderBenchmarking_results")
RESULTS_DIR = BASE_DIR / "ExperimentalResults"
RANKINGS_DIR = BASE_DIR / "Rankings"
SENSITIVITY_DIR = BASE_DIR / "Sensitivity"
FIGURES_DIR = BASE_DIR / "Figures"

# --- Datasets
DATASETS = MappingProxyType({
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
    'amazon_employee_access': 4135,
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
    'mushroom': 43922
})
DATASETS_SMALL = MappingProxyType({
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
})
LEFT_DATASETS = MappingProxyType({
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
})

# --- Better names
AGGREGATION_NAMES = MappingProxyType({
    "rank_mean": "RM",
    "rank_median": "RMd",
    "rank_numbest": "RB",
    "rank_num_worst": "RW",
    "rank_numworst": "RW",
    "rank_kemeny": "RK",
    "rank_nemenyi": "RN",
    "rank_nemenyi_0.01": "RN01",
    "rank_nemenyi_0.05": "RN05",
    "rank_nemenyi_0.10": "RN10",
    "qual_mean": "QM",
    "qual_median": "QMd",
    "qual_thrbest_0.95": "QT95",
    "qual_rescaled_mean": "QR"
})
AGGREGATION_LATEX = MappingProxyType({
    "RM": "R-M",
    "RMd": "R-Md",
    "RB": "R-B",
    "RW": "R-W",
    "RK": "R-Kem",
    "RN": "R-Nem",
    "RN01": "R-Nem$_{0.01}$",
    "RN05": "R-Nem$_{0.05}$",
    "RN10": "R-Nem$_{0.1}$",
    "QM": "Q-M",
    "QMd": "Q-Md",
    "QT95": r"Q-Th$_{0.95}$",
    "QR": "Q-RM"

})
SIMILARITY_LATEX = MappingProxyType({
    "taub": r"$\tau_b$",
    "taub_std": r"$\sigma(\tau_b)$",
    "ptaub": r"$p \tau_b$",
    "ptaub_std": r"$\sigma(p \tau_b)$",
    "rho": r"$\rho$",
    "rho_std": r"$\sigma(\rho)$",
    "agrbest": r"$\alpha_{best}$",
    "agrbest_std": r"$\sigma(\alpha_{best})$",
    "agrworst": r"$\alpha_{worst}$",
    "agrworst_std": r"$\sigma(\alpha_{worst})$",
})
ENCODER_LATEX = MappingProxyType({
    "BE": "Bin",
    "BUCV2TE": "BU$_{2}$MT",
    "BUCV5TE": "BU$_{5}$MT",
    "BUCV10TE": "BU$_{10}$MT",
    "BUCV2RGLMME": "BU$_{2}$GLMM",
    "BUCV5RGLMME": "BU$_{5}$GLMM",
    "BUCV10RGLMME": "BU$_{10}$GLMM",
    "CBE": "CB",
    "CE": "Count",
    "CV2TE": "CV$_{2}$MT",
    "CV5TE": "CV$_{5}$MT",
    "CV10TE": "CV$_{10}$MT",
    "CV2RGLMME": "CV$_{2}$GLMM",
    "CV5RGLMME": "CV$_{5}$GLMM",
    "CV10RGLMME": "CV$_{10}$GLMM",
    "DE": "Drop",
    "DTEM2": "D$_{2}$",
    "DTEM5": "D$_{5}$",
    "DTEM10": "D$_{10}$",
    "ME01E": "ME$_{0.1}$",
    "ME1E": "ME$_{1}$",
    "ME10E": "ME$_{10}$",
    "MHE": "MH",
    "OE": "Ord",
    "OHE": "OH",
    "PBTE0001": "PB$_{0.001}$",
    "PBTE001": "PB$_{0.01}$",
    "PBTE01": "PB$_{0.1}$",
    "RGLMME": "GLMM",
    "SE": "Sum",
    "TE": "MT",
    "WOEE": "WoE",
})
FACTOR_LATEX = MappingProxyType({
    "model": MappingProxyType({
        "DTC": "DT",
        "KNC": "k-NN",
        "LGBMC": "LGBM",
        "LR": "LogReg",
        "SVC": "SVM"
    }),
    "tuning": MappingProxyType({
        "full": "full",
        "model": "model",
        "no": "no"
    }),
    "scoring": MappingProxyType({
        "ACC": "Acc",
        "F1": "F$_1$",
        "AUC": "AUC"
    }),
    "interpretation": AGGREGATION_LATEX,
    "aggregation": AGGREGATION_LATEX
})
AGGREGATION_PALETTE = MappingProxyType({
    "QM": "#ff0000",
    "QMd": "#d40909",
    "QR": "#af0d0d",
    "QT95": "#8e0a0a",
    "RB": "#0096ff",
    "RM": "#0a79c7",
    "RMd": "#0e5e96",
    "RW": "#114467",
    "RN01": "#00253c",
    "RN05": "#00303c",
    "RN10": "#00323c",
    "RK": None
})


# --- Functions
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_disjoint_samples(S,
                         n_samples: int,
                         sample_size: int,
                         seed: int = 0,
                         bootstrap: bool = False) -> list[set]:
    """
    S is the set of objects from which to sample (Iterable and Sized)
    n_samples is the number of disjoint subsets of S to return
    sample_size is the number of items in each sample
    seed is the temporary numpy random seed
    bootstrap is used ONLY IF n_samples*sample_size > len(S)
    """
    if n_samples * sample_size > len(S):
        if not bootstrap:
            raise ValueError("Not enough elements in S.")
        else:
            return get_disjoint_samples_bootstrap(set(S), n_samples, sample_size, seed)
    S_ = set(S)
    with temp_seed(seed):
        out = []
        for _ in range(n_samples):
            Snew = set(np.random.choice(list(S_), sample_size, replace=False))
            out.append(Snew)
            S_ = S_ - Snew
    return out


def get_disjoint_samples_bootstrap(S: set,
                                   n_samples: int,
                                   sample_size: int,
                                   seed: int = 0) -> list[np.array]:
    """
    Gets the maximum number of samples possible for n_samples, then bootstrap these samples to reach sample_size
    """

    to_sample = int(len(S) / n_samples)
    with temp_seed(seed):
        out = []
        for _ in range(n_samples):
            # get the samples, remove them from S
            Snew = set(np.random.choice(list(S), to_sample, replace=False))
            S = S - Snew

            # bootstrap the sample
            Snew = np.random.choice(list(Snew), sample_size, replace=True)

            out.append(Snew)

    return out


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
    i = 0
    while len(presorted_indices) > 0:
        print(i)
        indices.append(presorted_indices.pop())
        indices.extend(presorted_indices[:num_small_per_large])
        presorted_indices = presorted_indices[num_small_per_large:]
        i += 1

    return [experiments[i] for i in indices]


def remove_failed_main8(experiments, result_folder):
    """
    logs are in the form "success_dataset_encoder"
    """

    print("Removing failed runs.")

    # --- Find the failed logs
    failed_runs = [os.path.split(x)[-1] for x in glob.glob(os.path.join(result_folder, "logs", "2*.json"))]

    return [x for x in experiments if
            f"2_{x[0]}_{get_acronym(x[1].__str__(), underscore=False)}.json" not in failed_runs]


def remove_failed_main9(experiments, result_folder):
    """
    logs are in the form "success_dataset_encoder"
    """

    print("Removing failed runs.")

    # --- Find the failed logs
    failed_runs = [os.path.split(x)[-1] for x in glob.glob(os.path.join(result_folder, "logs", "2*.json"))]

    return [x for x in experiments if
            f"2_{x[0]}_{get_acronym(x[1].__str__(), underscore=False)}.json" not in failed_runs]


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
            raise (ValueError("Parameter experiment has wrong number of entries."))

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
            if main_version in (6,):
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

    inds = np.array((inds,))
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
            "C": [0.1, 1],
            "gamma": [0.1, 1, 10]
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
            cv=StratifiedKFold(n_splits=n_splits, random_state=random_state + 1, shuffle=True),
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


# --- Load experimental results and manipulations

def load_df() -> pd.DataFrame:
    """
    Load evaluations dataframe from hard-coded path
    """
    return pd.read_csv(RESULTS_DIR / "final.csv")


def load_rf() -> pd.DataFrame:
    """
    Load rank functions from hard-coded path
    """
    return pd.read_csv(RANKINGS_DIR / "rank_function_from_average_cv_score.csv", index_col=0, header=[0, 1, 2, 3])


def load_df_rf() -> Tuple[pd.DataFrame, ...]:
    return load_df(), load_rf()


def load_aggrf() -> pd.DataFrame:
    """
    Load dataframe of aggregate rankings from hard-coded path
    """

    return pd.read_csv(RANKINGS_DIR / "aggregated_ranks_from_average_cv_score.csv", index_col=0,
                       header=[0, 1, 2, 3])


def load_agg_similarities() -> dict[str, pd.DataFrame]:
    """
    Loads similarity between aggregations dataframes in wide format.
    The output dict format is "filename : dataframe"
    """
    out = {}
    for path in glob.glob(str(RANKINGS_DIR / "pw_AGG*.csv")):
        out[Path(path).name] = pd.read_csv(path, index_col=[0, 1, 2, 3], header=[0, 1, 2, 3])
    return out


def load_similarity_dataframes() -> dict:
    """
    Load similarity square dataframes, with similarity computed with Kendall's tau_b, agreement on the best and on the
        worst tiers.
    Each cell is indexed by a pair of ("dataset", "model", "tuning", "scoring") tuples and contains the rank similarity
        of the corresponding rankings.
    """

    out = {}
    # filter out aggregated rankings
    for path in set(glob.glob(str(RANKINGS_DIR / "pw*.csv"))) - set(glob.glob(str(RANKINGS_DIR / "pw_AGG*.csv"))):
        tmp = pd.read_csv(path, index_col=[0, 1, 2, 3], header=[0, 1, 2, 3])
        try:
            tmp.index = tmp.index.astype(object).rename(["dataset", "model", "tuning", "scoring"])
            tmp.columns = tmp.columns.astype(object).rename(["dataset", "model", "tuning", "scoring"])
        except ValueError:
            continue
        else:
            out[Path(path).name] = tmp

    return out


def load_sample_similarity_dataframe(tuning) -> pd.DataFrame:
    try:
        return pd.read_csv(RANKINGS_DIR / f"sample_sim_{tuning}.csv")
    except FileNotFoundError:
        print(f"'sample_sim_{tuning}.csv' not found in {RANKINGS_DIR}.")
        return pd.DataFrame()


def format_df_sim(df_sim: pd.DataFrame) -> pd.DataFrame:
    tmp = df_sim.copy().rename(columns=SIMILARITY_LATEX)
    for col in tmp.columns:
        if "model" in col:
            tmp[col] = tmp[col].map(FACTOR_LATEX["model"])
        elif "tuning" in col:
            tmp[col] = tmp[col].map(FACTOR_LATEX["tuning"])
        elif "scoring" in col:
            tmp[col] = tmp[col].map(FACTOR_LATEX["scoring"])
    return tmp


def pairwise_similarity_wide_format(aggrf: pd.DataFrame,
                                    simfuncs: Iterable[Callable[[Iterable, Iterable], float]],
                                    shared_levels: Union[slice, Iterable] = slice(-1)) -> List[pd.DataFrame]:
    """
    simfunc is assumed to be a similarity between rankings, i.e., a symmetric function of two rankings with
        maximum in 1.
    The output is an upper triangular dataframe
    """

    factor_combinations = aggrf.columns.sort_values()

    out = []
    for simfunc in simfuncs:
        mat = np.full((len(aggrf.columns), len(aggrf.columns)), np.nan)
        for (i1, col1), (i2, col2) in product(enumerate(factor_combinations), repeat=2):
            if i1 >= i2:
                continue
            # compare only if they have equal model, tuning, scoring
            if not np.alltrue(np.array(col1)[shared_levels] == np.array(col2)[shared_levels]):
                continue
            try:
                mat[i1, i2] = simfunc(aggrf[col1], aggrf[col2])
            except Exception as error:
                raise Exception(f"The following combinations gave problems: {col1}, {col2}.{error}")

        # make symmetric and add diagonal
        np.fill_diagonal(mat, 1)
        out.append(pd.DataFrame(mat, index=factor_combinations, columns=factor_combinations))

    return out


def pairwise_level_comparison_long_format(df_sim: pd.DataFrame, comparison_level: str):
    """
    df_sim is a square dataframe whose cells are indexed by tuples.
    df_sim.index and df_sim.columns are pd.MultiIndex objects.

    Fix all levels except 'comparison_level', thwn compare different values in the 'comparison_levle' level in the
        columns and index of df_sim.
    Finally, convert in long format.
    The output is a pd.DataFrame with schema (where level_i's are the fixed levels)
        (level_0, level_1, level_2, level_3, comparison_level_1, comparison_level_2, similarity)
    """

    levels = list(df_sim.columns.names)

    try:
        levels.remove(comparison_level)
    except ValueError:
        raise ValueError(f"{comparison_level} is not a valid comparison level. Valid levels: {levels}")

    l = []
    for idx, temp in df_sim.groupby(level=levels):
        # cross section
        t = temp.xs(tuple(str(x) for x in idx), level=levels, axis=1)
        # change names
        t.index = t.index.rename({comparison_level: f"{comparison_level}_1"})
        t.columns.name = f"{comparison_level}_2"
        # stack: indexed matrix -> dataframe
        t = t.stack().reorder_levels(levels + [f"{comparison_level}_1", f"{comparison_level}_2"])
        l.append(t)

    return pd.concat(l, axis=0).rename("similarity").to_frame().reset_index()


def join_wide2long(named_dataframes: dict[str: pd.DataFrame], comparison_level: str) -> pd.DataFrame:
    """
    named_dataframes = {similarity_name : similarity matrix in square format}

    Converts all values of named_dataframes to long format, then joins them with appropriate column names.
    All values of named_dataframes have as columns a pd.MultiIndex objects with the same levels, 'levels'.

    output has schema 'levels' + ['comparison_level'_1, 'comparison_level'_2] + list(named_dataframes.keys())
    """
    levels = ""
    for similarity, df_sim in named_dataframes.items():
        if len(levels) > 0:
            if levels != list(df_sim.columns.names):
                raise ValueError("The entered dataframes must have the same column levels.")
        levels = list(df_sim.columns.names)

    try:
        levels.remove(comparison_level)
    except ValueError:
        raise ValueError(f"{comparison_level} is not a valid comparison level. Valid levels: {levels}")
    levels.extend([f"{comparison_level}_1", f"{comparison_level}_2"])

    joined_df_sim = reduce(
        lambda l, r: l.merge(r, on=levels, how="inner"),
        [pairwise_level_comparison_long_format(df_sim, comparison_level).rename(columns={"similarity": similarity})
         for similarity, df_sim in named_dataframes.items()]
    )

    return joined_df_sim


def index_sorted_by_median(df, groupby_col, target_col):
    return df.groupby(groupby_col)[target_col].median().sort_values().index


def sorted_boxplot_horizontal(data, x, y, **kwargs):
    return sns.boxplot(data, x=x, y=y,
                       order=index_sorted_by_median(data, groupby_col=y, target_col=x),
                       **kwargs)


def sorted_boxplot_vertical(data, x, y, **kwargs):
    return sns.boxplot(data, x=x, y=y,
                       order=index_sorted_by_median(data, groupby_col=x, target_col=y),
                       **kwargs)


def plot_long_format_similarity_dataframe(df_sim: pd.DataFrame,
                                          similarity: str,
                                          comparison_level: str,
                                          plot_type: Literal["boxplots", "heatmap"],
                                          latex_name: Union[dict, MappingProxyType] = SIMILARITY_LATEX,
                                          directory: Union[Path, str] = FIGURES_DIR,
                                          save_plot: bool = False,
                                          show_plot: bool = False,
                                          draw_points: bool = True,
                                          color: str = "black",  # boxplots
                                          cmap: str = "rocket",  # heatmap
                                          figsize_inches=None,
                                          fontsize=10,
                                          annot_fontsize=10) -> None:
    """
    df_sim has schema (dataset, tuning, scoring, factor_1, factor_2, taub, agrbest, agrworst),
        each entry represents the correlation between the ranking produced with model1 and model2, with the rest fixed
    """

    if plot_type not in ["boxplots", "heatmap"]:
        raise ValueError(f"Plot type {plot_type} not implemented.")

    # set default plot size
    if plot_type == "boxplots":
        figsize_inches = figsize_inches or (3.25, 5)
    elif plot_type == "heatmap":
        figsize_inches = figsize_inches or (5, 5)

    # rename columns to their latex_name for pertty plotting
    df_sim = df_sim.rename(columns=latex_name)
    similarity_ = similarity
    similarity = latex_name[similarity]

    # --- Plotting
    if plot_type == "boxplots":
        # IEEE column width = 3.25 inches; dpi is 300, 600 for black&white
        grid = sns.FacetGrid(df_sim, col=f"{comparison_level}_2", row=f"{comparison_level}_1",
                             margin_titles=True, sharex="all", sharey="all",
                             aspect=1, ylim=(-1, 1) if similarity == "taub" else (0, 1))
        grid.set_titles(row_template="{row_name}", col_template="{col_name}")

        if draw_points:
            grid.map_dataframe(sns.stripplot, y=similarity, color=color, size=1)
        grid.map_dataframe(sns.boxplot, y=similarity, color=color, boxprops=dict(alpha=0.5), fliersize=0.5)

        grid.fig.set_size_inches(*figsize_inches)
        grid.fig.tight_layout()

        if save_plot:
            grid.savefig(directory / f"{plot_type}_{comparison_level}_{similarity_}.pdf", dpi=600)
            print(f"Saved figure in {directory}/{plot_type}_{comparison_level}_{similarity_}.pdf")

    elif plot_type == "heatmap":
        """
        First, compute the median, min, and max similarity. 
        Then, plot the median. 
        """

        cl = [f"{comparison_level}_1", f"{comparison_level}_2"]
        median_similarity = df_sim[cl + [similarity]].groupby(cl).median().reset_index() \
            .pivot(index=cl[0], columns=cl[1]) \
            .droplevel([0], axis=1)

        fig, ax = plt.subplots(1, 1, figsize=figsize_inches)
        g = sns.heatmap(median_similarity, annot=True, ax=ax,
                        vmin=-1 if similarity_ in {"taub", "rho"} else 0,
                        vmax=1,
                        cmap=cmap,
                        square=True,
                        cbar=False,
                        annot_kws={"fontsize": annot_fontsize})
        g.set(xlabel=None, ylabel=None)
        g.set_xticklabels(
            g.get_xticklabels(),
            rotation=90,
            horizontalalignment='right',
            fontweight='light',
            fontsize=fontsize
        )
        g.set_yticklabels(
            g.get_yticklabels(),
            rotation=0,
            horizontalalignment='right',
            fontweight='light',
            fontsize=fontsize
        )

        plt.tight_layout(pad=0.5)

        if save_plot:
            plt.savefig(directory / f"{plot_type}_{comparison_level}_{similarity_}.pdf", dpi=600)
            print(f"Saved figure in {directory}/{plot_type}_{comparison_level}_{similarity_}.pdf")

    if show_plot:
        plt.show()

    return g


def heatmap_longformat_multisim(df_sim: pd.DataFrame,
                                similarities: list,
                                comparison_level: str,
                                fontsize=10,
                                annot_fontsize=10,
                                fmt: str = ".1f",
                                cmaps: tuple = (sns.color_palette("Reds", as_cmap=True),
                                                sns.color_palette("Blues", as_cmap=True)),
                                save_plot: bool = True,
                                show_plot: bool = True,
                                ax=None,
                                summary_statistic: Literal["mean", "median"] = "mean",
                                ):
    """
    Gets and heatmap of the similarity dataframe df_im in long format.
    """

    eye = np.eye(df_sim[f"{comparison_level}_1"].nunique())
    eye[eye == 0] = np.nan
    diag = pd.DataFrame(eye,
                        index=df_sim[f"{comparison_level}_1"].unique(),
                        columns=df_sim[f"{comparison_level}_1"].unique())

    # diagonal
    ax = sns.heatmap(diag,
                     annot=False, ax=ax,
                     cmap=sns.color_palette("Greys", as_cmap=True),
                     vmin=0, vmax=1,
                     square=True,
                     cbar=False,
                     fmt=fmt,
                     annot_kws={"fontsize": annot_fontsize})

    for i, (similarity, cmap) in enumerate(zip(similarities, cmaps)):
        cl = [f"{comparison_level}_1", f"{comparison_level}_2"]
        aggsim = df_sim[cl + [similarity]].groupby(cl).agg(summary_statistic).reset_index() \
            .pivot(index=cl[0], columns=cl[1]) \
            .droplevel([0], axis=1) \
            .rename(index=FACTOR_LATEX[comparison_level],
                    columns=FACTOR_LATEX[comparison_level])

        # remove diagonal
        tmp = aggsim.to_numpy()
        np.fill_diagonal(tmp, np.nan)
        aggsim = pd.DataFrame(tmp, index=aggsim.index, columns=aggsim.columns)

        if i == 1:
            aggsim = aggsim.T

        ax = sns.heatmap(aggsim, annot=True, ax=ax,
                         vmin=0 if similarity in ["taub", "rho"] else 0,
                         vmax=1,
                         cmap=cmap,
                         square=True,
                         cbar=False,
                         fmt=fmt,
                         annot_kws={"fontsize": annot_fontsize})

    ax.set(xlabel=None, ylabel=None)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        # horizontalalignment='right',
        fontweight='light',
        fontsize=fontsize
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        # horizontalalignment='right',
        fontweight='light',
        fontsize=fontsize
    )

    plt.tight_layout(pad=0.5)

    if save_plot:
        plt.savefig(FIGURES_DIR / f"heatmap_interpretation_{similarities}.pdf", dpi=600)

    if show_plot:
        plt.show()


def lineplot_longformat_sample_sim(df_sim, similarity,
                                   save_plot=False, show_plot=False,
                                   # factor_hue: Iterable = tuple("interpretation"),
                                   hue: str = "interpretation",
                                   **kwargs):
    """
    factor_hue makes sense if multiple factors are used in the hue.
    As I don't think I'll use that in the plots, use just one factor.
    """

    df_sim = df_sim.copy()

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))

    df_sim[hue] = df_sim[hue].map(FACTOR_LATEX[hue])

    markers = dict(zip(df_sim[hue].unique(),
                       ["o", "v", "^", "<", "s", "P", "*", "X", "D"]))

    ax = sns.lineplot(df_sim, x="sample_size", y=similarity,
                      style=hue,
                      hue=hue,
                      markers=markers,
                      ax=ax,
                      **kwargs
                      # palette=sns.light_palette("grey", n_colors=np.prod(df_sim.nunique()[hue]))
                      )
    ax.set_xticks([5, 10, 15, 20, 25])

    # 'interpretation' is already in its correct form, but the other handles for hue have to be changed
    ax.legend(loc="lower right",
              ncols=3,
              fontsize=7)
    ax.set_ylabel(SIMILARITY_LATEX[similarity])

    fig.tight_layout(pad=0.5)

    if show_plot:
        plt.show()

    if save_plot:
        fig.savefig(FIGURES_DIR / f"lineplot_sample_{hue}_{similarity}.pdf", dpi=600)
        print(f"Saved figure in {FIGURES_DIR}/lineplot_sample_{hue}_{similarity}.pdf")


def boxplots_longformat_sample_sim(df_sim, similarity,
                                   save_plot=False, show_plot=False,
                                   # factor_hue: Iterable = tuple("interpretation"),
                                   hue: str = "interpretation"):
    """
    factor_hue makes sense if multiple factors are used in the hue.
    As I don't think I'll use that in the plots, use just one factor.
    """

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))

    df_sim[hue] = df_sim[hue].map(FACTOR_LATEX[hue])

    sns.boxplot(df_sim, x="sample_size", y=similarity,
                # hue=df_sim[factor_hue].apply(tuple, axis=1),
                hue=hue,
                ax=ax,
                fliersize=1,
                palette=sns.light_palette("grey", n_colors=np.prod(df_sim.nunique()[hue])),
                medianprops=dict(color="red", linewidth=1),
                )

    # sns.move_legend(ax, loc="upper right", bbox_to_anchor=(1.4, 1))

    # 'interpretation' is already in its correct form, but the other handles for hue have to be changed
    ax.legend(loc="lower right",
              ncols=3,
              fontsize=7)
    ax.set_ylabel(SIMILARITY_LATEX[similarity])

    fig.tight_layout(pad=0.5)

    if show_plot:
        plt.show()

    if save_plot:
        fig.savefig(FIGURES_DIR / f"boxplot_sample_{hue}_{similarity}.pdf", dpi=600)
        print(f"Saved figure in {FIGURES_DIR}/boxplot_sample_{hue}_{similarity}.pdf")
