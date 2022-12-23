# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 14:41:36 2022

@author: federicom
"""

#TODO: solve the following error
""" 
/home/i40/federicom/.local/lib/python3.8/site-packages/rpy2/robjects/pandas2ri.py:60: UserWarning: Error while trying to 
convert the column "education". Fall back to string conversion. The error is:
  warnings.warn('Error while trying to convert '
"""

"""!!! 
Any iteration of RGLMME before 08.12 is falsely long-running. 
Experiments should be re-run for those. So annoying. 

"""


import contextlib
import datetime
import itertools
import logging
import json
import numpy as np
import os
import pandas as pd
import time
import warnings

from datetime import date
from functools import reduce
from glob import glob
from importlib import reload
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier, early_stopping
from numpy.random import default_rng
from openml.datasets import get_dataset
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from rpy2.robjects.packages import importr
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from stopit import ThreadingTimeout as Timeout
from tqdm import tqdm

import src.utils as u
import src.encoders as e

reload(u)
reload(e)

# suppress any warning (from python AND R)
rpy2_logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

np.random.seed(0)

def main_loop(result_folder,
              dataset, encoder, scaler, cat_imputer, num_imputer, model, scoring, index=0, num_exp=0,
              n_splits=5, random_state=1444, timeout=6000):
    """
    output into logfile:
        0 : no computation
        1 : success
        2 : error raised OR timeout (in which case the error message is empty)
    """

    c1 = isinstance(encoder, e.RGLMMEncoder)
    c2 = isinstance(encoder, e.CVRegularized) and isinstance(encoder.base_encoder, e.RGLMMEncoder)
    c3 = isinstance(encoder, e.CVBlowUp) and isinstance(encoder.base_encoder, e.RGLMMEncoder)
    if c1 or c2 or c3:
        importr("lme4")
        importr("base")
        importr("utils")

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )

    # -- Drop empty columns and lines
    X = X.dropna(axis=0, how="all").dropna(axis=1, how="all")
    y = pd.Series(e.LabelEncoder().fit_transform(
        y[X.index]), name="target")

    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    # -- define output - dataset saving and log
    saveset = pd.DataFrame()
    exp_name = "{}_{}_{}_{}".format(
        dataset.name,
        encoder.__str__().split('(')[0],
        model.__class__.__name__,
        scoring.__name__
    )
    # default is timeout
    exec_log = {
        "exit_status": 0,  # see doc
        "dataset": dataset.name,
        "encoder": encoder.__str__().split('(')[0],
        # "scaler": scaler.__class__.__name__,
        # "cat_imputer": cat_imputer.__str__(),
        # "num_imputer": num_imputer.__str__(),
        "model": model.__class__.__name__,
        "scoring": scoring.__name__,
        "error_message": ""
    }

    saveset_name = exp_name + ".csv"

    # -- define pipeline
    cats = X.select_dtypes(include=("category", "object")).columns
    nums = X.select_dtypes(exclude=("category", "object")).columns

    catpipe = Pipeline([
        ("imputer", cat_imputer),
        ("encoder", encoder)
    ])
    numpipe = Pipeline([
        ("imputer", num_imputer),
        ("scaler", scaler)
    ])

    CT = ColumnTransformer(
        [
            (
                "encoder",
                catpipe,
                cats
            ),
            (
                "scaler",
                numpipe,
                nums
            ),
        ],
        remainder="passthrough"
    )

    pipe = Pipeline([
        ("preproc", CT),
        # ("second_imputation", num_imputer),
        ("model", model)
    ])

    # when using LGBM, searc_space is used just to initialize out
    search_space = u.get_pipe_search_space_one_encoder(model, encoder)

    cv = StratifiedKFold(n_splits=n_splits, random_state=None)

    out = {
        "dataset": dataset.name,
        "encoder": str(encoder),
        "scaler": scaler.__class__.__name__,
        "model": model.__class__.__name__,
        "scoring": scoring.__name__,
        "cv_scores": [],
        "tuning_scores": [],
        "tuning_time": [],
    }
    out.update({
        hpar: [] for hpar in search_space.keys()
    })

    # ---- To know what is going on
    print(f"{index:5}/{num_exp} {datetime.datetime.now().strftime('%d.%m %H:%M')} {out['dataset']:25} {u.get_acronym(out['encoder']):15}\
            {u.get_acronym(out['model']):10} {out['scoring']}")

    # failsafe tuning
    try:
        with Timeout(timeout, swallow_exc=False) as timeout_ctx:
            for tr, te in cv.split(X, y):
                Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

                Xtr, ytr = Xtr.reset_index(drop=True), ytr.reset_index(drop=True)
                Xte, yte = Xte.reset_index(drop=True), yte.reset_index(drop=True)

                # - LGBM tuning - early stopping
                if isinstance(model, LGBMClassifier):

                    start_time = time.time()
                    Xtrtr, Xtrval, ytrtr, ytrval = train_test_split(Xtr, ytr, test_size=0.1, random_state=random_state + 1)

                    # For some reason imputing messes up the indices (resets)
                    Xtrtr, ytrtr = Xtrtr.reset_index(drop=True), ytrtr.reset_index(drop=True)
                    Xtrval, ytrval = Xtrval.reset_index(drop=True), ytrval.reset_index(drop=True)

                    # in order to pass the evaluation set to LGBM, we need to encode it
                    # it has no effect on the overall evaluation, it just makes the code uglier
                    XEtrtr = pipe["preproc"].fit_transform(Xtrtr, ytrtr)
                    XEtrval = pipe["preproc"].transform(Xtrval)

                    # make LGBM silent again
                    with contextlib.redirect_stdout(None):
                        pipe["model"].fit(
                            XEtrtr, ytrtr,
                            eval_set=[(XEtrval, ytrval)],
                            eval_metric=u.get_lgbm_scoring(scoring),
                            callbacks=[early_stopping(50, first_metric_only=True)]
                        )

                    # just to keep the same name used below. Not an instance of BayesianSearchCV
                    search_space = {
                        'model__n_estimators': None
                    }
                    if isinstance(encoder, e.SmoothedTE):
                        search_space["preproc__encoder__encoder__w"] = None

                    BS = pipe
                    tuning_result = {
                        "best_score": scoring(ytrval, pipe.predict(Xtrval)),
                        "time": time.time() - start_time,
                        "best_params": {
                            "model__n_estimators": pipe["model"].best_iteration_,
                            "preproc__encoder__encoder__w": e.SmoothedTE().w  # TODO: default - no tuning
                        }
                    }
                # - tuning
                else:
                    tuning_result, BS = u.tune_pipe(
                        pipe,
                        Xtr, ytr,
                        search_space,
                        make_scorer(scoring),
                        n_jobs=1, n_splits=3, max_iter=5
                    )

                # - save
                out["cv_scores"].append(scoring(yte, BS.predict(Xte)))
                out["tuning_scores"].append(tuning_result["best_score"])
                out["tuning_time"].append(tuning_result["time"])
                for hpar in search_space.keys():
                    out[hpar].append(tuning_result["best_params"][hpar])

            saveset = pd.concat([saveset, pd.DataFrame(out)], ignore_index=True)

    except Exception as tuning_error:
        exec_log["exit_status"] = 2
        exec_log["error_message"] = str(tuning_error)

    # if no Exception was raised -> success
    if exec_log["exit_status"] == 0:
        saveset.to_csv(os.path.join(result_folder, saveset_name))
        exec_log["exit_status"] = 1

    # remove default time-out log
    log_name = f'{exec_log["exit_status"]}_{exp_name}.json'
    try:
        with open(os.path.join(result_folder, "logs", log_name), "w") as fw:
            json.dump(exec_log, fw)
    except Exception as log_error:
        exec_log["log_error_message"] = str(log_error)
        with open(os.path.join(os.getcwd(), "logs", log_name), "w") as fw:
            json.dump(exec_log, fw)
    return

# ---- Execution

# -- import R libraries
rlibs = {
    "lme4": importr("lme4"),
    "base": importr("base"),
    "utils": importr("utils"),
}
rlibs = None

random_state = 1444

std = [e.BinaryEncoder(), e.CatBoostEncoder(), e.CountEncoder(), e.DropEncoder(), e.RGLMMEncoder(rlibs=rlibs),
       e.OneHotEncoder(), e.SmoothedTE(), e.TargetEncoder(), e.MinHashEncoder()]
cvglmm = [e.CVRegularized(e.RGLMMEncoder(rlibs=rlibs), n_splits=ns) for ns in [2, 5, 10]]
cvte = [e.CVRegularized(e.TargetEncoder(), n_splits=ns) for ns in [2, 5, 10]]
buglmm = [e.CVBlowUp(e.RGLMMEncoder(rlibs=rlibs), n_splits=ns) for ns in [2, 5, 10]]
bute = [e.CVBlowUp(e.TargetEncoder(), n_splits=ns) for ns in [2, 5, 10]]
dte = [e.Discretized(e.TargetEncoder(), how="minmaxbins", n_bins=nb) for nb in [2, 5, 10]]
binte = [e.PreBinned(e.TargetEncoder(), thr=thr) for thr in [1e-3, 1e-2, 1e-1]]
encoders = reduce(lambda x, y: x+y, [std, cvglmm, cvte, buglmm, bute, dte, binte])

scalers = [
    u.RobustScaler(),
]

cat_imputers = [
    e.DFImputer(u.SimpleImputer(strategy="most_frequent"))
]
num_imputers = [
    e.DFImputer(u.SimpleImputer(strategy="median"))
]

models = [
    u.DecisionTreeClassifier(random_state=random_state+2),
    u.LGBMClassifier(random_state=random_state+3, n_estimators=3000, metric="None"),
    u.SVC(random_state=random_state+4),
    u.KNeighborsClassifier()
]

scorings = [
    u.accuracy_score,
    u.roc_auc_score,
    u.f1_score
]

kwargs = {
    "n_splits": 5,
    "random_state": random_state,
    "timeout": 36000,
}

gbl_log = {
    "datetime": date.today().__str__(),
    "arguments": kwargs,
    "datasets": list(u.DATASETS.values()),
    "failed_datasets": list(),
    "encoders": [enc.__str__().split('(')[0] for enc in encoders],
    # "scalers": [sc.__class__.__name__ for sc in scalers],
    # "cat_imputers": [ci.__str__() for ci in cat_imputers],
    # "num_imputers": [ni.__str__() for ni in num_imputers],
    "models": [m.__class__.__name__ for m in models],
    "scorings": [s.__name__ for s in scorings],
}

test = True
update_previous_experiment = True

if __name__ == "__main__":
    experiment_name = "final"
    result_folder = os.path.join(u.RESULT_FOLDER, experiment_name)
    try:
        os.mkdir(u.RESULT_FOLDER)
    except FileExistsError:
        pass

    # TODO: this creation is not robust at all. Make another function

    same_name_exps = glob(os.path.join(u.RESULT_FOLDER, f"{experiment_name}*"), recursive=False)
    experiment_name = f"{experiment_name}_{len(same_name_exps)}"
    result_folder = os.path.join(u.RESULT_FOLDER, experiment_name)

    if update_previous_experiment:
        print(f"Updating {experiment_name}")
    else:
        # make a new experiment directory
        try:
            os.mkdir(result_folder)
            os.mkdir(os.path.join(result_folder, "logs"))
        except FileExistsError:
            same_name_exps = glob(os.path.join(u.RESULT_FOLDER, f"{experiment_name}*"), recursive=False)
            experiment_name = f"{experiment_name}_{len(same_name_exps)}"
            result_folder = os.path.join(u.RESULT_FOLDER, experiment_name)
            os.mkdir(result_folder)
            os.mkdir(result_folder + '/logs')

    if not test:
        print("Preloading datasets")
        # -- Load dataset
        datasets = []
        for did in tqdm(u.DATASETS.values()):
            try:
                dataset = get_dataset(did)
            except:
                gbl_log["datasets"].remove(did)
                gbl_log["failed_datasets"].append(did)
            else:
                datasets.append(dataset)

    # -- Testing?
    else:
        print('-'*20 + "Test activated" + '-'*20)
        datasets = [get_dataset(u.DATASETS["amazon_employee_access"])]
        encoders = [e.SmoothedTE(), e.TargetEncoder()]
        models = [u.LGBMClassifier()]
        scorings = [u.roc_auc_score]

    # -- Experiment
    rng = default_rng() # permute to avoid heavy encoders being used all together
    nj = 1 if test else -1


    all_experiments = list(rng.permutation(tuple(itertools.product(datasets,
                                                                   encoders,
                                                                   scalers,
                                                                   cat_imputers,
                                                                   num_imputers,
                                                                   models,
                                                                   scorings))))

    experiments = u.remove_concluded_runs(all_experiments, result_folder)


    Parallel(n_jobs=nj, verbose=0)(
        delayed(main_loop)(result_folder, dataset, encoder, scaler, cat_imputer, num_imputer, model, scoring,
                           index=index, num_exp=len(experiments), **kwargs)
        for (index, (dataset,
                     encoder,
                     scaler,
                     cat_imputer,
                     num_imputer,
                     model,
                     scoring))
        in enumerate(experiments)
    )
