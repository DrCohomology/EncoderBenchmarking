# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 14:41:36 2022

@author: federicom

ONLY CLASSIFICATION

write Pargent
!!! consider switching to hyperopt for hpar tuning
"""

import contextlib
import itertools
import json
import numpy as np
import os
import pandas as pd
import time

from catboost import CatBoostClassifier
from datetime import date
from glob import glob
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from openml.datasets import get_dataset
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline

import src.utils as u
import src.encoders as e

import warnings
warnings.filterwarnings("ignore")

np.random.seed(43)

# os.chdir('C:/Users/federicom/Documents/Github/EncoderComparison')

# ---- Single iteration

def main_loop(result_folder, dataset, encoder, scaler, cat_imputer, num_imputer, model, scoring, n_splits=5, random_state=1444):
    """
    output into logfile:
        0 : no computation
        1 : success
        2 : partial success
        3 : failed dumping
        -1 : failed

    """

    if isinstance(encoder, str) and not isinstance(model, CatBoostClassifier):
        return

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
    exp_name = "{}_{}_{}_{}_{}_{}_{}".format(
        dataset.name,
        encoder.__str__().split('(')[0],
        scaler.__class__.__name__,
        cat_imputer.__str__(),
        num_imputer.__str__(),
        model.__class__.__name__,
        scoring.__name__
    )
    exec_log = {
        "exit_status": 0, # see doc
        "dataset": dataset.name,
        "encoder": encoder.__str__().split('(')[0],
        "scaler": scaler.__class__.__name__,
        "cat_imputer": cat_imputer.__str__(),
        "num_imputer": num_imputer.__str__(),
        "model": model.__class__.__name__,
        "scoring": scoring.__name__,
        "error_message": ""
    }

    saveset_name = exp_name + ".csv"
    
    # -- Close iteration if the run was already concluded
    if os.path.join(result_folder, saveset_name) in glob(os.path.join(result_folder, "*.csv")):
        return
    
    # log_name = exp_name + ".json"

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
    # failsafe tuning
    try:
        for tr, te in cv.split(X, y):
            Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

            Xtr, ytr = Xtr.reset_index(drop=True), ytr.reset_index(drop=True)
            Xte, yte = Xte.reset_index(drop=True), yte.reset_index(drop=True)

            # - LGBM tuning - early stopping
            if isinstance(model, LGBMClassifier):

                start_time = time.time()
                Xtrtr, Xtrval, ytrtr, ytrval = train_test_split(Xtr, ytr, test_size=0.1, random_state=random_state+1)

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
                BS = pipe
                tuning_result = {
                    "best_score": scoring(ytrval, pipe.predict(Xtrval)),
                    "time": time.time() - start_time,
                    "best_params": {
                        "model__n_estimators": pipe["model"].best_iteration_
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
        try:
            exec_log["error_message"] = str(tuning_error)
            saveset.to_csv(os.path.join(result_folder, saveset_name))
        except Exception as dumping_error:
            exec_log["dumping_error_message"] = str(dumping_error)
            saveset.to_csv(os.path.join(os.getcwd(), saveset_name))
            exec_log["exit_status"] = 3
        else:
            exec_log["exit_status"] = 2

    saveset.to_csv(os.path.join(result_folder, saveset_name))

    # if no Exception was raised -> success
    if exec_log["exit_status"] == 0:
        exec_log["exit_status"] = 1

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

random_state = 1444

encoders = [
    # "default",  # only usable with CatBoostClassifier
    e.BinaryEncoder(),
    e.CatBoostEncoder(),
    e.CountEncoder(),
    e.CollapseEncoder(),
    e.GLMMEncoder(),
    e.OneHotEncoder(),
    e.SmoothedTE(),
    e.TargetEncoder(),
] + [
    e.CVRegularized(e.GLMMEncoder(handle_missing="return_nan",
                    handle_unknown="return_nan"), n_splits=ns)
    for ns in range(2, 11, 4)
] + [
    e.CVRegularized(e.TargetEncoder(default=np.nan), n_splits=ns)
    for ns in range(2, 11, 4)
] + [
    e.CVBlowUp(e.GLMMEncoder(), n_splits=ns)
    for ns in range(2, 11, 4)
] + [
    e.CVBlowUp(e.TargetEncoder(), n_splits=ns)
    for ns in range(2, 11, 4)
] + [
    e.Discretized(e.TargetEncoder(), how="minmaxbins", n_bins=nb)
    for nb in range(2, 11, 4)
]

scalers = [
    u.RobustScaler(),
    # e.CollapseEncoder()
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
    "n_splits" : 5,
    "random_state": random_state
}

gbl_log = {
    "datetime": date.today().__str__(),
    "arguments": kwargs,
    "datasets": list(u.DATASETS.values()),
    "failed_datasets": list(),
    "encoders": [enc.__str__().split('(')[0] for enc in encoders],
    "scalers": [sc.__class__.__name__ for sc in scalers],
    "cat_imputers": [ci.__str__() for ci in cat_imputers],
    "num_imputers": [ni.__str__() for ni in num_imputers],
    "models": [m.__class__.__name__ for m in models],
    "scorings": [s.__name__ for s in scorings],
}

test = True
update_previous_experiment = True

if __name__ == "__main__":

    experiment_name = "benchmark_motivation"
    result_folder = os.path.join(u.RESULT_FOLDER, experiment_name)
    try:
        os.mkdir(u.RESULT_FOLDER)
    except FileExistsError:
        pass

    same_name_exps = glob(os.path.join(u.RESULT_FOLDER, f"{experiment_name}*"), recursive=False)
    experiment_name = f"{experiment_name}_{len(same_name_exps)}"
    result_folder = os.path.join(u.RESULT_FOLDER, experiment_name)

    if not update_previous_experiment:
        try:
            os.mkdir(result_folder)
            os.mkdir(os.path.join(result_folder, "logs"))
        except FileExistsError:
            same_name_exps = glob(os.path.join(u.RESULT_FOLDER, f"{experiment_name}*"), recursive=False)
            experiment_name = f"{experiment_name}_{len(same_name_exps)}"
            result_folder = os.path.join(u.RESULT_FOLDER, experiment_name)
            os.mkdir(result_folder)
            os.mkdir(result_folder + '/logs')
    else:
        print(f"Updating {experiment_name}")

    # -- Load dataset
    datasets = []
    for did in u.DATASETS.values():
        try:
            dataset = get_dataset(did)
        except:
            gbl_log["datasets"].remove(did)
            gbl_log["failed_datasets"].append(did)
        else:
            datasets.append(dataset)

    # -- Save experimental design
    with open(os.path.join(result_folder, "experimental_description.json"), "w") as fw:
        json.dump(gbl_log, fw)

    # -- Runtime information
    number_iterations = 1
    for l in [datasets, encoders, scalers, cat_imputers, num_imputers, models, scorings]:
        number_iterations *= len(l)

    iteration_cost = 20
    days = number_iterations*iteration_cost / (3600*24)
    print(f"{number_iterations} iterations at {iteration_cost} seconds each = {round(days,1)} days")

    # -- Testing?
    if test:
        print('-'*20 + "Test activated" + '-'*20)
        for l in [datasets, encoders, scalers, cat_imputers, num_imputers, models, scorings]:
            np.random.seed(15)
            l[:] = [l[np.random.randint(len(l))]]

        models = [u.LGBMClassifier(random_state=random_state+3, n_estimators=3000, metric="None")]
        # encoders = [e.CVBlowUp(e.GLMMEncoder(), n_splits=5)]

    # -- Experiment
    nj = 1 if test else -1
    Parallel(n_jobs=nj, verbose=100)(
        delayed(main_loop)(result_folder, dataset, encoder, scaler, cat_imputer, num_imputer,
                            model, scoring, **kwargs)
        for (index, (dataset,
                      encoder,
                      scaler,
                      cat_imputer,
                      num_imputer,
                      model,
                      scoring))
        in enumerate(itertools.product(datasets,
                                        encoders,
                                        scalers,
                                        cat_imputers,
                                        num_imputers,
                                        models,
                                        scorings))
    )
