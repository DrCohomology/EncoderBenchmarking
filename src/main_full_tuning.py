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
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier, early_stopping
from numpy.random import default_rng
from openml.datasets import get_dataset
from openml.exceptions import OpenMLServerException
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
import src.config as cfg


def main_loop(result_dir,
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
    exec_log = {
        "exit_status": 0,  # 0 = no computation; 1 = success; 2 = fail
        "dataset": dataset.name,
        "encoder": u.get_acronym(encoder.__str__(), underscore=False),
        "model": u.get_acronym(model.__class__.__name__, underscore=False),
        "scoring": scoring.__name__,
        "error_message": ""
    }
    exp_name = "{}_{}_{}_{}".format(exec_log["dataset"], exec_log["encoder"], exec_log["model"], exec_log["scoring"])

    # -- define pipeline
    cats = X.select_dtypes(include=("category", "object")).columns
    nums = X.select_dtypes(exclude=("category", "object")).columns
    catpipe = Pipeline([("imputer", cat_imputer), ("encoder", encoder)])
    numpipe = Pipeline([("imputer", num_imputer), ("scaler", scaler)])
    prepipe = ColumnTransformer([("encoder", catpipe, cats), ("scaler", numpipe, nums)], remainder="passthrough")
    pipe = Pipeline([("preproc", prepipe), ("model", model)])

    # when using LGBM, search_space is used just to initialize out
    search_space = u.get_pipe_search_space_one_encoder(model, encoder)

    out = {
        "dataset": dataset.name,
        "encoder": str(encoder),
        "scaler": scaler.__class__.__name__,
        "model": model.__class__.__name__,
        "scoring": scoring.__name__,
        "cv_score": [],
        "tuning_score": [],
        "tuning_time": [],
    }
    out.update({
        hpar: [] for hpar in search_space.keys()
    })

    # -- Feedback
    print(
        f"{index+1:5}/{num_exp} {datetime.datetime.now().strftime('%d.%m %H:%M')} "
        f"{exec_log['dataset']:50}{exec_log['encoder']:15}{exec_log['model']:10}{exec_log['scoring']}"
    )

    # -- Computation
    cv = StratifiedKFold(n_splits=n_splits, random_state=None)
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

                    BS = pipe
                    tuning_result = {
                        "best_score": scoring(ytrval, pipe.predict(Xtrval)),
                        "time": time.time() - start_time,
                        "best_params": {
                            "model__n_estimators": pipe["model"].best_iteration_,
                        }
                    }
                # - tuning
                else:
                    if scoring.__name__ == "roc_auc_score":
                        scorer = make_scorer(scoring, needs_proba=True)
                    else:
                        scorer = make_scorer(scoring)

                    tuning_result, BS = u.tune_pipe(
                        pipe,
                        Xtr, ytr,
                        search_space,
                        scorer,
                        n_jobs=1, n_splits=3, max_iter=5
                    )

                # - save
                if scoring.__name__ == "roc_auc_score":
                    cv_score = scoring(yte, BS.predict_proba(Xte)[:, 1])
                else:
                    cv_score = scoring(yte, BS.predict(Xte))

                out["cv_score"].append(cv_score)
                out["tuning_score"].append(tuning_result["best_score"])
                out["tuning_time"].append(tuning_result["time"])
                for hpar in search_space.keys():
                    out[hpar].append(tuning_result["best_params"][hpar])

            saveset = pd.concat([saveset, pd.DataFrame(out)], ignore_index=True)

    except Exception as tuning_error:
        exec_log["exit_status"] = 2
        exec_log["error_message"] = str(tuning_error)

    # if no Exception was raised -> success
    if exec_log["exit_status"] == 0:
        saveset_name = exp_name + ".csv"
        saveset.to_csv(os.path.join(result_dir, saveset_name))
        exec_log["exit_status"] = 1

    # dump exec_log
    log_name = f'{exec_log["exit_status"]}_{exp_name}.json'
    try:
        with open(os.path.join(result_dir, "logs", log_name), "w") as fw:
            json.dump(exec_log, fw)
    except Exception as log_error:
        exec_log["log_error_message"] = str(log_error)
        with open(os.path.join(os.getcwd(), "logs", log_name), "w") as fw:
            json.dump(exec_log, fw)
    return

# ---- Execution
gbl_log = {
    "datetime": date.today().__str__(),
    "arguments": cfg.MAIN_PARAMETERS,
    "datasets": cfg.DATASET_IDS["full tuning"],
    "failed_datasets": list(),
    "encoders": [enc.__str__().split('(')[0] for enc in cfg.ENCODERS],
    "models": [m.__class__.__name__ for m in cfg.MODELS["full tuning"]],
    "scorings": [s.__name__ for s in cfg.SCORINGS],
}

if __name__ == "__main__":

    # -- Configure
    rpy2_logger.setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MKL_NUM_THREADS='1')
    np.random.seed(0)
    gbl_log = {
        "datetime": date.today().__str__(),
        "arguments": cfg.MAIN_PARAMETERS,
        "datasets": cfg.DATASET_IDS["full tuning"],
        "failed_datasets": list(),
        "encoders": [enc.__str__().split('(')[0] for enc in cfg.ENCODERS],
        "models": [m.__class__.__name__ for m in cfg.MODELS["full tuning"]],
        "scorings": [s.__name__ for s in cfg.SCORINGS],
    }

    # -- Create directories
    experiment_name = "full tuning"
    result_dir = u.RESULTS_DIR / experiment_name
    try:
        os.mkdir(result_dir)
        os.mkdir(result_dir / "logs")
    except FileExistsError:
        pass

    # -- Experiments: each experiment is identified by: (dataset, encoder, scaler, imputers, model, scoring)
    experiments = itertools.product(cfg.DATASET_NAMES["full tuning"], cfg.ENCODERS,
                                    cfg.SCALERS, cfg.IMPUTERS_CAT, cfg.IMPUTERS_NUM,
                                    cfg.MODELS["full tuning"], cfg.SCORINGS)
    # experiments = u.remove_concluded_runs(experiments, result_dir, repeat_unsuccessful=False) !!! OUTDATED
    experiments = u.smart_sort(experiments, random=True)

    # -- Load datasets
    datasets = {}
    for dname, did in zip(cfg.DATASET_NAMES["full tuning"], cfg.DATASET_IDS["full tuning"]):
        try:
            datasets[dname] = get_dataset(did)
        except OpenMLServerException:
            gbl_log["datasets"].remove(did)
            gbl_log["failed_datasets"].append(did)

    experiments = [
        (datasets[dname], encoder, scaler, cat_imputer, num_imputer, model, scoring)
        for (dname, encoder, scaler, cat_imputer, num_imputer, model, scoring) in experiments
    ]

    Parallel(n_jobs=cfg.NUM_PROCESSES, verbose=0)(
        delayed(main_loop)(result_dir, dataset, encoder, scaler, cat_imputer, num_imputer, model, scoring,
                           index=index, num_exp=len(experiments), **cfg.MAIN_PARAMETERS)
        for (index, (dataset, encoder, scaler, cat_imputer, num_imputer, model, scoring)) in
        enumerate(experiments)
    )


