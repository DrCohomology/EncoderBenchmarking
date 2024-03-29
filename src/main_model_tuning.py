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
from numpy.random import default_rng
from openml.datasets import get_dataset
from openml.exceptions import OpenMLServerException
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from rpy2.robjects.packages import importr
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from stopit import ThreadingTimeout as Timeout

import src.utils as u
import src.encoders as e
import src.config as cfg


def main_loop(result_folder,
              dataset, encoder, scaler, cat_imputer, num_imputer, models=tuple(), scorings=tuple(), index=0, num_exp=0,
              n_splits=5, timeout=6000):
    """
    output into logfile:
        0 : no computation
        1 : success
        2 : error raised
    """

    # -- LGBM requires tuning and cannot be handled by this function
    for model in models:
        if isinstance(model, u.LGBMClassifier):
            raise ValueError(f"{str(model)} is an invalid model as it requires tuning.")

    # -- Import R libraries if required by the encoder
    c1 = isinstance(encoder, e.RGLMMEncoder)
    c2 = isinstance(encoder, e.CVRegularized) and isinstance(encoder.base_encoder, e.RGLMMEncoder)
    c3 = isinstance(encoder, e.CVBlowUp) and isinstance(encoder.base_encoder, e.RGLMMEncoder)
    if c1 or c2 or c3:
        importr("lme4")
        importr("base")
        importr("utils")

    # -- Load the dataset
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    # -- Drop empty columns and lines
    X = X.dropna(axis=0, how="all").dropna(axis=1, how="all").reset_index(drop=True)
    y = pd.Series(e.LabelEncoder().fit_transform(y[X.index]), name="target").reset_index(drop=True)

    # -- define preprocessing pipeline
    cats = X.select_dtypes(include=("category", "object")).columns
    nums = X.select_dtypes(exclude=("category", "object")).columns
    catpipe = Pipeline([("imputer", cat_imputer), ("encoder", encoder)])
    numpipe = Pipeline([("imputer", num_imputer), ("scaler", scaler)])
    prepipe = ColumnTransformer([("encoder", catpipe, cats), ("scaler", numpipe, nums)], remainder="passthrough")

    # -- define output - dataset saving and log
    saveset = pd.DataFrame()
    exec_log = {
        "exit_status": 0,  # 0 = no computation; 1 = success; 2 = fail
        "dataset": dataset.name,
        "encoder": u.get_acronym(encoder.__str__(), underscore=False),
        "error_message": ""
    }
    exp_name = "{}_{}".format(exec_log["dataset"], exec_log["encoder"])

    # -- Feedback
    print(
        f"{index+1:5}/{num_exp} {datetime.datetime.now().strftime('%d.%m %H:%M')} "
        f"{exec_log['dataset']:50}{exec_log['encoder']:15}"
    )

    # -- Computation
    cv = StratifiedKFold(n_splits=n_splits, random_state=None)
    try:
        with Timeout(timeout, swallow_exc=False) as timeout_ctx:
            for icv, (tr, te) in enumerate(cv.split(X, y)):
                Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
                Xtr, ytr = Xtr.reset_index(drop=True), ytr.reset_index(drop=True)
                Xte, yte = Xte.reset_index(drop=True), yte.reset_index(drop=True)

                start = time.time()
                XEtr = prepipe.fit_transform(Xtr, ytr)
                end = time.time()

                for model in models:
                    for scoring in scorings:
                        if isinstance(model, u.LGBMClassifier):
                            raise ValueError("Cannot tune LGBM.")

                        GS = u.tune_model(model, XEtr, ytr, scoring)

                        model = GS.best_estimator_

                        if scoring.__name__ == "roc_auc_score":
                            cv_score = scoring(yte, model.predict_proba(prepipe.transform(Xte))[:, 1])
                        else:
                            cv_score = scoring(yte, model.predict(prepipe.transform(Xte)))

                        out = {
                            # "index": i,
                            "dataset": dataset.name,
                            "fold": icv,
                            "encoder": str(encoder),
                            "scaler": scaler.__class__.__name__,
                            "model": model.__class__.__name__,
                            "scoring": scoring.__name__,
                            "cv_score": cv_score,
                            "tuning_score": GS.best_score_,
                            "time": end - start
                        }
                        out.update(GS.best_params_)
                        saveset = pd.concat([saveset, pd.DataFrame(out, index=[0])], ignore_index=True)
        saveset = saveset.sort_values(["encoder", "scaler", "model", "scoring"])
    except Exception as tuning_error:
        exec_log["exit_status"] = 2
        exec_log["error_message"] = str(tuning_error)

    # if no Exception was raised -> success
    if exec_log["exit_status"] == 0:
        saveset_name = exp_name + ".csv"
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


if __name__ == "__main__":

    # -- Configure
    rpy2_logger.setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MKL_NUM_THREADS='1')
    np.random.seed(0)
    gbl_log = {
        "datetime": date.today().__str__(),
        "arguments": cfg.MAIN_PARAMETERS,
        "datasets": cfg.DATASET_IDS["model tuning"],
        "failed_datasets": list(),
        "encoders": [enc.__str__().split('(')[0] for enc in cfg.ENCODERS],
        "models": [m.__class__.__name__ for m in cfg.MODELS["model tuning"]],
        "scorings": [s.__name__ for s in cfg.SCORINGS],
    }

    # -- Create directories
    experiment_name = "model tuning"
    result_dir = u.RESULTS_DIR / experiment_name
    try:
        os.mkdir(result_dir)
        os.mkdir(result_dir / "logs")
    except FileExistsError:
        pass

    # -- Experiments: each experiment is identified by: (dataset, encoder, scaler, imputers)
    experiments = list(itertools.product(cfg.DATASET_NAMES["model tuning"], cfg.ENCODERS,
                                         cfg.SCALERS, cfg.IMPUTERS_CAT, cfg.IMPUTERS_NUM))
    experiments = u.remove_concluded_modeltuning(experiments, result_dir, model=None)
    experiments = u.remove_failed_modeltuning(experiments, result_dir)
    experiments = u.smart_sort(experiments, random=True)

    # -- Load datasets
    datasets = {}
    for dname, did in zip(cfg.DATASET_NAMES["no tuning"], cfg.DATASET_IDS["no tuning"]):
        try:
            datasets[dname] = get_dataset(did)
        except OpenMLServerException:
            gbl_log["datasets"].remove(did)
            gbl_log["failed_datasets"].append(did)

    experiments = [
        (datasets[dname], encoder, scaler, cat_imputer, num_imputer)
        for (dname, encoder, scaler, cat_imputer, num_imputer) in experiments
    ]

    Parallel(n_jobs=cfg.NUM_PROCESSES, verbose=0)(
        delayed(main_loop)(result_dir, dataset, encoder, scaler, cat_imputer, num_imputer,
                           models=cfg.MODELS["model tuning"], scorings=cfg.SCORINGS,
                           index=index, num_exp=len(experiments), **cfg.MAIN_PARAMETERS)
        for (index, (dataset, encoder, scaler, cat_imputer, num_imputer)) in enumerate(experiments)
    )
