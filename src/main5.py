# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 14:41:36 2022

@author: federicom

ONLY CLASSIFICATION


"""

# -*- coding: utf-8 -*-
"""

blending (encode test set into a lot of encoded versions, predict each and thn blend prediction)
dependence on n_folds in CVRegularized
optimal folds?
try no tuning, see what happens
draw scheme of experiments
write Pargent
"""

import itertools
import json
import numpy as np
import openml
import os
import pandas as pd
import re
import time
import traceback

from catboost import CatBoostClassifier
from datetime import date
from glob import glob
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from openml.datasets import get_dataset
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier

import src.utils as u
import src.encoders as e

import warnings
warnings.filterwarnings("ignore")

np.random.seed(43)

# os.chdir('C:/Users/federicom/Documents/Github/EncoderComparison')

experiment_name = "benchmark_motivation"
result_folder = os.path.join(u.RESULT_FOLDER, experiment_name)
try:
    os.mkdir(u.RESULT_FOLDER)
except FileExistsError:
    pass
        
try:
    os.mkdir(result_folder)
    os.mkdir(result_folder + '/logs')
except FileExistsError:
    same_name_exps = glob(os.path.join(u.RESULT_FOLDER, f"{experiment_name}*"), recursive=False)
    experiment_name = f"{experiment_name}_{len(same_name_exps)}"
    result_folder = os.path.join(u.RESULT_FOLDER, experiment_name)
    os.mkdir(result_folder)
    os.mkdir(result_folder + '/logs')

# ---- Single iteration

def main_loop(did, encoder, scaler, model, scoring, resample_size=10000, n_resamples=5, n_splits=5):
    """
    from logfile: 
        0 : no computation
        1 : success
        2 : partial success
        -1 : failed 

    Parameters
    ----------
    did : TYPE
        DESCRIPTION.
    encoder : TYPE
        DESCRIPTION.
    scaler : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    scoring : TYPE
        DESCRIPTION.
    resample_size : TYPE, optional
        DESCRIPTION. The default is 10000.
    n_resamples : TYPE, optional
        DESCRIPTION. The default is 5.
    n_splits : TYPE, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    None.

    """
    
    
    if isinstance(encoder, str) and not isinstance(model, CatBoostClassifier):
        return
    
    # -- load dataset
    loaded = False
    t_end = time.time() + 10
    while time.time() < t_end:
        try:
            dataset = get_dataset(did)
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=dataset.default_target_attribute, dataset_format="dataframe"
            )
        except Exception as dataset_error:
            dataset_error_message = str(dataset_error)
        else:
            loaded = True
            break
    if not loaded:
        exp_name = "dataset_{}_{}_{}_{}_{}".format(
            did,
            encoder.__str__().split('(')[0],
            scaler.__class__.__name__,
            model.__class__.__name__,
            scoring.__name__
        )
        exec_log = {
            "dataset_id": did, 
            "encoder": encoder.__str__().split('(')[0],
            "scaler": scaler.__class__.__name__,
            "model": model.__class__.__name__,
            "scoring": scoring.__name__,
            "exit_status": 0, # see doc
            "error_message": dataset_error_message
        }
        log_name = f'{exp_name}_{exec_log["exit_status"]}.json'
        try:
            
            with open(os.path.join(result_folder, "logs", log_name), "w") as fw:
                json.dump(exec_log, fw)
        except Exception as log_error:
            exec_log["log_error_message"] = log_error
            with open(os.path.join(os.getcwd(), "logs", log_name), "w") as fw:
                json.dump(exec_log, fw)
    
    # -- define output - dataset saving and log
    saveset = pd.DataFrame()
    exp_name = "{}_{}_{}_{}_{}".format(
        dataset.name,
        encoder.__str__().split('(')[0],
        scaler.__class__.__name__,
        model.__class__.__name__,
        scoring.__name__
    )
    exec_log = {
        "dataset": dataset.name, 
        "encoder": encoder.__str__().split('(')[0],
        "scaler": scaler.__class__.__name__,
        "model": model.__class__.__name__,
        "scoring": scoring.__name__,
        "exit_status": 0, # see doc
        "error_message": ""
    }

    saveset_name = exp_name + ".csv"
    # log_name = exp_name + ".json"
    
    # -- drop nans, downsample to 10k, make y numerical
    X = X.dropna(axis=0, how="any")

    # -- define pipeline
    #!!! depends on openml syntax
    cats = X.dtypes[X.dtypes == 'category'].index.to_list()
    nums = X.dtypes[X.dtypes != 'category'].index.to_list()
    
    if hasattr(encoder, "fit"):
        CT = ColumnTransformer(
            [
                (
                    "encoder",
                    encoder,
                    cats
                ),
                (
                    "scaler",
                    scaler,
                    nums
                ),
            ],
            remainder="passthrough"
        )
    else:
        CT = ColumnTransformer(
            [
                (
                    "scaler",
                    scaler,
                    nums
                ),
            ],
            remainder="passthrough"
        )

    pipe = Pipeline([
        ("preproc", CT),
        ("model", model)
    ])

    search_space = u.get_pipe_search_space_one_encoder(model, encoder)
    cv = StratifiedKFold(n_splits=n_splits)

    # for each resample, run a cv (with tuning)
    for r in range(n_resamples):

        try:
            X = X.sample(resample_size, random_state=r)
        except ValueError:
            X = X.sample(resample_size, replace=True, random_state=r)
        y = pd.Series(e.LabelEncoder().fit_transform(
            y[X.index]), name="target")

        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        out = {
            "dataset": dataset.name,
            "resample": r,
            "encoder": str(encoder),
            "scaler": scaler,
            "model": model,
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

                # - tune
                tuning_result, BS = u.tune_pipe(
                    pipe,
                    Xtr, ytr,
                    search_space,
                    make_scorer(scoring),
                    n_jobs=1, n_splits=5, n_iter=10
                )

                # - save
                out["cv_scores"].append(scoring(yte, BS.predict(Xte)))
                out["tuning_scores"].append(tuning_result["best_score"])
                out["tuning_time"].append(tuning_result["time"])
                for hpar in search_space.keys():
                    out[hpar].append(tuning_result["best_params"][hpar])
        except Exception as tuning_error:
            try:
                exec_log["error_message"] = tuning_error
                saveset.to_csv(os.path.join(result_folder, saveset_name))
            except Exception as dumping_error:
                exec_log["dumping_error_message"] = dumping_error
                saveset.to_csv(os.path.join(os.getcwd(), saveset_name))
            else:
                exec_log["exit_status"] = 2
            
        saveset = pd.concat([saveset, pd.DataFrame(out)], ignore_index=True)

    saveset.to_csv(os.path.join(result_folder, saveset_name))
    
    # if no Exception was raised -> success
    if exec_log["exit_status"] == 0:
        exec_log["exit_status"] = 1

    log_name = f'{exp_name}_{exec_log["exit_status"]}.json'    
    try:
        with open(os.path.join(result_folder, "logs", log_name), "w") as fw:
            json.dump(exec_log, fw)
    except Exception as log_error:
        exec_log["log_error_message"] = log_error
        with open(os.path.join(os.getcwd(), "logs", log_name), "w") as fw:
            json.dump(exec_log, fw)
    
    
    return
    
    
# ---- Execution


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
    for ns in range(2, 10)
] + [
    e.CVRegularized(e.TargetEncoder(default=np.nan), n_splits=ns)
    for ns in range(2, 10)
] + [
    e.CVBlowUp(e.GLMMEncoder(), n_splits=ns)
    for ns in range(2, 10)
] + [
    e.CVBlowUp(e.TargetEncoder(), n_splits=ns)
    for ns in range(2, 10)
] + [
    e.Discretized(e.TargetEncoder(), how="minmaxbins", n_bins=nb)
    for nb in range(2, 23, 4)    
]

scalers = [
    u.RobustScaler(),
    e.CollapseEncoder()
]

models = [
    # CatBoostClassifier(verbose=0, thread_count=1)
    DecisionTreeClassifier(),
    LGBMClassifier()
]

scorings = [
    # u.balanced_accuracy_score,
    u.accuracy_score,
    u.roc_auc_score,
    u.f1_score
]

good = {
    # 'credit-g': 31,
    'nursery': 959,     
    'adult': 1590, 
    'mv': 881,
    'kdd_internet_usage': 981,
    'KDDCup09_appetency': 1111,
    'KDDCup09_churn': 1112,
    'KDDCup09_upselling': 1114, 
    'airlines': 1169, 
    'Agrawal1': 1235, 
    'bank-marketing': 1461, 
    'nomao': 1486, 
    # 'altri': 'non ancora inseriti'
}

kwargs = {
    "resample_size" : 10000,
    "n_resamples" : 10,
    "n_splits" : 5
}
gbl_log = {
    "datetime": date.today().__str__(), 
    "arguments": kwargs,    
    "datasets": list(good.keys()), 
    "encoders": [enc.__str__().split('(')[0] for enc in encoders], 
    "scalers": [sc.__class__.__name__ for sc in scalers], 
    "models": [m.__class__.__name__ for m in models], 
    "scorings": [s.__name__ for s in scorings], 
}

datasets_id = list(good.values())
if __name__ == "__main__":
    
    with open(os.path.join(result_folder, "experimental_description.json"), "w") as fw:
        json.dump(gbl_log, fw)
        
    Parallel(n_jobs=-1, verbose=100)(
        delayed(main_loop)(did, encoder, scaler,
                            model, scoring, **kwargs)
        for (index, (did,
                      encoder,
                      scaler,
                      model,
                      scoring))
        in enumerate(itertools.product(datasets_id,
                                        encoders,
                                        scalers,
                                        models,
                                        scorings))
    )
