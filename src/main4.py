# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 10:29:44 2022

@author: federicom

Remove CatMEW, optimize encoder hpars. 
Considers only the first encoder in every element of encoders_list, for semplicity
"""


import itertools
import json
import numpy as np
import os
import pandas as pd
import re
import src.utils as u
import src.encoders as enc
import time
import traceback

from glob import glob
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

np.random.seed(43)

# os.chdir('C:/Users/federicom/Documents/Github/EncoderComparison')

experiment_name = "new_exps"
result_folder = u.RESULT_FOLDER + '/' + experiment_name
try:
    os.mkdir(result_folder)
    os.mkdir(result_folder + '/logs')
except FileExistsError as e:
    print(e)


# ----
# After discrimination
encoders_list = [
    (enc.CollapseEncoder, ),
    (enc.TargetEncoder, ),
    (enc.BinaryEncoder, ),
    (enc.CatBoostEncoder, ),
    (enc.CountEncoder, ),
    (enc.CVTargetEncoder, ),
    (enc.SmoothedTE, ),
    (enc.LeaveOneOutEncoder, ),
    (enc.OneHotEncoder, ),
]
scaler_list = [
    u.RobustScaler,
    enc.CollapseEncoder
]

train_size = 10000
test_size = 10000

models = [u.DecisionTreeClassifier, u.LGBMClassifier, u.LogisticRegression]
metrics = [
    u.balanced_accuracy_score,
    u.cohen_kappa,
    u.accuracy_score
]

# datasets = glob(u.DATASET_FOLDER + "/*.csv")
datasets = [
    u.DATASET_FOLDER + '/' + x
    for x in (
        "adult.csv",
        "credit.csv",
        "kaggle_cat_dat_1.csv",
        "kaggle_cat_dat_2.csv",
        "kick.csv",
        "promotion.csv",
        "telecom.csv"
    )
]
resample_states = np.random.randint(1, 1000, 100)

def main_loop(dataset, encoders, scaler, model, scoring, resample_states):
    # returns 1 if success, 0 if continuous (no operations), -1 if failed
    start_time = time.time()

    dataset_name = re.split(r'[(\\)(/)]', dataset)[-1].split('.csv')[0]
    encoders_name = ', '.join(encoder.__name__ for encoder in encoders)
    scaler_name = scaler + \
        'Scaler' if isinstance(scaler, str) else scaler.__name__
    outs = []

    df = pd.read_csv(dataset)
    X, y = u.pre2process(df)

    # X_test will be resampled again
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=1444, stratify=y
    )

    out = {
        "dataset": dataset_name,
        "encoder": encoders_name,
        "exit_status": 0,
        "exception": "",
        "runtime": 0,
    }

    try:
        encoder = encoders[0]()
        CT = ColumnTransformer(
            [
                (
                    "encoder",
                    encoder,
                    [col for col in X.columns if "cat" in col]
                ),
                (
                    "scaler",
                    scaler(),
                    [col for col in X.columns if "num" in col]
                ),
            ],
            remainder="passthrough"
        )

        pipe = Pipeline([
            ("preproc", CT),
            ("model", model())
        ])

        # Case specific
        if "Logistic" in repr(model):
            pipe = Pipeline([
                ("preproc", CT),
                ("standardize", u.RobustScaler()),
                ("model", model())
            ]).set_params(**{
                "model__max_iter" : 1000    
            })
            
        if "Count" in repr(encoders[0]):
            pipe = pipe.set_params(**{
                "preproc__encoder__min_group_size": 0.05
            })

        search_space = u.get_pipe_search_space(model, encoders)

        tuning_result, BS = u.tune_pipe(
            pipe,
            X_train, y_train,
            search_space,
            make_scorer(scoring),
            n_jobs=-1, n_splits=7, n_iter=20
        )

        best_params = tuning_result["best_params"]
        best_score = tuning_result["best_score"]
        runtime = tuning_result["time"]

        best = pipe.set_params(**best_params)
        best.fit(X_train, y_train)

        saveset = pd.DataFrame()

        for resample_num, random_state in enumerate(resample_states):

            X_test_res = X_test.sample(
                test_size, random_state=random_state)
            y_test_res = y_test.sample(
                test_size, random_state=random_state)

            y_pred_res = best.predict(X_test_res)
            test_score = scoring(y_test_res, y_pred_res)

            new_exp = pd.DataFrame({
                "dataset": dataset_name,
                "resample_num": resample_num,
                "encoder": encoders_name,
                "scaler": scaler_name,
                "model": model.__name__,
                "scoring": scoring.__name__,
                "train_size": train_size,
                "test_size": test_size,
                "best_params": best_params,
                "best_cv_score": best_score,
                "tuning_time": runtime,
                "test_score": test_score,
                # "feature_importances": f_imp,
                # "tree_depth": best.named_steps['model'].tree_.max_depth
            }, index=[0])

            saveset = pd.concat([saveset, new_exp], ignore_index=True)

    except Exception as e:
        print("-"*20, e)
        out.update({"exit_status": -1, "exception": str(e),
                   "traceback": traceback.format_exc()})
        try:
            saveset.to_csv(
                f"{result_folder}/{dataset_name}_{encoders_name}_{scaler_name}_{model.__name__}_{scoring.__name__}_{resample_num}.csv")
        except NameError:
            pass
        else:
            out["exit_status"] = -0.5
    else:
        saveset.to_csv(
            f"{result_folder}/{dataset_name}_{encoders_name}_{scaler_name}_{model.__name__}_{scoring.__name__}_{resample_num}.csv")
        out["exit_status"] = 1
    finally:
        
        out["runtime"] = time.time() - start_time
        try:
            with open(f"{result_folder}/logs/log_{out['exit_status']}_{dataset_name}_{encoders_name}_{scaler_name}_{model.__name__}_{scoring.__name__}_{resample_num}.json", 'w+') as fw:
                json.dump(out, fw)
        except:
            print('I could not dump where you asked me to ')
            with open(f'_____LOG_{dataset_name}_{encoders_name}.json', 'w+') as fw:
                json.dump(out, fw)
            outs.append(out)
    return outs


# ---- Execution


x = Parallel(n_jobs=-1, verbose=100)(
    delayed(main_loop)(dataset, encoders, scaler, model, scoring, resample_states)
    for (index, (dataset, 
                 encoders, 
                 scaler, 
                 model, 
                 scoring)) 
    in enumerate(itertools.product(datasets, 
                                   encoders_list, 
                                   scaler_list, 
                                   models, 
                                   metrics))
)
# for (index, (dataset, encoders, scaler)) in enumerate(itertools.product(datasets, encoders_list, scaler_list)):
#     x = main_loop(dataset, scaler, encoders, resample_states)


num_logs = len(glob(result_folder + "/log*.json"))

try:
    with open(result_folder + f'/log{num_logs}.json', 'w+') as fw:
        json.dump(x, fw)
except:
    print('I could not dump where you asked me to ')
    with open('LOG_________________________________.json', 'w+') as fw:
        json.dump(x, fw)
