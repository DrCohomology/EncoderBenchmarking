# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 10:29:44 2022

@author: federicom

The big difference is that the training set is fixed for every resample
"""


import itertools
import json
import numpy as np
import os
import pandas as pd
import re
import utils as u
import encoders as enc
import time
import traceback

from glob import glob
from joblib import Parallel, delayed
from category_encoders import (
    BackwardDifferenceEncoder,
    BinaryEncoder,
    CatBoostEncoder,
    CountEncoder,
    GLMMEncoder,
    HashingEncoder,
    HelmertEncoder,
    JamesSteinEncoder,
    LeaveOneOutEncoder,
    MEstimateEncoder,
    OneHotEncoder,
    PolynomialEncoder,
    SumEncoder,
    WOEEncoder,
)
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

np.random.seed(43)

# os.chdir('C:/Users/federicom/Documents/Github/EncoderComparison')

experiment_name = "DT_10k10k_multiscoring"
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
    (enc.CVTargetEncoder2, ),
    (enc.CVTargetEncoder3, ),
    (enc.CVTargetEncoder5, ),
    # (enc.CVTargetEncoder10, ),
    # (enc.CVTargetEncoder20, ),
    (BinaryEncoder, ), 
    # (enc.TargetEncoder, BinaryEncoder), 
    # (enc.CVTargetEncoder2, BinaryEncoder),
    # (enc.CVTargetEncoder3, BinaryEncoder),
    # (enc.CVTargetEncoder5, BinaryEncoder),
    # (enc.CVTargetEncoder10, BinaryEncoder),
    # (enc.CVTargetEncoder20, BinaryEncoder),
    (CatBoostEncoder, ), 
    (CountEncoder, ),
    # (enc.TargetEncoder, enc.CVTargetEncoder5), 
    # (enc.CVTargetEncoder5, enc.CVTargetEncoder10, enc.CVTargetEncoder20, BinaryEncoder),
]
scaler_list = [
    u.RobustScaler,
    'Collapse'    
]

train_size = 10000
test_size = 10000

models = [u.DecisionTreeClassifier]
metrics = [u.balanced_accuracy_score, u.cohen_kappa, u.accuracy_score]

def main_loop(dataset, scaler, encoders, resample_states):
    # returns 1 if success, 0 if continuous (no operations), -1 if failed
    start_time = time.time()

    dataset_name = re.split(r'[(\\)(/)]', dataset)[-1].split('.csv')[0]
    encoders_name = ', '.join(encoder.__name__ for encoder in encoders)
    scaler_name = scaler+'Scaler' if isinstance(scaler, str) else scaler.__name__
    outs = []

    df = pd.read_csv(dataset)
    X, y = u.pre2process(df)
    
    # X_test will be resampled again
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=1444, stratify=y
    )
    
    try:
        CEW = enc.CatMEW(encoders, scaler)

        for model in models:
            search_space = u.get_pipe_search_space(model)
            for scoring in metrics:

                pipe = Pipeline([
                    ('encoder', CEW),
                    ('model', model())    
                ])                

                tuning_result, BS = u.tune_pipe(
                    pipe, X_train, y_train, search_space, make_scorer(scoring)
                )

                best_params = tuning_result["best_params"]
                best_score = tuning_result["best_score"]
                runtime = tuning_result["time"]

                best = pipe.set_params(**best_params)
                best.fit(X_train, y_train)
                
                # A bit clunky, but it works
                # feature_names = pipe['encoder'].transform(X_train).columns
                # feature_importances = pipe['model'].feature_importances_
                # f_imp = dict(zip(feature_names, feature_importances))
                
                saveset = pd.DataFrame()    

                for resample_num, random_state in enumerate(resample_states):
                    
                    X_test_res = X_test.sample(test_size, random_state=random_state)
                    y_test_res = y_test.sample(test_size, random_state=random_state)
                    
                    out = {
                        "dataset": dataset_name,
                        "encoder": encoders_name,
                        "exit_status": 0,
                        "exception": "",
                        "runtime": 0,
                    }
                
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
        # raise e
        out.update({"exit_status": -1, "exception": str(e), "traceback": traceback.format_exc()})
        try: 
            saveset.to_csv(result_folder + f"/{dataset_name}_{encoders_name}_{resample_num}.csv")
        except NameError:
            pass
        else:
            out["exit_status"] = -0.5
    else:
        saveset.to_csv(result_folder + f"/{dataset_name}_{encoders_name}_{scaler_name}_{resample_num}.csv")
        out["exit_status"] = 1
    finally:
        out["runtime"] = time.time() - start_time
        try:
            with open(result_folder+f"/logs/log_{out['exit_status']}_{dataset_name}_{encoders_name}_{resample_num}.json", 'w+') as fw:
                json.dump(out, fw)    
        except:
            print('I could not dump where you asked me to ')
            with open(f'_____LOG_{dataset_name}_{encoders_name}.json', 'w+') as fw:
                json.dump(out, fw)
            outs.append(out)
    return outs


# ---- Execution

datasets = glob(u.DATASET_FOLDER + "/*.csv")

resample_states = np.random.randint(1, 1000, 100)

x = Parallel(n_jobs=-1, verbose=100)(
    delayed(main_loop)(dataset, scaler, encoders, resample_states)
    for (index, (dataset, encoders, scaler)) in enumerate(itertools.product(datasets, encoders_list, scaler_list))
)

num_logs = len(glob(result_folder + "/log*.json"))

try:
    with open(result_folder + f'/log{num_logs}.json', 'w+') as fw:
        json.dump(x, fw)    
except:
    print('I could not dump where you asked me to ')
    with open('LOG_________________________________.json', 'w+') as fw:
        json.dump(x, fw)
