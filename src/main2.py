# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 10:29:44 2022

@author: federicom

Implements Pipelines

"""

"""
Experiments should be expanded by considering as test set a collection of sets
The results will be averaged on these
Would it make sene to use cv? we could just repeat the experiment ad libitum with different sampling

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
from sklearn.preprocessing import RobustScaler

import warnings
warnings.filterwarnings("ignore")

# os.chdir('C:/Users/federicom/Documents/Github/EncoderComparison')

experiment_name = "DT_10k10k-100k"
result_folder = u.RESULT_FOLDER + '/' + experiment_name

os.mkdir(result_folder)
os.mkdir(result_folder + '/logs')

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
    (enc.TargetEncoder, BinaryEncoder), 
    (enc.CVTargetEncoder2, BinaryEncoder),
    (enc.CVTargetEncoder3, BinaryEncoder),
    (enc.CVTargetEncoder5, BinaryEncoder),
    # (enc.CVTargetEncoder10, BinaryEncoder),
    # (enc.CVTargetEncoder20, BinaryEncoder),
    (CatBoostEncoder, ), 
    (CountEncoder, ),
    (enc.TargetEncoder, enc.CVTargetEncoder5), 
    # (enc.CVTargetEncoder5, enc.CVTargetEncoder10, enc.CVTargetEncoder20, BinaryEncoder),
]

n = 110000
train_size = 10000
test_sizes = [10000*x for x in range(1, 11)]
models = [u.DecisionTreeClassifier]
metrics = [u.balanced_accuracy_score]

def main_loop(dataset, encoders, index, resample_states):
    # returns 1 if success, 0 if continuous (no operations), -1 if failed
    start_time = time.time()

    dataset_name = re.split(r'[(\\)(/)]', dataset)[-1].split('.csv')[0]
    encoders_name = ', '.join(encoder.__name__ for encoder in encoders)
    outs = []

    for resample_num, random_state in enumerate(resample_states):
        
        out = {
            "dataset": dataset_name,
            "encoder": encoders_name,
            "exit_status": 0,
            "exception": "",
            "runtime": 0,
        }
        
        saveset = pd.DataFrame()
        
        df = pd.read_csv(dataset)
        df = df.sample(n=min(n, len(df)), random_state=random_state, ignore_index=True)
    
        X, y = u.pre2process(df)
        
        # ----  Noise columns
        # num_noise = 3
        # X[[f"noise_{i}" for i in range(num_noise)]] = np.random.random((len(X), num_noise))
        
        # ---- Cat only
        # X = X[[col for col in X if "cat" in col]]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=1444, stratify=y
        )
        
        if len(y.unique()) > 2:
            out['exception'] = f'target has {len(y.unique())} values'
            return out
        
        try:
            # Each dataset is saved differently to ensure independence
    
            CEW = enc.CatMEW(encoders, scaler=RobustScaler)
    
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
                    
                    old_size = 0
                    stop = False
                    for it, test_size in enumerate(test_sizes):
                        # As they have already been shuffled rom train_test_split, 
                        # we can just take the n, n+1 entries
                        # We break if the test size is bigger than wanted
                        
                        test_size, stop = min(test_size, len(X_test)), test_size > len(X_test)
                            
                        X_test_temp = X_test.iloc[old_size : test_size]
                        y_test_temp = y_test.iloc[old_size : test_size]
                        y_pred_temp = best.predict(X_test_temp)
                        test_score = scoring(y_test_temp, y_pred_temp)
    
                        new_exp = pd.DataFrame({
                            "dataset": dataset_name + f'_{it}', 
                            "encoder": encoders_name,
                            "model": model.__name__,
                            "scoring": scoring.__name__,
                            "test_size": test_size-old_size,
                            "best_params": best_params,
                            "best_cv_score": best_score,
                            "tuning_time": runtime,
                            "test_score": test_score,
                            # "tree_depth": best.named_steps['model'].tree_.max_depth
                        }, index=[0])
    
                        saveset = pd.concat([saveset, new_exp], ignore_index=True)
                        
                        old_size = test_size
                        
                        if stop:
                            break
    
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
            saveset.to_csv(result_folder + f"/{dataset_name}_{encoders_name}_{resample_num}.csv")
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

resample_states = [1, 10, 100, 1444, 43, 76, 234, 897, 8765, 121, 97, 34]

x = Parallel(n_jobs=-1, verbose=100)(
    delayed(main_loop)(dataset, encs, index, resample_states)
    for (index, (dataset, encs)) in enumerate(itertools.product(datasets, encoders_list))
)

num_logs = len(glob(result_folder + "/log*.json"))

try:
    with open(result_folder + f'/log{num_logs}.json', 'w+') as fw:
        json.dump(x, fw)    
except:
    print('I could not dump where you asked me to ')
    with open('LOG_________________________________.json', 'w+') as fw:
        json.dump(x, fw)
