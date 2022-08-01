# -*- coding: utf-8 -*-
"""
Created on Wed May 25 09:28:53 2022

@author: federicom
"""

import numpy as np
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler

from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_validate
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

import src.encoders as enc
import src.utils as u

from scipy.stats import chi2_contingency

dataset = "credit"
df = pd.read_csv(f"C:/Data/{dataset}.csv")
dt = DecisionTreeClassifier(class_weight='balanced')

RUS = RandomUnderSampler()
X, y = u.pre2process(df)

# Balanced
Xb, yb = RUS.fit_sample(X, y)

encoders = [
    enc.BackwardDifferenceEncoder,
    enc.BinaryEncoder,
    enc.CatBoostEncoder,
    # enc.CountEncoder, # fails
    # enc.GLMMEncoder, # too slow
    # enc.HashingEncoder, # fails for other reasons
    enc.HelmertEncoder,
    enc.JamesSteinEncoder,
    enc.LeaveOneOutEncoder,
    enc.MEstimateEncoder,
    enc.OneHotEncoder,
    enc.PolynomialEncoder,
    enc.SumEncoder,
    enc.WOEEncoder,
    enc.SmoothedTE,
]

scores_list = [
    u.balanced_accuracy_score,
    u.cohen_kappa,
    u.accuracy_score,
]

scores = {
    s.__name__: make_scorer(s)
    for s in (scores_list)
}

results = {}
exceptions = []
for encoder in tqdm(encoders):
    # transform one part and rescale the other
    CT = ColumnTransformer([
        (
            "encoder",
            encoder(),
            [col for col in X.columns if "cat" in col]
        ),
        (
            "scaler",
            RobustScaler(),
            [col for col in X.columns if "num" in col]
        ),

    ])
    LR = u.LogisticRegression(max_iter=1000, solver="liblinear")

    PP = Pipeline([
        ("preproc", CT),
        ("model", LR)
    ])

    try:
        out = cross_validate(PP, X, y,
                             scoring=scores,
                             n_jobs=-1,
                             verbose=0,
                             )
        outb = cross_validate(PP, Xb, yb,
                              scoring=scores,
                              n_jobs=-1,
                              verbose=0,
                              )
        outb = {
            f"{k}_bal": v for k, v in outb.items()
        }
        out.update(outb)

    except Exception as e:
        out = None
        outb = None
        exceptions.append(e)
    finally:
        results[encoder.__name__] = out


# Reorganize results into a df
new = True
end = False
while not end:
    try:
        for k, v in results.items():
            if v is None:
                if new:
                    print("Failed encoders:")
                    new = False
                print(k)
                del results[k]
        end = True
    except:
        pass

metrics = results[list(results.keys())[0]]
means = {
    metric: {
        enc_name: np.mean(enc_res[metric])
        for enc_name, enc_res in results.items()
    }
    for metric in metrics
}
stds = {
    metric: {
        enc_name: np.std(enc_res[metric])
        for enc_name, enc_res in results.items()
    }
    for metric in metrics
}

ms = pd.DataFrame().from_dict(means)
ss = pd.DataFrame().from_dict(stds)
ress = ms.join(ss, rsuffix='_std')
ress.columns = [
    "fit_time", "score_time", "Bacc", "ck", "acc",
    "fit_time_bal", "score_time_bal", "Bacc_bal", "ck_bal", "acc_bal",
    "fit_time_std", "score_time_std", "Bacc_std", "ck_std", "acc_std",
    "fit_time_bal_std", "score_time_bal_std", "Bacc_bal_std", "ck_bal_std", "acc_bal_std",
]
# reorder
ress = ress[[
    "Bacc", "Bacc_std", "Bacc_bal", "Bacc_bal_std", 
    "ck", "ck_std", "ck_bal", "ck_std",
    "acc", "acc_std", "acc_bal", "acc_bal_std",
    "fit_time", "fit_time_bal"
]]
#%% save
ress.to_csv(u.RESULT_FOLDER + f"/{dataset}_LR_test.csv")
