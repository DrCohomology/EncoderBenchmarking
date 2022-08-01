# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:59:39 2022

@author: federicom
"""

import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import time

from collections import defaultdict, Counter
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.pipeline import make_pipeline
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from category_encoders import (
    BackwardDifferenceEncoder,
    CatBoostEncoder,
    OneHotEncoder,
    CountEncoder,
    BinaryEncoder,
)
from numba import njit
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
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

from warnings import filterwarnings
"""
DiscretizedTE: if one groups values with similar targets and then add a second
dimension to still discriminate them, overfitting will reduce but information
will be there if needed. 
RESEARCH QUESTIONS
- does scaling an attribute differently change the tree? 
    - no, only target matters
- can we discretize TE effectively?
    - 
- Better results?
    - Looks like yes! Discretized TE and minmaxnbins outperform Target
"""

# % Test with actual classes
dataset = "adult"
df = pd.read_csv(f"{u.DATASET_FOLDER}/{dataset}.csv")
df = df.sample(min(50000, len(df)), random_state=1444)
X, y = u.pre2process(df)

encoders = [
    enc.TargetEncoder(),
    enc.DiscretizedTargetEncoder(),
    enc.DiscretizedTargetEncoder(how="minmaxbins"),
    enc.DiscretizedTargetEncoder(how="adaptive"),
    enc.BinaryEncoder(),
    enc.CatBoostEncoder()
]

# DT = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10)
model = LGBMClassifier()

scores_list = [
    balanced_accuracy_score,
    u.cohen_kappa,
    accuracy_score,
]
scores = {
    s.__name__: make_scorer(s)
    for s in (scores_list)
}

# ---- Execution
ress = pd.DataFrame()
for encoder in encoders:

    CT = ColumnTransformer([
        (
            "encoder",
            encoder,
            [col for col in X.columns if "cat" in col]
        ),
        (
            "scaler",
            RobustScaler(),
            [col for col in X.columns if "num" in col]
        ),
    
    ])    

    PP = Pipeline([
        ("preproc", CT),
        ("model", model)
    ])

    # encoder.fit(X, y)

    # performance
    out = cross_validate(PP, X, y,
                          scoring=scores,
                          n_jobs=-2,
                          verbose=0,
                          cv=10
                          )
    bacc_ = out['test_balanced_accuracy_score']
    ck_ = out['test_cohen_kappa']
    acc_ = out['test_accuracy_score']
    tt_ = out['fit_time']

    bacc, bacc_std = bacc_.mean(), bacc_.std()
    ck, ck_std = ck_.mean(), ck_.std()
    acc, acc_std = acc_.mean(), acc_.std()
    tt, tt_std = tt_.mean(), tt_.std()

    print(f"""{str(encoder)}
    Balanced Accuracy: {bacc:.03f} ({bacc_std:.03f})
    Cohen Kappa: {ck:.03f} ({ck_std:.03f})
    Accuracy: {acc:.03f} ({acc_std:.03f})
    """)
    
    ress = ress.append({
        'name': str(encoder),
        'bacc': bacc, 'bacc_std': bacc_std,
        'ck': ck, 'ck_std': ck_std,
        'acc': acc, 'acc_std': acc_std,
        'tt': tt, 'tt_std': tt_std
    }, ignore_index=True)

# %%
# ---- Scaling

dataset = "adult"
df = pd.read_csv(
    f"{u.DATASET_FOLDER}/{dataset}.csv").sample(1000, random_state=1444)
X, y = u.pre2process(df)

encoder = enc.TargetEncoder()

DT = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10)

scores_list = [
    balanced_accuracy_score,
    u.cohen_kappa,
    accuracy_score,
]
scores = {
    s.__name__: make_scorer(s)
    for s in (scores_list)
}

# -- Execution

CT = ColumnTransformer([
    (
        "encoder",
        encoder,
        [col for col in X.columns if X[col].dtype not in ("float64", "int64")]
    ),
], remainder="passthrough")

PP = Pipeline([
    ("preproc", CT),
    ("model", DT)
])

PP.fit(X, y)

# performance
out = cross_validate(PP, X, y,
                     scoring=scores,
                     n_jobs=-2,
                     verbose=0,
                     cv=10
                     )

bacc_ = out['test_balanced_accuracy_score']
ck_ = out['test_cohen_kappa']
acc_ = out['test_accuracy_score']

bacc, bacc_std = bacc_.mean(), bacc_.std()
ck, ck_std = ck_.mean(), ck_.std()
acc, acc_std = acc_.mean(), acc_.std()

# feature names
try:
    ftn = CT.transformers_[0][1].feature_names
except:
    ftn = ['constant']
ftn.extend([col for col in X.columns if X[col].dtype in ("float64", "int64")])

fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
plot_tree(PP['model'], feature_names=ftn, ax=ax)
ax.set_title(
    f"""{str(encoder)}
Balanced Accuracy: {bacc:.03f} ({bacc_std:.03f})
Cohen Kappa: {ck:.03f} ({ck_std:.03f})
Accuracy: {acc:.03f} ({acc_std:.03f})
"""
)

# %% Test the idea

dd = X[['cat_4']]
dd.loc[:, 'target'] = y
dd.loc[:, 'te'] = dd.groupby('cat_4').target.transform(np.mean)
ee = dd.groupby('cat_4').target.agg(np.mean).to_dict()

ee2 = {}
i = 0
for k, v in ee.items():
    if v > 0.1:
        ee2[k] = v, 0
    else:
        ee2[k] = 0.1, i
        i += 1

dd.loc[:, 'te2_0'] = [x[0] for x in dd['cat_4'].map(ee2)]
dd.loc[:, 'te2_1'] = [x[1] for x in dd['cat_4'].map(ee2)]

X1 = dd[['te']]
X2 = dd[['te2_0', 'te2_1']]
X3 = OneHotEncoder().fit_transform(dd, y)

outs = []
for xx in (X1, X2, X3):
    out = cross_validate(DT, xx, y,
                         scoring=scores,
                         n_jobs=-2,
                         verbose=0,
                         cv=10
                         )
    outs.append(out)


for name, out in zip(('TE', 'DTE', 'OHE'), outs):
    bacc_ = out['test_balanced_accuracy_score']
    ck_ = out['test_cohen_kappa']
    acc_ = out['test_accuracy_score']

    bacc, bacc_std = bacc_.mean(), bacc_.std()
    ck, ck_std = ck_.mean(), ck_.std()
    acc, acc_std = acc_.mean(), acc_.std()

    print(f"""{name}
    Balanced Accuracy: {bacc:.03f} ({bacc_std:.03f})
    Cohen Kappa: {ck:.03f} ({ck_std:.03f})
    Accuracy: {acc:.03f} ({acc_std:.03f})
    """)
