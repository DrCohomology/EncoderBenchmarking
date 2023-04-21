# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:44:13 2022

@author: federicom
"""
import glob
import itertools
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import openml
import os
import matplotlib as mpl
import pandas as pd
import random
import re
import sklearn
import string
import time

from catboost import CatBoostClassifier
from collections import defaultdict, Counter
from functools import reduce
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.pipeline import make_pipeline
from joblib import Parallel, delayed
from importlib import reload
from itertools import product
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from numba import njit
from openml.datasets import edit_dataset, fork_dataset, get_dataset
from rpy2.robjects.packages import importr
from scipy.stats import chi2_contingency, spearmanr, kendalltau
import seaborn as sns
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM as bgmm
from tqdm import tqdm

from sklearn.compose import ColumnTransformer
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

import src.encoders as e
import src.utils as u
import src.results_concatenator as rc
reload(e)
reload(u)
reload(rc)

#%% Label encoder VS Ordinal encoder

be = e.BinaryEncoder()

df = pd.DataFrame([
    ("b", 0),
    ("a", 1),
    ("a", 0),
    ("c", 1),
    ("d", 0)
], columns=["A", "y"])

X = df.drop("y", axis=1)
y = df.y

xl = be.fit_transform(X)



#%% Sum encoder
from sklearn.linear_model import LinearRegression
from category_encoders import SumEncoder, OneHotEncoder
from scipy.stats import pearsonr as corr


df = pd.DataFrame([
    ("b", 0),
    ("a", 1),
    ("a", 0),
    ("c", 1),
], columns=["A", "y"])

se = SumEncoder()
ohe = OneHotEncoder()
lr = LinearRegression(fit_intercept=False)

X = df.drop("y", axis=1)
y = df.y

XE = se.fit_transform(X)
XE2 = ohe.fit_transform(X)
lr.fit(XE, y)
print(lr.coef_)

lr2 = LinearRegression(fit_intercept=False)
lr2.fit(XE2, y)
print(lr2.coef_)

# for col in XE2:
#     print(corr(XE2[col], y))


#%%

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "times new roman",
    'axes.unicode_minus': False
})


df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [10, 20, 30]}, index=range(3))

a = "a"

sns.set_theme(style="whitegrid", font_scale=0.5)
sns.despine(trim=True, left=True)

grid = sns.FacetGrid(df, col=f"{a}", row="b",
                     margin_titles=True, sharex="all", sharey="all",
                     aspect=0.4)
grid.set_titles(row_template="${row_name}$", col_template="${col_name}$")

grid.map_dataframe(sns.lineplot)

# t = np.linspace(0.0, 1.0, 100)
#
# s = np.cos(4 * np.pi * t) + 2
#
# fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
# ax.plot(t, s)
#
# ax.set_xlabel(r'\textbf{time (s)}')
# ax.set_ylabel('\\textit{Velocity (\N{DEGREE SIGN}/sec)}', fontsize=16)
# ax.set_title(r'\TeX\ is Number $\sum_{n=1}^\infty'
#              r'\frac{-e^{i\pi}}{2^n}$!', fontsize=16, color='r')
plt.show()

#%% Matricial computation of kendall distance. it is inefficient :(

from numba import njit

r1 = np.array((1, 3, 2, 4))
r2 = np.array((2, 4, 1, 3))

"""
1. Take a matrix by rolling the vector
2. Take the difference matrix with the first element of each row
3. hadamard product, take sign, opposite, add 1, divide by 2
"""

@njit
def get_m(r):

    a = np.zeros((len(r), len(r)))
    for i in range(len(a)):
        a[i] = np.roll(r, i)



    return np.triu(a).T

@njit
def get_d(m):
    """
    d[i, j] = m[i, 0] - m[i, j+1] for j = 0...
    """
    return -np.tril(np.cumsum(np.diff(m), axis=1)[1:])

@njit
def dk(d1, d2):
    return np.sum(-np.minimum(np.sign(d1*d2), 0))

def d_kendall2(r1, r2):
    return dk(get_d(get_m(r1)), get_d(get_m(r2)))

def d_kendall3(r1, r2):
    return np.sum(-np.minimum(np.sign(
        -np.tril(np.cumsum(np.diff(np.triu([np.roll(r1, i) for i in np.arange(len(r1))]).T), axis=1)[1:]) *\
        -np.tril(np.cumsum(np.diff(np.triu([np.roll(r2, i) for i in np.arange(len(r2))]).T), axis=1)[1:])
    ), 0))


#%% not sure what this is
np.random.seed(10)

df = pd.read_csv(u.RESULT_FOLDER + "\\main6_final.csv")

dataset = "adult"
model1 = "DecisionTreeClassifier"
model2 = "LGBMClassifier"
scoring1 = "roc_auc_score"
scoring2 = "accuracy_score"

def get_rank(df, dataset, model, scoring):
    return df.query("dataset == @dataset and model == @model and scoring == @scoring").groupby("encoder")\
           .cv_score.agg(["mean", "std"]).sort_values("mean", ascending=False).index

rank11 = get_rank(df, dataset, model1, scoring1)
rank12 = get_rank(df, dataset, model1, scoring2)
rank21 = get_rank(df, dataset, model2, scoring1)

df2 = pd.read_csv(u.RESULT_FOLDER + "\\main8_final.csv")
rank112 = get_rank(df2, dataset, model1, scoring1)

ranks = [rank11, rank12, rank21]

for (i1, r1), (i2, r2) in itertools.product(enumerate(ranks), repeat=2):
    print(i1, i2, kendalltau(r1, r2, variant='b').correlation)

#%%

dataset = "tic-tac-toe"
X, y, categorical_indicator, attribute_names = get_dataset(u.DATASETS[dataset]).get_data(
    target=get_dataset(u.DATASETS[dataset], download_data=False).default_target_attribute, dataset_format="dataframe"
)
X = X.dropna(axis=0, how="all").dropna(axis=1, how="all").sample(10)
y = pd.Series(e.LabelEncoder().fit_transform(y[X.index]), name="target")

X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

enc1 = e.MEstimateEncoder(m=10)
XE = enc1.fit_transform(X, y)


col = "top-left-square"
c1 = XE[col]

encoding = defaultdict(lambda: {})

X = X.join(y.squeeze())
global_mean = y.mean()
w = 10
temp = X.groupby(col).agg(['sum', 'count'])
temp['ME'] = (temp.target['sum'] + w * global_mean) / (temp.target['count'] + w)
# temp['TE'] = temp.target['sum'] / temp.target['count']
encoding[col].update(temp['ME'].to_dict())







#%% Identify the missing 30th small dataset

# candidate: adult
dataset = "adult"
completed_dataset = "kr-vs-kp"

df_notuning = rc.concatenate_results("29dats")
df = rc.concatenate_results("main6 results with tuning and 6-encoders")

required_encs = df_notuning.encoder.unique()
completed_encs = df.loc[df.dataset == dataset].encoder.unique()

required_models = df_notuning.model.unique()
completed_models = df.loc[df.dataset == dataset].model.unique()

torun_encs = set(required_encs) - set(completed_encs)
torun_models = set(required_models) - set(completed_models)

# print(torun_encs)
# print(torun_models)

pk = ["encoder", "model", "scoring"]
dfe = df.loc[df.dataset == dataset].set_index(pk)
dfc = df.loc[df.dataset == completed_dataset].set_index(pk)

torun = set(dfc.index) - set(dfe.index)
for encoder, model, scoring in torun:
    print(encoder, model, scoring)






#%%

X, y, categorical_indicator, attribute_names = get_dataset(u.DATASETS[dataset]).get_data(
    target=get_dataset(u.DATASETS[dataset], download_data=False).default_target_attribute, dataset_format="dataframe"
)
X = X.dropna(axis=0, how="all").dropna(axis=1, how="all")
y = pd.Series(e.LabelEncoder().fit_transform(
    y[X.index]), name="target")

X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

cats = X.select_dtypes(include=("category", "object")).columns
nums = X.select_dtypes(exclude=("category", "object")).columns

cat_imputer = e.DFImputer(u.SimpleImputer(strategy="most_frequent"))
num_imputer = e.DFImputer(u.SimpleImputer(strategy="median"))

scaler = u.RobustScaler()
solvers = ["lbfgs", "saga", "liblinear"]
model = None
scoring = u.roc_auc_score

importr("lme4")
importr("base")
importr("utils")

encoders = [e.OneHotEncoder(), e.DropEncoder()]
run = True
if run:
    runtimes = defaultdict(lambda: [])
    scores = defaultdict(lambda: [])
    cv = StratifiedKFold(n_splits=5, random_state=None)
    for icv, (tr, te) in tqdm(enumerate(cv.split(X, y))):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        Xtr, ytr = Xtr.reset_index(drop=True), ytr.reset_index(drop=True)
        Xte, yte = Xte.reset_index(drop=True), yte.reset_index(drop=True)

        for encoder in encoders:
            cats = X.select_dtypes(include=("category", "object")).columns
            nums = X.select_dtypes(exclude=("category", "object")).columns
            catpipe = Pipeline([("imputer", cat_imputer), ("encoder", encoder)])
            numpipe = Pipeline([("imputer", num_imputer), ("scaler", scaler)])
            prepipe = ColumnTransformer([("encoder", catpipe, cats), ("scaler", numpipe, nums)],
                                        remainder="passthrough")

            XEtr = prepipe.fit_transform(Xtr, ytr)
            for solver in solvers:
                model = u.LogisticRegression(solver=solver)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s = time.time()
                    model.fit(XEtr, ytr)
                    runtimes[solver].append(time.time() - s)

                scores[solver].append(scoring(yte, model.predict(prepipe.transform(Xte))))

    runtimes = pd.DataFrame(runtimes)
    scores = pd.DataFrame(scores)

fig, axes = plt.subplots(1, 2)
fig.suptitle(f"{dataset}, shape: {X.shape}")
ax = axes[0]
ax.set_title("Run time")
sns.boxplot(runtimes, ax=ax)
ax = axes[1]
ax.set_title(scoring.__name__)
sns.boxplot(scores, ax=ax)

plt.tight_layout()
plt.show()


#%%
experiment_name = "complete results"
df = rc.concatenate_results(experiment_name, clean=False)

df = df.loc[df.dataset.isin(["adult"])]

encs = ["DE", "OHE", "RGLMME", "CV5RGLMME"]

fig, axes = plt.subplots(1, len(encs), sharey="all")
fig.suptitle("Runtimes in hours for the adult dataset")
for ax, enc in zip(axes.flatten(), encs):

    dfe = df.loc[df.encoder == enc]
    # ax.set_title(f"{enc}")
    ax.boxplot(dfe.tuning_time / 3600)
    ax.set_yscale("log")
    ax.set_xlabel(f"{enc}")
    ax.set_xticks([])
plt.show()

mods = df.model.unique()
fig, axes = plt.subplots(1, len(mods), sharey="all")
fig.suptitle("Runtimes in hours for the adult dataset")
for ax, mod in zip(axes.flatten(), mods):
    dfe = df.loc[df.model == mod]
    ax.boxplot(dfe.tuning_time / 3600)
    ax.set_yscale("log")
    ax.set_xlabel(f"{mod}")
    ax.set_xticks([])
plt.show()
#%%
fig, axes = plt.subplots(len(mods), len(encs), sharey="all", figsize=(10, 10))
fig.suptitle("Runtimes in hours for the adult dataset")
for axs, mod in zip(axes, mods):
    for ax, enc in zip(axs, encs):
        ax.set_ylabel(f"{u.get_acronym(mod, underscore=False)}")
        dfe = df.loc[(df.model == mod) & (df.encoder == enc)]
        ax.boxplot(dfe.tuning_time / 3600)
        ax.set_yscale("log")
        ax.set_xlabel(f"{enc}")
        ax.set_xticks([])

plt.show()

#%%
r1 = [1, 1, 3, 1, 4.5, 4.5, 6]
r2 = [6, 1, 3, 3, 3, 5, 0]

r1d = [0, 0, 1, 0, 2, 2, 3]
r2d = [4, 1, 2, 2, 2, 3, 0]

print("not dense")
print(spearmanr(r1, r2).correlation)
print(kendalltau(r1, r2).correlation)
print(kendalltau(r1, r2, variant='c').correlation)

print("dense")
print(spearmanr(r2d, r1d).correlation)
print(kendalltau(r2d, r1d).correlation)
print(kendalltau(r2d, r1d, variant='c').correlation)


#%% is RGLMME faster?
did = u.DATASETS["adult"]

X, y, categorical_indicator, attribute_names = get_dataset(did).get_data(
    target=get_dataset(did, download_data=False).default_target_attribute, dataset_format="dataframe"
)
X = X.dropna(axis=0, how="all").dropna(axis=1, how="all")
y = pd.Series(e.LabelEncoder().fit_transform(
    y[X.index]), name="target")

X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

cats = X.select_dtypes(include=("category", "object")).columns
nums = X.select_dtypes(exclude=("category", "object")).columns

cat_imputer = e.DFImputer(u.SimpleImputer(strategy="most_frequent"))
num_imputer = e.DFImputer(u.SimpleImputer(strategy="median"))

scaler = u.RobustScaler()
model = u.LGBMClassifier()

encoders = [e.GLMMEncoder(), e.RGLMMEncoder(rlibs=None)]

runtime = {}
for encoder in tqdm(encoders):
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
        ("model", model)
    ])


    start = time.time()
    pipe.fit(X, y)
    runtime[str(encoder)] = time.time() - start



#%%

# load df_concatenated
result_folder = os.path.join(u.RESULT_FOLDER, "big_benchmark_backup")
df_concatenated = pd.read_csv(os.path.join(result_folder, "_concatenated.csv"))


#%% statsmodels is slow af

df = pd.read_csv(u.DATASET_FOLDER + "/adult.csv")
df["target"] = e.LabelEncoder().fit_transform(df["target"])

# 2: with factors, 3: with another formula
df_ = df.copy()[["cat_0", "target"]]
df_.columns = ["x", "y"]


# ---- inner part
df2 = df_.copy()


# OE = e.OrdinalEncoder()
# df2.x = OE.fit_transform(df2.x, df2.y)
start = time.time()

model2 = bgmm.from_formula('y ~ 1', {'coefficient': '0 + C(x)'}, df2).fit_vb() # C(X) -> treat x as categorical
timetime = time.time() - start

# col_0
better_names = [re.search(r"\[(.+)\]", index_name).groups()[0] for index_name in model2.model.vc_names]
estimate2 = pd.Series(model2.vc_mean, index=better_names)

print("GLMM time: ", timetime)

# ---- whole encoder

df2 = df_.copy()
GE = category_encoders.GLMMEncoder()
start = time.time()
GE.fit(df2.x, df2.y)
b = GE.transform(df2.x)
tim = time.time() - start

print("encoder time: ", tim)

# %% is BayesSearch very slow? No

did = u.DATASETS["Agrawal1"]

X, y, categorical_indicator, attribute_names = get_dataset(did).get_data(
    target=get_dataset(did, download_data=False).default_target_attribute, dataset_format="dataframe"
)
X = X.dropna(axis=0, how="all").dropna(axis=1, how="all")
y = pd.Series(e.LabelEncoder().fit_transform(
    y[X.index]), name="target")

X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

cats = X.select_dtypes(include=("category", "object")).columns
nums = X.select_dtypes(exclude=("category", "object")).columns

cat_imputer = e.DFImputer(u.SimpleImputer(strategy="most_frequent"))
num_imputer = e.DFImputer(u.SimpleImputer(strategy="median"))

encoder = e.Discretized(e.TargetEncoder())
# encoder = e.CVRegularized(e.GLMMEncoder())
scaler = u.RobustScaler()
model = SVC()

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

bayes_search_space = {
    "model__C": Real(0.1, 2),
    "model__gamma": Real(0.1, 100, prior="log-uniform")
}
grid_search_space = {
    "model__C": [0.1, 0.5, 1, 2],
    "model__gamma": [0.1, 1, 10, 100]
}

score = accuracy_score

cv = StratifiedKFold(
    n_splits=3, random_state=1000, shuffle=True
)

GS = GridSearchCV(pipe, param_grid=grid_search_space, n_jobs=1, cv=cv, scoring=score)
BS = BayesSearchCV(
    pipe,
    search_spaces=bayes_search_space,
    n_jobs=1,
    cv=cv,
    verbose=False,
    n_iter=16,
    random_state=10 + 1,
    scoring=score,
    refit=True
)

tgrids = []
tbayess = []
tpipes = []
for _ in tqdm(range(0)):
    start = time.time()
    pipe.fit(X, y)
    tpipes.append(time.time() - start)

    print("Yeha")
    start = time.time()
    GS.fit(X, y)
    tgrids.append(time.time() - start)

    print("OH")

    start = time.time()
    BS.fit(X, y)
    tbayess.append(time.time() - start)

tgrid = np.array(tgrids)
tbayes = np.array(tbayess)

print(f"Grid: {tgrid.mean()}; Bayes: {tbayes.mean()}")

# %% encoder mimicking the splitting structure of a decision tree

from sklearn import clone

did = u.DATASETS["KDDCup09_appetency"]

X, y, categorical_indicator, attribute_names = get_dataset(did).get_data(
    target=get_dataset(did, download_data=False).default_target_attribute, dataset_format="dataframe"
)

X = X.dropna(axis=0, how="all").dropna(axis=1, how="all")
y = pd.Series(e.LabelEncoder().fit_transform(
    y[X.index]), name="target")

X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

cats = X.select_dtypes(include=("category", "object")).columns
nums = X.select_dtypes(exclude=("category", "object")).columns

cat_imputer = e.DFImputer(u.SimpleImputer(strategy="most_frequent"))
num_imputer = e.DFImputer(u.SimpleImputer(strategy="median"))

encoder = e.Discretized(e.TargetEncoder())
scaler = u.RobustScaler()
model = SVC()

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

# TE = e.TargetEncoder()
# DTE = e.Discretized(e.TargetEncoder())
#
# a = DTE.fit_transform(X[cats], y)
# b = TE.fit_transform(X[cats], y)
#
# print(np.isnan(b).sum())

pipe.fit(X, y)
ypr = pipe.predict(X)
print(ypr.shape)


# %% DTE


def gini(x: np.ndarray):
    return np.sum(np.abs([xi - xj for xi, xj in product(x, repeat=2)])) / (2 * len(x) * x.sum())


def gini_impurity(x: np.ndarray):
    return 1 - (np.mean(x == 0) ** 2 + np.mean(x == 1) ** 2)


class MySplitter:

    def __init__(self):
        self.thresholds = {}

    def fit(self, X, y=None):
        X = X.copy()
        for col in X.columns:
            lowest_impurity_thr = None
            lowest_impurity = np.infty
            for i, thr in enumerate(sorted(X[col].unique())):
                if i == len(X[col].unique()) - 1:
                    break
                y1 = y.loc[X[col] <= thr]
                y2 = y.loc[X[col] > thr]

                g1 = gini_impurity(y1)
                g2 = gini_impurity(y2)

                w1 = len(y1) / len(y)
                w2 = len(y2) / len(y)

                impurity = np.inner((w1, w2), (g1, g2))

                if impurity < lowest_impurity:
                    lowest_impurity = impurity
                    lowest_impurity_thr = thr
            self.thresholds[col] = lowest_impurity_thr
        return self

    def transform(self, X, y=None):
        pass


"""
This implementation is stupid and does not work: I am just learnin a TargetEncoder in a more complicated way. 
Ideally, what I would have is a blowup of dimensions, where I split eacgh column according to every other column. 
Or I can just split according to the most promising column, ie perfect up to first layer of DT
"""


class DTEncoder(e.Encoder):
    def __init__(self, splitter=MySplitter(), base_encoder: e.Encoder() = e.TargetEncoder(), default=-1, **kwargs):
        super().__init__(default=default, **kwargs)
        self.splitter = splitter
        self.base_encoder = base_encoder
        self.fold_encoders = {}
        self.initial_encoder = e.TargetEncoder()
        self.thresholds = {}

    def fit(self, X: pd.DataFrame, y=None, **kwargs):
        """
        Encode columns with TE
        Split the columns according to the thresholds, encode each part with its own EargetEncoder
        """
        X = X.copy()

        self.cols = X.columns

        self.fold_encoders = {
            col: [clone(self.base_encoder) for _ in range(2)]
            for col in self.cols
        }

        XE = self.initial_encoder.fit_transform(X, y)
        self.thresholds = self.splitter.fit(XE, y).thresholds

        for col in self.cols:
            c1 = XE[col] <= self.thresholds[col]
            c2 = XE[col] > self.thresholds[col]

            self.fold_encoders[col][0].fit(X.loc[c1, col].to_frame(), y.loc[c1])
            self.fold_encoders[col][1].fit(X.loc[c2, col].to_frame(), y.loc[c2])

            enc0 = self.fold_encoders[col][0].encoding[col]
            enc1 = self.fold_encoders[col][1].encoding[col]

            temp = {
                cat: (enc0[cat], enc1[cat])
                for cat in set(enc0.keys()).union(set(enc1.keys()))
            }

            self.encoding[col].update(temp)

        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        X = X.copy().astype(object)

        # defines the splits
        XE = self.initial_encoder.transform(X)

        for col in self.cols:
            c1 = XE[col] <= self.thresholds[col]
            c2 = XE[col] > self.thresholds[col]

            X.loc[c1, col] = self.fold_encoders[col][0].transform(X.loc[c1, col].to_frame())
            X.loc[c2, col] = self.fold_encoders[col][1].transform(X.loc[c2, col].to_frame())

        return X.applymap(float)


X, y, categorical_indicator, attribute_names = get_dataset(1590).get_data(
    target=get_dataset(1590, download_data=False).default_target_attribute, dataset_format="dataframe"
)

X = X.iloc[:2000]

cats = X.select_dtypes(include=("category", "object")).columns
nums = X.select_dtypes(exclude=("category", "object")).columns

y = pd.Series(e.LabelEncoder().fit_transform(y[X.index]), name="target")

DTE = DTEncoder()
DTE.fit(X[cats], y)

TE = e.TargetEncoder()
TE.fit(X[cats], y)

X1 = DTE.transform(X[cats])
X2 = TE.transform(X[cats])


# %% rank agreement
def agreement(df, r1, r2, strict=False):
    score = 0
    for enc1, data1 in df.iterrows():
        for enc2, data2 in df.iterrows():
            # if enc1 is better than enc2 according to r1, but not according to r2 -> problem
            # same as opposite holds.
            # ie if the rankings disagree or not
            # if two are equal according to a rank, keep it as a refinement and do not penalize. Can be improved
            if strict:
                if (data1[r1] - data2[r1]) * (data1[r2] - data2[r2]) > 0:
                    score += 1
            else:
                if (data1[r1] - data2[r1]) * (data1[r2] - data2[r2]) >= 0:
                    score += 1
    score /= len(df) ** 2
    return score


print("-" * 10, "Loosen agreement:")
for r1, r2 in product(df_ranks.columns, repeat=2):
    # stupid condition
    if r1 < r2:
        print(r1, r2, agreement(df_ranks, r1, r2, strict=False))

print("-" * 10, " Strict agreement:")
for r1, r2 in product(df_ranks.columns, repeat=2):
    # stupid condition
    if r1 < r2:
        print(r1, r2, agreement(df_ranks, r1, r2, strict=True))

# %% LEV distance

dataset = get_dataset(1590)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute, dataset_format="dataframe"
)

X = X.dropna(axis=0, how="all").dropna(axis=1, how="all")
y = pd.Series(e.LabelEncoder().fit_transform(
    y[X.index]), name="target")

X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)


def lev_dist(a, b):
    def min_dist(s1, s2):

        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2

        # no change required
        if a[s1] == b[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + min(
            min_dist(s1, s2 + 1),  # insert character
            min_dist(s1 + 1, s2),  # delete character
            min_dist(s1 + 1, s2 + 1),  # replace character
        )

    return min_dist(0, 0)


for s1, s2 in itertools.product(X.education.unique(), repeat=2):
    print(s1, s2, lev_dist(s1, s2))


# %%

class RandomModel():

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return pd.Series(np.random.randint(0, 2, len(X)))


np.random.seed(32)
n = 10000
l = 20

m = np.random.randint(0, 2, (n, l))

X = pd.DataFrame(m)
y = (np.average(m, axis=1).round())

# model1 = RandomModel()
model1 = RandomForestClassifier()
# model2 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
scoring = u.roc_auc_score

Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.8)

# train model1 on X, predict y1, compute y2
model1.fit(Xtr, ytr)
y_ = (ytr != model1.predict(Xtr)).astype(int)

# train model2 to learn the correct predictions of model1
model2.fit(Xtr, y_)

train_acc = scoring(ytr, model1.predict(Xtr))
try:
    pred_train_acc = scoring((ytr != model1.predict(Xtr)).astype(int), model2.predict(Xtr))
except:
    pred_train_acc = 1

test_acc = scoring(yte, model1.predict(Xte))
pred_test_acc = scoring((yte != model1.predict(Xte)).astype(int), model2.predict(Xte))

print(
    f"train: {train_acc:.04f}, test: {test_acc:.04f}, pred_train: {pred_train_acc:.04f}, pred_test: {pred_test_acc:.04f}")

# %%

odf = openml.datasets.list_datasets(output_format="dataframe")
cat1 = odf.loc[(odf.NumberOfClasses > 0) & (odf.NumberOfSymbolicFeatures >= 2)].shape[0]
cat2 = odf.loc[((odf.NumberOfClasses == 0) | (odf.NumberOfClasses.isna())) & (odf.NumberOfSymbolicFeatures >= 1)].shape[
    0]
num1 = odf.loc[(odf.NumberOfClasses > 0) & (odf.NumberOfSymbolicFeatures < 2)].shape[0]
num2 = odf.loc[((odf.NumberOfClasses == 0) | (odf.NumberOfClasses.isna())) & (odf.NumberOfSymbolicFeatures < 1)].shape[
    0]

bc = odf.loc[odf.NumberOfClasses == 2].shape[0]

print(f'BC: {bc}')
print(f"total: {odf.shape[0]}, cat: {cat1 + cat2}, num: {num1 + num2}")
# %%
dataset = get_dataset(1169)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute, dataset_format="dataframe"
)

X = X.dropna(axis=0, how="all").dropna(axis=1, how="all")
y = pd.Series(e.LabelEncoder().fit_transform(
    y[X.index]), name="target")

X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

cats = X.select_dtypes(include=("category", "object")).columns
nums = X.select_dtypes(exclude=("category", "object")).columns

cat_imputer = e.DFImputer(u.SimpleImputer(strategy="most_frequent"))
num_imputer = e.DFImputer(u.SimpleImputer(strategy="median"))
encoder = e.TargetEncoder()
scaler = u.RobustScaler()
model = u.LGBMClassifier(random_state=3, n_estimators=500, metric="None", verbosity=-1)
# model = u.DecisionTreeClassifier()
scoring = u.roc_auc_score

catpipe = Pipeline([
    ("imputer", cat_imputer),
    ("encoder", encoder)
])
numpipe = Pipeline([
    # ("imputer", num_imputer),
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
    ("model", model)
])

search_space = u.get_pipe_search_space_one_encoder(model, encoder)
cv = StratifiedKFold(n_splits=5)
for tr, te in cv.split(X, y):
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    # out = cross_val_score(pipe, X, y, cv=cv, verbose=-1)

    Xtrtr, Xtrval, ytrtr, ytrval = train_test_split(Xtr, ytr, test_size=0.5, random_state=11)

    Xtrtr.reset_index(drop=True, inplace=True)
    ytrtr.reset_index(drop=True, inplace=True)
    # Xtrtr, ytrtr = Xtr, ytr

    Xtrval, ytrval = Xtrval.reset_index(drop=True), ytrval.reset_index(drop=True)

    XIcat = cat_imputer.fit_transform(Xtrtr, ytrtr)
    # XInum = num_imputer.fit_transform(XIcat, ytrtr)

    XIE = encoder.fit_transform(XIcat, ytrtr)
    # XEEE = encoder.fit_transform(Xtrtr, ytrtr)

    XE = CT.fit_transform(Xtrtr, ytrtr)
    if np.isnan(XE).sum():
        print(np.isnan(XE).sum())
        break
    # pipe.fit(Xtrtr, ytrtr)

    Xtrans = CT.transform(Xtrval)

    pipe.fit(
        Xtrtr, ytrtr,
        model__eval_set=[(Xtrans, ytrval)],
        model__eval_metric=u.get_lgbm_scoring(scoring),
        model__callbacks=[early_stopping(50, first_metric_only=True), log_evaluation(-1)]
    )


# %%
def custom_metric(y_true, y_pred):
    metric_name = 'custom'
    value = 0.1
    is_higher_better = False
    return metric_name, value, is_higher_better


def lgbm(scoring):
    def lgbm_scoring(y_true, y_pred):
        y_pred = np.round(y_pred)
        return scoring.__name__, scoring(y_true, y_pred), True

    return lgbm_scoring


y = pd.Series(e.LabelEncoder().fit_transform(
    y[X.index]), name="target")

X_train, X_eval, y_train, y_eval = train_test_split(X.select_dtypes(exclude=("category", "object")), y, test_size=0.1,
                                                    random_state=1)

clf = LGBMClassifier(objective="binary", n_estimators=10000, random_state=1, metric="None", verbose=-1)
eval_set = [(X_eval, y_eval)]

clf.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    early_stopping_rounds=100,
    eval_metric=lgbm(u.accuracy_score),
    verbose=1
)
print(u.accuracy_score(y_eval, clf.predict(X_eval)))

# %% test retrieval

good = {
    'credit-g': 31,
    'nursery': 959,
    'adult': 1590,
    'mv': 881,
    'kdd_internet_usage': 981,
    'KDDCup09_appetency': 1111,
    'KDDCup09_churn': 1112,
    'KDDCup09_upselling': 1114,
    'airlines': 1169,
    'Agrawal1': 1235,
    'bank_marketing': 1461,
    'nomao': 1486,
    'altri': 'non ancora inseriti'
}

dataset = get_dataset(959)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute, dataset_format="dataframe"
)

X = X.dropna(axis=0, how="any")
y = pd.Series(e.LabelEncoder().fit_transform(y[X.index]), name="target")

# -- define pipeline
# !!! depends on openml syntax
cats = X.dtypes[X.dtypes == 'category'].index.to_list()
nums = X.dtypes[X.dtypes != 'category'].index.to_list()

from sklearn.preprocessing import RobustScaler

encoder = e.TargetEncoder()
scaler = RobustScaler()

encoder = e.CollapseEncoder()
scaler = e.CollapseEncoder()

models = [
    # CatBoostClassifier(verbose=0),
    LGBMClassifier(),
    RandomForestClassifier(),
    DecisionTreeClassifier()
]

scoring = roc_auc_score

# for each resample, run a cv (with tuning)
resample_size = 500

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=resample_size)

score = {}
perf = {}
for model in models:
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

    t = time.time()
    pipe.fit(X_train, y_train)
    t = time.time() - t

    perf[model.__class__.__name__] = t
    score[model.__class__.__name__] = scoring(y_test, pipe.predict(X_test))

print(json.dumps(perf, indent=4))
print(json.dumps(score, indent=4))
# %% OpenML find interesting datasets

# The same can be done with lesser lines of code
datasets = openml.datasets.list_datasets(output_format="dataframe")
datasets = datasets.query(""" NumberOfInstances >= 10000 \
                          and NumberOfSymbolicFeatures > 1 \
                          and version == 1 \
                          """)

datasets = datasets[['BNG' not in x for x in datasets.name]]

print(len(datasets.name.unique()))

# %% Single Experimenal design
# datasets = openml.datasets.list_datasets(output_format="dataframe")
# adult = datasets[datasets.name == "adult"].index[0].__int__()
dataset = get_dataset(31)

CVTE = enc.CVRegularized(enc.TargetEncoder(default=np.nan), n_splits=5)
scaler = u.RobustScaler()
dt = DecisionTreeClassifier(class_weight='balanced')

X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute, dataset_format="dataframe"
)

# Transform y in 0, 1
X = X.dropna(axis=0, how="any").reset_index(drop=True)
y = y[X.index]

y = pd.Series(enc.LabelEncoder().fit_transform(y), name="target")

cats = X.dtypes[X.dtypes == 'category'].index.to_list()
nums = X.dtypes[X.dtypes != 'category'].index.to_list()

CT = ColumnTransformer(
    [
        (
            "encoder",
            CVTE,
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

pipe = Pipeline([
    ("preproc", CT),
    ("model", dt)
])

pipe.fit(X, y)

# %%

CVTE = enc.CVRegularized(enc.TargetEncoder(default=np.nan), n_splits=2)
TE = enc.TargetEncoder()
CVGLMM = enc.CVRegularized(enc.GLMMEncoder(handle_unknown="return_nan"), n_splits=2)
GLMM = enc.GLMMEncoder(handle_unknown="return_nan")

dataset = "adult"
df = pd.read_csv(f"C:/Data/{dataset}.csv").iloc[:4].reset_index(drop=True)

score = u.roc_auc_score

X, y = u.pre2process(df)

X = X[[col for col in X.columns if "cat" in col]]

X1 = CVTE.fit_transform(X, y)
X2 = TE.fit_transform(X, y)
X3 = CVGLMM.fit_transform(X, y)
X4 = GLMM.fit_transform(X, y)

for Z in [X1, X2, X3, X4]:
    print(Z)

# %%
encoder = enc.WOEEncoder

# % Test new encoders
dataset = "adult"
df = pd.read_csv(f"C:/Data/{dataset}.csv").sample(n=100)
dt = DecisionTreeClassifier(class_weight='balanced')

score = u.roc_auc_score

X, y = u.pre2process(df)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, train_size=10000, test_size=10000, stratify=y, random_state=142925
# )
# X = X.reset_index(drop=True)
# y = y.reset_index(drop=True)

col = 'cat_0'
kfold = StratifiedKFold()

# OOFTE just maps an instance intot he avg target of the value in the first training fold
# where the instance appears
# X = X.reset_index(drop=True)
# new_x = X[[col]].copy()
# X.loc[:, 'TE'] = 0
# _d = {}
# for n_fold, (trn_idx, val_idx) in enumerate(kfold.split(new_x, y)):
#     trn_x = new_x.iloc[trn_idx].copy()
#     trn_x.loc[:, 'target'] = y.iloc[trn_idx]
#     val_x = new_x.iloc[val_idx].copy()
#     val_x.loc[:, 'target'] = y.iloc[val_idx]
#     val = trn_x.groupby(col)['target'].mean().to_dict()
#     _d[n_fold] = val
#     X.iloc[val_idx].TE = X.iloc[val_idx, col].map(val)
#     print(X.iloc[10000, :])

# LOOTE works as expected
new_x = X[[col]].copy()
new_x.loc[:, 'target'] = y
a = (new_x.groupby(col)['target'].transform(np.sum) - y) \
    / (new_x.groupby(col)['target'].transform(len) - 1)
b = (new_x.groupby(col)['target'].apply(np.sum) - y) \
    / (new_x.groupby(col)['target'].apply(len) - 1)
c = (new_x.groupby(col)['target'].agg(['sum', 'count'])) \
    / (new_x.groupby(col)['target'].apply(len) - 1)
# X.loc[:, 'TE'] = a
# _d = X.groupby(col)['TE'].mean()

# d = new_x.groupby(col)['target']

# SmoothTE test

DT = DecisionTreeClassifier()
CT = ColumnTransformer([
    (
        "encoder",
        enc.SmoothedTE(),
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
    ("model", DT)
])

ss = u.get_pipe_search_space(DecisionTreeClassifier)
# ss.update({
#     "preproc__encoder__w": Integer(0, 30)
# })
outs = []
for w in tqdm(np.linspace(0, 10, 20)):
    PP = PP.set_params(preproc__encoder__w=w)
    out = u.tune_pipe(PP, X, y, ss, make_scorer(
        balanced_accuracy_score), n_iter=20, n_jobs=1)
    outs.append(out)
# %%
scores = [out[0]["best_score"] for out in outs]
depths = [out[0]["best_params"]["model__max_depth"] for out in outs]
plt.plot(np.linspace(0, 10, 20), scores)
plt.plot(np.linspace(0, 10, 20), depths)
# CM = e.CatMEW([e.SmoothedTE], RobustScaler, classes_as_arguments=True)
# DT = DecisionTreeClassifier()

# ss = u.get_pipe_search_space(DecisionTreeClassifier)
# ss.update({
#     'encoder__w': Real(0, 100)
# })

# BS = BayesSearchCV(
#     Pipeline([
#         ('encoder', CM),
#         ('model', DT)
#     ]),
#     ss
# )

# BS.fit(X)
# Xe = CM.fit_transform(X, y)

# %%
a = [1, 2] + 10 * [x for x in range(100)]
b = ['a', 'b', 'c'] + ['c'] * 100
f = itertools.product(a, b)
f = zip(a, b)

d = pd.DataFrame(f, columns=['aaaa', 'bbbb'])
e = pd.crosstab(d.aaaa, d.bbbb)
c2, p, dof, ex = chi2_contingency(e)

"we fail to reject H0 that they are independent"

# %% Add a categorical column at a time

dataset = "adult"
df = pd.read_csv(f"C:/Data/{dataset}.csv").sample(n=20000)
dt = DecisionTreeClassifier(class_weight='balanced')

score = u.roc_auc_score

X, y = u.pre2process(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=10000, test_size=10000, stratify=y, random_state=142925
)
#
# ---- Correlation analysis
catcols = [col for col in X.columns if 'cat' in col]
corrs = np.zeros([len(X.columns), len(X.columns)])
corrsdict = {}
for (i1, c1), (i2, c2) in tqdm(itertools.product(enumerate(catcols), repeat=2)):
    if i1 < i2:
        ct = pd.crosstab(X[c1], X[c2])
        chi2, p, *_ = chi2_contingency(ct)
        corrs[i1, i2] = p
        corrsdict[c1, c2] = p
    elif i1 == i2:
        pass
    else:
        corrs[i1, i2] = corrs[i2, i1]
        corrsdict[c1, c2] = corrsdict[c2, c1]

# %%


# ---- Experiment
np.random.seed(130)
catcols = np.random.permutation([col for col in X.columns if 'cat' in col])

train = []
test = []
used_columns = []
for col in catcols:
    encoder = enc.CatMEW([CountEncoder])

    used_columns.append(col)

    X_train_ = X_train[used_columns]
    X_test_ = X_test[used_columns]

    Xenc_train = encoder.fit_transform(X_train_, y_train)
    Xenc_test = encoder.transform(X_test_)

    dt.fit(Xenc_train, y_train)
    yp_train = dt.predict(Xenc_train)
    yp_test = dt.predict(Xenc_test)

    train.append(np.round(score(y_train, yp_train), 3))
    test.append(np.round(score(y_test, yp_test), 3))

fig, ax = plt.subplots()

ax.plot(used_columns, train, c='red', label='train')
ax.plot(used_columns, test, c='g', label='test')
ax.legend()

# %%
# Test LocalOptimizerEncoder
dataset = "adult"
df = pd.read_csv(f"C:/Data/{dataset}.csv")
dt = DecisionTreeClassifier(class_weight='balanced')
scores = [
    u.accuracy_score,
    u.roc_auc_score,
    u.r2_score,
    u.balanced_accuracy_score,
    u.confusion_matrix,
    u.precision_score,
    u.recall_score,
    u.f1_score,
    u.jaccard_score,
]
score = u.roc_auc_score

E = enc.LocalOptimizerEncoder(score=score, majority_class=True)
TE = enc.TargetEncoder()
OHE = OneHotEncoder()

X, y = u.pre2process(df)
X = X[[col for col in X.columns if 'cat' in col]]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=10000, test_size=10000, stratify=y, random_state=142925
)

train, test = {}, {}
ENCs = [E, TE, OHE]
ENC_names = [str(x).split("(")[0] for x in ENCs]
Xs = {}
for encoder, enc_name in zip(ENCs, ENC_names):
    Xenc_train = encoder.fit_transform(X_train, y_train)
    Xenc_test = encoder.transform(X_test)

    dt.fit(Xenc_train, y_train)
    yp_train = dt.predict(Xenc_train)
    yp_test = dt.predict(Xenc_test)

    train[enc_name] = np.round(score(y_train, yp_train), 5)
    test[enc_name] = np.round(score(y_test, yp_test), 5)
    Xs[enc_name] = Xenc_train

ymaj_test = (
    np.ones_like(y_test)
    if y_train.sum() / len(y_train) > 0.5
    else np.zeros_like(y_test)
)

print(score.__name__)
print("train:", json.dumps(train, indent=4))
print("test:", json.dumps(test, indent=4))
print("majority:", score(y_test, ymaj_test))

# %%
dataset = "promotion"
df = pd.read_csv(f"C:/Data/{dataset}.csv")
dt = DecisionTreeClassifier()
CM = enc.CatMEW([enc.DistanceEncoder])
CM2 = enc.CatMEW([enc.TargetEncoder])
CM3 = enc.CatMEW([OneHotEncoder])

X, y = u.pre2process(df)

X = pd.DataFrame(X.cat_2)

X_train, X_test, y_train, yt = train_test_split(
    X, y, train_size=10000, test_size=15000, random_state=14912
)

bas, das = [], []
CMs = [CM, CM2, CM3]
encs = []
yps = []
Xs = []
for cm in CMs:
    Xtr_enc = cm.fit_transform(X_train, y_train)
    Xte_enc = cm.transform(X_test)

    dt.fit(Xtr_enc, y_train)
    yptrain = dt.predict(Xtr_enc)
    yp = dt.predict(Xte_enc)

    Xs.append((Xtr_enc, Xte_enc))
    yps.append(yp)
    bas.append(balanced_accuracy_score(yt, yp))
    das.append(balanced_accuracy_score(y_train, yptrain))
print(X.value_counts())
print(bas)
print(das)

# %% Analyze kick, which gives the worse encoder performance

dataset = "kaggle_cat_dat_1"
df = pd.read_csv(f"C:/Data/{dataset}.csv")
dt = DecisionTreeClassifier()

repeat = 10
base = 1000
sizes = [base * x for x in range(1, 101, 20) if 2 * base * x < len(df)]
print(f"Total iterations: {len(sizes) ** 2}")

brs, bds = {}, {}
for train_size, test_size in tqdm(itertools.product(sizes, repeat=2)):
    (
        brs[train_size / base, test_size / base],
        bds[train_size / base, test_size / base],
    ) = ([], [])
    for rep in range(repeat):

        CM = enc.CatMEW([enc.CollapseEncoder])
        CM2 = enc.CatMEW([enc.TargetEncoder])
        CM3 = enc.CatMEW([enc.DistanceEncoder])

        X, y = u.pre2process(df)
        X_train, X_test, y_train, yt = train_test_split(
            X, y, train_size=train_size, test_size=test_size
        )

        bas = []
        CMs = [CM, CM2]
        for cm in CMs:
            pipe = Pipeline([("encoder", cm), ("model", dt)])
            pipe.fit(X_train, y_train)
            yp = pipe.predict(X_test)
            # print(cm.encoders, balanced_accuracy_score(yt, yp))
            bas.append(balanced_accuracy_score(yt, yp))
        brs[train_size / base, test_size / base].append(bas[1] / bas[0])
        bds[train_size / base, test_size / base].append(bas[1] - bas[0])
# %%

ls = [(x, y, np.mean(r)) for (x, y), r in bds.items()]
xs = [l[0] for l in ls]
ys = [l[1] for l in ls]
rs = [l[2] for l in ls]

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
fig.suptitle(f"{dataset}")

ax.scatter(xs, ys, c=rs, cmap="PiYG")
for i, r in enumerate(rs):
    ax.annotate(f"{r:.02f}", (xs[i], ys[i]), fontsize="smaller")
ax.set_xlabel("train_size")
ax.set_ylabel("test_size")
ax.set_title("difference of TargetEncoder on CollapseEncoder")

# %%

X0 = X_test.iloc[:10000]
y0 = y_test.iloc[:10000]
X1 = X_test.iloc[10000:200000]
y1 = y_test.iloc[10000:200000]
X2 = X_test.iloc[100000:]
y2 = y_test.iloc[100000:]

s0 = balanced_accuracy_score(y0, dt.predict(X0))
s1 = balanced_accuracy_score(y1, dt.predict(X1))
s2 = balanced_accuracy_score(y2, dt.predict(X2))
s_test = balanced_accuracy_score(y_test, dt.predict(X_test))

ss = [s0, s1, s2]
ls = [len(X0), len(X1), len(X2)]
avg = sum(l * s for (l, s) in zip(ls, ss)) / sum(ls)

print(avg - s_test)

# %%
df = pd.read_csv("C:/Data/kaggle_cat_dat_2.csv")
X, y = u.pre2process(df)
df = X.copy()
df["target"] = y

cm = enc.CatMEW([enc.CollapseEncoder], scaler=u.RobustScaler)
dt = DecisionTreeClassifier(random_state=2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1444, train_size=10000, test_size=10000
)

pipe = Pipeline([("encoder", cm), ("model", dt)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# print(f"{cm.encoders}, {cm.scaler}: ", balanced_accuracy_score(y_test, y_pred))
# print(f"{cm.encoders}, {cm.scaler}: ", u.cohen_kappa(y_test, y_pred))
# print(f"{cm.encoders}, {cm.scaler}: ", accuracy_score(y_test, y_pred))

feature_names = pipe["encoder"].transform(X_train).columns
feature_importances = pipe["model"].feature_importances_
f_imp = dict(zip(feature_names, feature_importances))

print(f_imp)


# %% Ease of sampling with KLdiv


@njit
def KLdiv(P, Q):
    P /= P.sum()
    Q /= Q.sum()
    PQ = np.column_stack((P, Q))
    tosum = np.array([p * np.log(p / q) for (p, q) in PQ])
    return tosum.sum()


@njit
def entropy(P):
    P /= P.sum()
    tosum = np.array([p * np.log(p) for p in P])
    return -tosum.sum()


dataset_name = "credit"
df = pd.read_csv(f"C:/Data/{dataset_name}.csv")
X, y = u.pre2process(df)
df = X.copy()
df["target"] = y

cds = {col: {} for col in df.columns}
for col in tqdm(df.columns):

    pp = df[col].value_counts().sort_index()
    entr = entropy((pp / pp.sum()).to_numpy())

    fracs = np.linspace(0.01, 0.99, 100)
    divs = []
    for f in fracs:
        df2 = df.sample(frac=f)
        qq = df2[col].value_counts().sort_index().reindex(pp.index).fillna(1)

        pp2 = (pp / pp.sum()).to_numpy()
        qq2 = (qq / qq.sum()).to_numpy()

        divs.append(KLdiv(pp2, qq2))
    cds[col] = {
        "H": entr,
        "divs": divs,
    }
# %%

for col, v in cds.items():
    divs = v["divs"]
    v["au_logdivs"] = np.sum(np.log10(divs)) / len(divs)
    v["au_divs"] = np.sum(divs) / len(divs)
es = [c["H"] for c in cds.values()]
aus = [c["au_logdivs"] for c in cds.values()]

plt.scatter(es, aus)
plt.xlabel("Entropy")
plt.ylabel("Integral of log KL divergence in sample size")

# %%
# Evolution of target values in dataset size
# Idea is that if we take the sample mean of a binary classificatoin, we have a bernoulli
# we can then compute the confidence interval of the actual mean (p) beloging to
# an interval around estimated p
# If we can show that TargetEncder should swap (quantistically) if two convidence intervals interlap...

df = pd.read_csv("C:/Data/kaggle_cat_dat_2.csv")
X, y = u.pre2process(df)
df = X.copy()
df["target"] = y

# let's balance it
d0 = df.loc[df.target == 0]
d1 = df.loc[df.target == 1]

assert len(d0) > len(d1)
df0 = df.sample(n=len(d1), random_state=1444)
d = pd.concat([df0, d1], ignore_index=True)

ttt, tstd, tcn, tconf = [[] for _ in range(4)]
random_states = range(1444, 1460)
for random_state in tqdm(random_states):
    df = d.sample(n=2 * len(d1), random_state=random_state, ignore_index=True)

    X = df.drop("target", axis=1)[["cat_14"]]
    y = df.target

    tt, std, cn, conf = [defaultdict(lambda: []) for _ in range(4)]

    xs = np.linspace(0.1, 1, 100)
    for val in X.cat_14.unique():
        for x in xs:
            yval = y.loc[X.cat_14 == val]
            ylim = yval.iloc[: int(x * len(yval))]
            tt[val].append(ylim.mean())
            std[val].append(ylim.std())
            cn[val].append(ylim.count())

            p = ylim.mean()
            n = ylim.count()
            conf[val].append(1.282 * np.sqrt(p * (1 - p) / n))
    tt = {val: np.array(arr) for val, arr in tt.items()}
    std = {val: np.array(arr) for val, arr in std.items()}
    conf = {val: np.array(arr) for val, arr in conf.items()}

    ttt.append(tt)
    tstd.append(std)
    tconf.append(conf)
# %%

colors = dict(
    zip(
        ttt[0].keys(),
        [
            "r",
            "b",
            "c",
            "g",
            "m",
            "blue",
            "maroon",
            "purple",
            "silver",
            "black",
            "coral",
            "crimson",
            "lime",
            "deeppink",
            "turquoise",
        ],
    )
)

fig, axes = plt.subplots(1, len(random_states), figsize=(30, 5), sharey=True)
for i, (tt, std, conf) in enumerate(zip(ttt, tstd, tconf)):
    ax = axes[i]

    for ival, val in enumerate(tt.keys()):
        m = tt[val]
        s = conf[val]
        ax.plot(xs, m, c=colors[val])
        ax.set_title(f"test {i}")
# fig, axes = plt.subplots(1, len(X.cat_14.unique()), figsize=(30,5), sharey=True)
# for i, val in enumerate(ttt.keys()):
#     m = ttt[val]
#     s = tconf[val]
#     c = tcn[val]

#     ax = axes[i]
#     ax.plot(xs, m)
#     ax.plot(xs, m + s, c='red', ls='--')
#     ax.plot(xs, m - s, c='red', ls='--')


#     ax.set_title(val)
# %% study rankings

c2i, i2c = u.cat2idx_dicts(ttt[0].keys())
ranks = defaultdict(lambda: [])
ranks2 = defaultdict(lambda: [])

# fix random order, time step
ro = 12

for ts in range(len(ttt[ro]["a"])):
    vals = [ttt[ro][key][ts] for key in ttt[ro].keys()]
    # get categories in edscending order
    idxs = list(np.argsort(vals))
    idxs.reverse()
    cats = [i2c[x] for x in idxs]

    for c, i in c2i.items():
        ranks[c].append(idxs.index(i))
        ranks2[c].append(cats.index(c))
order = {cat: rk[-1] for cat, rk in ranks.items()}

order = {k: v for k, v in sorted(order.items(), key=lambda item: item[1])}

order_ranks = {cat: ranks[cat] for cat in order.keys()}

fig, axes = plt.subplots(1, len(ranks), figsize=(30, 5), sharey=True)
fig.suptitle(f"Test {ro}")
for ic, (cat, rk) in enumerate(order_ranks.items()):
    ax = axes[ic]
    ax.plot(xs, rk, c="b")
    ax.set_title(cat)
    ax.invert_yaxis()
# %%

dt = RandomForestClassifier()
TE = enc.CatMEW([enc.TargetEncoder])

X = df.drop("target", axis=1)
y = df.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=1444)

# preprocessing = ColumnTransformer(
#     [
#         ("cat", categorical_encoder, categorical_columns),
#         ("num", numerical_pipe, numerical_columns),
#     ]
# )

pipe = Pipeline([("encoder", TE), ("model", dt)])
pipe.fit(X_train, y_train)

# cheeky change
a = pipe[0].encoders[0].encoding["cat_14"]

swaps = [
    # ('i', 'j'),
    # ('k', 'l'),
    # ('b', 'c'),
    # ('g', 'h'),
    # ('a', 'n')
]
for k1, k2 in swaps:
    a[k1], a[k2] = a[k2], a[k1]
# result = permutation_importance(pipe, X_test, y_test, n_jobs=-1, random_state=1444)

# sorted_idx = result.importances_mean.argsort()

# fig, ax = plt.subplots()
# ax.boxplot(
#     result.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx]
# )
# ax.set_title("Permutation Importances (test set)")
# fig.tight_layout()
# plt.show()

# print(f'depth: {dt.tree_.max_depth}')
print(f"test acc: {pipe.score(X_test, y_test)}")

# %%

fig, ax = plt.subplots(figsize=(10, 10), dpi=500)

plot_tree(dt, ax=ax, max_depth=3, feature_names=X.columns)

# %%

# How does TargetEncoder trained on the training set differ from TargetEncoder trained on the whole dataset?

df = pd.read_csv("C:/Data/kaggle_cat_dat_2.csv")
X, y = u.pre2process(df)
df = X.copy()
df["target"] = y

# let's balance it
d0 = df.loc[df.target == 0]
d1 = df.loc[df.target == 1]

if len(d0) > len(d1):
    df0 = df.sample(n=len(d1), random_state=1444)
    d = pd.concat([df0, d1], ignore_index=True)
    df = d.sample(n=10000, random_state=1445, ignore_index=True)
X = df.drop("target", axis=1)
y = df.target

X = X[["cat_14"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1448
)

# Train distribution
TEtr = enc.CatMEW([enc.TargetEncoder])
TEtr.fit(X_train, y_train)
XE_train = TEtr.transform(X_train)

dttr = DecisionTreeClassifier()
dttr.fit(XE_train, y_train)
y_predtr = dttr.predict(TEtr.transform(X_test))
sctr = balanced_accuracy_score(y_test, y_predtr)

# True distribution
TE = enc.CatMEW([enc.TargetEncoder])
TE.fit(X, y)
XE = TE.transform(X)

dt = DecisionTreeClassifier()
dt.fit(XE, y)
y_pred = dt.predict(TE.transform(X_test))
sc = balanced_accuracy_score(y_test, y_pred)

# get disagreements
XEE = X.join(XE.join(XE_train, rsuffix="_train"), lsuffix="_")
xee = XEE.loc[np.abs(XEE.TE_cat_14 - XEE.TE_cat_14_train) > 0]

# %% Analyze why CountEncoder is the best one
df = pd.read_csv("C:/Data/kaggle_cat_dat_2.csv").sample(
    n=20000, random_state=1444, ignore_index=True
)
X, y = u.pre2process(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=1444
)

CEW = enc.CatMEW(
    [enc.TargetEncoder, enc.CollapseEncoder, enc.CVTargetEncoder, BinaryEncoder]
)

pipe = Pipeline([("encoder", CEW), ("model", DecisionTreeClassifier())])
# pipe.fit(X_train, y_train)

# wev = (y_test, pipe.predict(X_test))

search_space = {
    "model": Categorical([DecisionTreeClassifier()]),
    "model__max_depth": Categorical([2, 10, None]),
}

tuning_result = {}
BS = {}


def f(score):
    # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1444)
    # BS = BayesSearchCV(
    #     pipe,
    #     search_spaces=search_space,
    #     n_jobs=1,
    #     cv=cv,
    #     verbose=100,
    #     n_iter=50,
    #     random_state=1444,
    #     scoring=make_scorer(score)
    # )
    # opt[score.__name__] = BS.fit(X, y)
    nn = score.__name__
    tuning_result[nn], BS[nn] = u.tune_pipe(
        pipe, X_train, y_train, search_space, make_scorer(score)
    )
    return tuning_result, BS


x = Parallel(n_jobs=-1, verbose=100)(
    delayed(f)(score)
    for score in (accuracy_score, balanced_accuracy_score, roc_auc_score)
)

# %%
encoder = enc.TargetEncoder
model = RandomForestClassifier

pipe = Pipeline([("encoder", encoder()), ("model", model())])

search_space = {
    "model": Categorical([model()]),
    "model__max_features": Real(0.5, 1, prior="uniform"),
    "model__bootstrap": Categorical([True, False]),
    "model__max_depth": Categorical([2, 10, None]),
}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1444)
BS = BayesSearchCV(
    pipe,
    search_spaces=search_space,
    n_jobs=1,
    cv=cv,
    verbose=100,
    n_iter=50,
    random_state=1444,
    scoring=make_scorer(sklearn.metrics.r2_score),
)
opt = BS.fit(X, y)

# %%
pp = Pipeline([("encoder", encoder()), ("model", model())])
pp.set_params(**opt.best_params_)

# ss = {
#       'model': Categorical([RandomForestRegressor()]),
#       'max_depth': range(4, 6)
# }

# rf = RandomForestRegressor()
# gs = GridSearchCV(estimator=rf, param_grid=ss, scoring=mean_squared_error)
# gs.fit(X, y)

# rf.fit(X, y)
# print(mean_squared_error(rf.predict(X), y))
