# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 09:34:14 2022

@author: federicom
"""

import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import src.utils as u
from tqdm import tqdm

os.chdir(u.RESULT_FOLDER)

experiment_name = "benchmark_motivation_5"

results_folder = u.RESULT_FOLDER + "/" + experiment_name
results_concated = results_folder + "\\_concated.csv"
results_files = glob.glob(results_folder + "/*.csv")

if results_concated not in results_files:
    print("Concatenating datasets")
    df = pd.DataFrame()
    for file in tqdm(results_files):
        temp = pd.read_csv(file)
        if len(temp) == 0:
            print(f"Empty dataframe: {file}")
            continue
        # num = file.split("_")[-1].split(".csv")[0]
        # temp["dataset"] = temp["dataset"] + [f"_{num}"] * len(temp)

        df = df.append(temp, ignore_index=True)
    df = df.drop(columns="Unnamed: 0", errors="ignore")

    df.to_csv(results_concated, index=False)
else:
    df = pd.read_csv(results_concated)

#%% Create baselines

pk = ["dataset", "resample", "encoder", "scaler", "model", "scoring"]
scorecols = ["cv_scores"]  

df_ = df[pk+scorecols].copy()

cc = df.scaler == "CollapseEncoder()" 
nc = df.encoder == "CollapseEncoder()"

# Standard baselines
dfmax = df_.groupby(pk).max().reset_index()
dfcat = df_.loc[cc & ~nc]
dfnum = df_.loc[~cc & nc]
dfnon = df_.loc[cc & nc]
dfall = df_.loc[~cc & ~nc]

dfmax.loc[:, "kind"] = "best"
dfcat.loc[:, "kind"] = "categorical_only"
dfnum.loc[:, "kind"] = "numerical_only"
# dfnon.loc[:, "kind"] = "empty_dataset"
dfall.loc[:, "kind"] = "full_dataset"

df_ = pd.concat([dfcat, dfnum, dfnon, dfall])

# Change the name for later
dfnum.rename(columns={"test_score": "baseline_numerical_only"}, inplace=True)
# dfnon.rename(columns={"test_score": "baseline_empty_dataset"}, inplace=True)
dfmax.rename(columns={"test_score": "baseline_best"}, inplace=True)

#%%




#%% OLD STUFF

# Due to repeated experiments, some entries are duplicated
# Restore unicity
keep = ["dataset", "resample_num", "model", "scoring", "encoder",
        "scaler", "test_score"]
df_ = df[keep].groupby(
    ["dataset", "resample_num", "model", "scoring", "encoder", "scaler"]
).mean().reset_index()

dfmax = df[keep].groupby(
    ["dataset", "resample", "model", "scoring"]
).max().reset_index()
dfmax.loc[:, "kind"] = "best"
dfmax.loc[:, "encoder"] = "best"
dfmax.loc[:, "scaler"] = "best"
# tt = df_[["tuning_time"]]
# df_ = df_.drop(columns="tuning_time")

cc = df_.scaler == "CollapseEncoder" 
nc = df_.encoder == "CollapseEncoder"

# Standard baselines
dfcat = df_.loc[cc & ~nc]
dfnum = df_.loc[~cc & nc]
dfnon = df_.loc[cc & nc]
dfall = df_.loc[~cc & ~nc]

dfcat.loc[:, "kind"] = "categorical_only"
dfnum.loc[:, "kind"] = "numerical_only"
dfnon.loc[:, "kind"] = "empty_dataset"
dfall.loc[:, "kind"] = "full_dataset"

df_ = pd.concat([dfcat, dfnum, dfnon, dfall])

# Change the name for later
dfnum.rename(columns={"test_score": "baseline_numerical_only"}, inplace=True)
dfnon.rename(columns={"test_score": "baseline_empty_dataset"}, inplace=True)
dfmax.rename(columns={"test_score": "baseline_best"}, inplace=True)

# Encoders of interest - baselines with full dataset 
encoders_of_interest = [
    "WOEEncoder",
    "SmoothedTE",
    "TargetEncoder", 
    "GLMMEncoder"
]
baselines = {}
for enc_name in encoders_of_interest:
    if enc_name not in df_.encoder.unique():
        print(f"{enc_name} is not a valid encoder.")
        break
    baselines[enc_name] = df_.loc[(df_.encoder == enc_name) & ~cc].rename(
        columns={"test_score": f"baseline_{enc_name}"})

baselines["numerical_only"] = dfnum
baselines["empty_dataset"] = dfnon
baselines["best"] = dfmax

# Add the baselines
for bk, baseline in baselines.items():
    df_ = df_.merge(
        baseline.drop(columns=["encoder", "scaler", "kind"]),
        left_on=["dataset", "resample_num", "model", "scoring"],
        right_on=["dataset", "resample_num", "model", "scoring"],
        how="left"
    )

# df_ = df_.join(tt)


# %% Show baseline

model = "CatBoostClassifier"

bk = None
bk = "numerical_only"
# bk = "SmoothedTE"
# bk = "TargetEncoder"
# bk = "CollapseEncoder"
# bk = "GLMMEncoder"
bk = "best"

df2 = df_.copy().loc[df_.model == model]
if bk is not None:
    df2.test_score = df2.test_score - df2[f"baseline_{bk}"]

df2 = df2.loc[df2.kind != "categorical_only"]

sns.set(rc={"figure.dpi": 200, "figure.figsize": (20, 10)})
sns.set_style("whitegrid")
fg = sns.catplot(
    data=df2,
    kind="box",
    x="cv_scores",
    y="encoder",
    hue="kind",
    hue_order=["empty_dataset", 
               # "categorical_only",
               # "numerical_only", 
               "full_dataset"],
    col="scoring",
    col_order=["balanced_accuracy_score", "cohen_kappa", "accuracy_score"],
    orient="h",
    sharex=False,
)
fg.fig.suptitle(f"Model: {model}\nBaseline: {bk}")
fg.fig.subplots_adjust(top=0.8)

for ax in fg.axes[0]:
    ax.axvline(0, c='purple', alpha=0.5)

# %%
dat = "kick"
model = "DecisionTreeClassifier"
bk = "numerical_only"
bk = None
df2 = df_.copy().loc[df_.dataset == dat].loc[df.model == model]

if bk is not None:
    df2.test_score = df2.test_score - df2[f"baseline_{bk}"]
sns.set(rc={"figure.dpi": 200, "figure.figsize": (20, 10)})
sns.set_style("whitegrid")
fg = sns.catplot(
    data=df2,
    kind="box",
    x="test_score",
    y="encoder",
    hue="kind",
    hue_order=["empty_dataset", "categorical_only",
               "numerical_only", "full_dataset"],
    col="scoring",
    col_order=["balanced_accuracy_score", "cohen_kappa", "accuracy_score"],
    orient="h",
    sharex=False,
)

fg.fig.suptitle(f"Dataset: {dat}\nModel: {model}\nBaseline: {bk}")
fg.fig.subplots_adjust(top=0.8)
for ia, ax in enumerate(fg.axes[0]):
    zero = None if bk is None else 0
    try:
        ax.axvline(zero, c='purple', alpha=0.5)
    except:
        pass
