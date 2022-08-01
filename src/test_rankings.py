# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 13:05:06 2022

@author: federicom

Test different tanking strategies

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

from collections import defaultdict
from itertools import product, combinations
from scipy.stats import t
from tqdm import tqdm

os.chdir(u.RESULT_FOLDER)

experiment_name = "benchmark_motivation"

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

df["cv_fold"] = list(range(5)) * int(len(df)/5)

df.encoder = df.encoder.replace("CollapseEncoder()", "DropEncoder()").apply(
    u.get_acronym, underscore=False)
df.scaler = df.scaler.replace("CollapseEncoder()", "DropScaler()").apply(
    u.get_acronym, underscore=False)

# %%


def t_test(v1, v2, alpha=0.05, corrected=True):
    """
    Test whether one of v1 or v2 is statistically greater than the other. 
    Assume v1 and v2 are results from a cross validation
    Add Bengio correction term (2003)
    """

    n = len(v1)

    diff = v1-v2
    avg = diff.mean()
    std = diff.std()

    # test training ratio
    ttr = 1 / (n-1)

    adjstd = np.sqrt(1 / n + ttr) * std if corrected else np.sqrt(1 / n) * std
    tstat = avg / adjstd

    df = n-1
    crit = t.ppf(1.0-alpha, df)
    p = (1.0-t.cdf(np.abs(tstat), df)) * 2.0
    return tstat, df, crit, p


def compare_with_ttest(v1, v2, alpha=0.05, corrected=True):
    """
    returns 0 if the two are equal
    returns 1 if v1 is "greater than" v2
    returns 2 if v2 is "greater than" v1

    """

    if (v1 == v2).all():
        return 0, 1

    tstat, df, crit, p = t_test(v1, v2, alpha=alpha, corrected=corrected)

    if p >= alpha:
        return 0, p
    else:
        if (v1-v2).mean() > 0:
            return 1, p
        else:
            return 2, p


def average_ranking(l: list):

    pass

manually_saved_results = {}
# %% Strategy 1: Pargent

"""
For every dataset, get a poset of encoders with corrected one-sided 0.05 t-test
They have no resample

Here: get poset with corrected ttest on resample and cv 
Then, get average ranking relation
"""

# run only once
try:
    manually_saved_results
except:
    manually_saved_results = {}
# alphas = [1, 0.1, 0.05, 0.01]

alpha = 0.01
corrected = False

e2i = {E: i for (i, E) in enumerate(df.encoder.unique())}
i2e = {v: k for (k, v) in e2i.items()}

es2i = {ES: i for (i, ES) in enumerate(
    product(df.encoder.unique(), df.scaler.unique()))}
i2es = {v: k for (k, v) in es2i.items()}

# For every dataset, get a weak ranking
rankings = []
for dataset in df.dataset.unique():
    if dataset in ['kdd_internet_usage', 'nomao', 'nursery']:
        continue

    c1 = df.dataset == dataset
    c4 = df.scoring == "accuracy_score"
    df_ = df[c1 & c4]

    nR = len(es2i)

    R = np.zeros((nR, nR))
    for (E1, S1), i1 in es2i.items():
        for (E2, S2), i2 in es2i.items():
            if i1 > i2:
                continue

            cv1 = df_[(df_.encoder == E1) & (
                df_.scaler == S1)].cv_scores.to_numpy()
            cv2 = df_[(df_.encoder == E2) & (
                df_.scaler == S2)].cv_scores.to_numpy()

            # if len(cv1) == 0:
            #     if E1 != 'DE' or S1 != 'DS':
            #         print(dataset, E1, S1)
            #     continue
            # elif len(cv2) == 0:
            #     print(E1, S1, E2, S2)
            #     continue

            if len(cv1)*len(cv2) == 0:
                continue

            comp, p = compare_with_ttest(cv1, cv2, alpha=alpha, corrected=corrected)

            # E1 and E2 not comparable
            if comp == 0:
                R[i1, i2] = 1
                R[i2, i1] = 1
            # E1 > E2
            if comp == 1:
                R[i1, i2] = 1
            # E2 < E1
            elif comp == 2:
                R[i2, i1] = 1

    rankings.append(R)
# # %% Strength of ranking

# # strategy = "sum"
# # strategy = "and"
# strategy = "or"
# # strategy = "mean"

# if strategy == "sum":
#     R = sum(rankings)
# if strategy == "and":
#     R = np.ones_like(rankings[0])
#     for r in rankings:
#         R = np.logical_and(R, r)
#     R = R.astype(int)
# elif strategy == "or":
#     R = np.zeros_like(rankings[0])
#     for r in rankings:
#         R = np.logical_or(R, r)
#     R = R.astype(int)

# elif strategy == "mean":
#     R = sum(rankings) / len(rankings)


se2i = {(S, E): i for ((E, S), i) in es2i.items()}
# # ses = [(S, E) for (E, S) in es2i.keys()]

# pdR = pd.DataFrame(R, columns=se2i.keys(), index=se2i.keys())

# pdR.drop(index=('DS', 'DE'), inplace=True)
# pdR.drop(columns=('DS', 'DE'), inplace=True)

# pdcat = pdR.loc['DS', 'DS']
# pdfull = pdR.loc['RS', 'RS']

# # pdR.index = pdR.index.to_flat_index()
# # pdcat = pdR.loc['DS']

# # print(pdR)

# """
# I risultati così ottenuti sono completamente diversi da quelli di Pargent. 
# Il design sperimentale è leggermente diverso nella fase finale, nella quale noi
# prendiamo il poset di encoder e poi troviamo un altro poset. 
# Altra cosa non chiara è il modo in cui Pargent ottiene il ranking finale se
# quello cha aveva finora è un poset. 
# I risultati summando i vari poset ottenuti per ogni dataset indicano che
# l'unico encoder non dominato è OHE. Il che è alquanto strano. 
# """
# # %% get rank from matrix
"""
Important remark: the ranking can only get us non-dominated encoders
"""

ranks = []
for ir, M in enumerate(rankings):
    pdR = pd.DataFrame(M, columns=se2i.keys(), index=se2i.keys())

    pdR.drop(index=('DS', 'DE'), inplace=True)
    pdR.drop(columns=('DS', 'DE'), inplace=True)

    # test hypothesis that everything is pairwise comparable
    if (pdR + pdR.T == 0).sum().sum() > 0:
        raise Exception(
            "This is not a weak order. Two elements are not comparable.")

    M = pdR.to_numpy()

    # for all intents and purposes, r[i] is max - the sum of row i
    r = len(M)*np.ones(len(M))
    for i in range(len(M)):
        r[i] -= M[i].sum()

    r2i = {
        rn: ir for ir, rn in enumerate(set(r))
    }
    rk = pd.DataFrame(r, index=pdR.index, columns=["rank"]).reset_index()
    rk["rank"] = rk["rank"].map(r2i)
    ranks.append(rk)

# rank = reduce(lambda df1, df2: df1.join(df2), ranks)
final_rank = pd.concat(ranks).reset_index(drop=True).rename(
    columns={"level_0": "scaler", "level_1": "encoder"})

# get order
temp = final_rank.groupby(["encoder", "scaler"])\
                 .agg([np.median, np.mean])\
                 .sort_values([('rank', 'median'), ('rank', 'mean')], ascending=[True, True])\
                 .reset_index()
temp = temp[temp.scaler == "RS"]
order = temp.encoder

manually_saved_results[alpha] = temp

#%% Repeat what above and manually save the results in a dictionary. Then plot together
# ---- Plotting


sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 4, figsize=(16, 3), dpi=1000, sharey=True)

for ia, (ax, (alpha, temp)) in enumerate(zip(axes, manually_saved_results.items())):
    
    bp = sns.boxplot(x="encoder", y="rank", hue="scaler",
                     data=final_rank, order=order, ax=ax)
    if ia > 0:
        bp.legend_.remove()
    
    medians = temp[("rank", "median")]
    for xtick in bp.get_xticks():
        bp.text(xtick + 0.2, -0.5, medians[xtick], 
                horizontalalignment='center', size='xx-small', color='black', weight="bold")
    
    ax.set_xticklabels(temp.encoder[bp.get_xticks()], rotation=90)
    ax.set_title(f"Significance {alpha}")
    # ax.legend("")
