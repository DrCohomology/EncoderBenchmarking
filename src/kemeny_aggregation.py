"""
Goal of this script is to study what happens with Kemeny rank aggregation, and in particular get an idea for how
different the optimal consensuses can be

Results:
    The open source library I could find uses differential_evolution from scipy. super slow.
    Results are from a continuous optimization problem which returns a vector, which is then interpreted as a vector
        of scores and passed to the ranking.
        Hornik is way more efficient and solves the very same problem.

"""

import cvxpy as cp
import gurobipy as gp
import matplotlib.pyplot as plt
import mip
import numpy as np
import os
import pandas as pd
import ranky as rk
import scikit_posthocs as sp
import seaborn as sns
import string
import time
import warnings

from collections import defaultdict
from importlib import reload
from itertools import permutations, product
from math import factorial
from scipy import stats
from tqdm import tqdm

import src.utils as u
import src.rank_utils as ru
reload(u)
reload(ru)
np.random.seed(1444)

#%% Optimal ranking comparison. Fixed dataset random seeds.

np.random.seed(1444)

repetitions = 10
seeds = np.random.randint(0, 100000, size=repetitions)
nint = [3, 6]
nrint = [3, 20]

ties = False

run = True
if run:
    ac = []
    for na in tqdm(np.arange(nint[0], nint[1] + 1)):
        for nv in tqdm(np.arange(nrint[0], nrint[1] + 1)):
            V = pd.Series(list(string.ascii_uppercase[:na]), name="object")
            dr = pd.DataFrame([np.random.choice(np.arange(na), na, replace=ties) for _ in range(nv)], columns=V).T

            ct = pd.DataFrame(columns=np.append(["n", "nr"], seeds), index=V)
            for seed in seeds:
                temp = rk.rank(rk.center(dr, method="kendall", verbose=False, seed=seed), reverse=True)
                ct[str(seed)] = temp
            ct["n"] = na
            ct["nr"] = nv

            ac.append(ct.reset_index())

    kyc = pd.concat(ac, ignore_index=True)

a = pd.melt(kyc, id_vars=["object", "n", "nr"])
a.value = a.value.astype("int")


g = sns.catplot(
    data=a,
    x="object", y="value",
    row="n", col="nr",
    kind="boxen"
)
g.set_titles(template="{row_name} items, {col_name} voters")

plt.show()

#%% Big experiment: how many optima are there?

def do_stuff(na, nv, ties, repetitions):
    np.random.seed(np.random.randint(0, 1000))
    consensus_kind = "total_order"

    V = pd.Series(list(string.ascii_uppercase[:na]), name="object")
    strategies = ["exhaustive", "random_gp_seed"]

    # --- How many times can we get a good coverage of the optima with random seeds?
    everything = []
    for _ in range(repetitions):

        dr = pd.DataFrame([np.random.choice(np.arange(na), na, replace=ties) for _ in range(nv)], columns=V).T
        ms = ru.dr2mat(dr, kind="preference")
        c = np.sum(ms, axis=0)  # c matrix in the paper, with diagonal 0

        optima = {strat: [] for strat in strategies}
        obj_values = {strat: [] for strat in strategies}
        runtime = {strat: [] for strat in strategies}

        # --- Exhaustive search
        time_start = time.time()
        kds = defaultdict(lambda: [])
        for sigma in list(permutations(range(na))):
            kds[ru.d_kendall_set(sigma, dr)].append(pd.Series(sigma, index=V))
        optima["exhaustive"] = kds[min(kds)]
        obj_values["exhaustive"] = [min(kds) for _ in range(len(optima["exhaustive"]))]
        runtime["exhaustive"].append(time.time() - time_start)

        # --- random_gp_seed

        for seed in np.random.randint(0, 100, nv):
            time_start = time.time()
            np.random.seed(seed)
            model = gp.Model()
            model.setParam("OutputFlag", 0)
            model.setParam("Seed", seed)

            median = model.addMVar(shape=(na, na), vtype=gp.GRB.BINARY)

            model.setObjective(gp.quicksum(gp.quicksum(median * c)), gp.GRB.MAXIMIZE)
            for cs in ru.get_constraints(median, consensus_kind):
                model.addConstr(cs)

            model.update()
            model.optimize()

            temp = ru.mat2rf(median.X, alternatives=dr.index, kind="incidence")
            optima["random_gp_seed"].append(temp)
            obj_values["random_gp_seed"].append(ru.d_kendall_set(temp, dr))
            runtime["random_gp_seed"].append(time.time() - time_start)

        distinct_optima = []
        for strategy, opts in optima.items():
            if not len(opts):
                continue
            temp = [pd.Series(opt, index=V) for opt in set(tuple(opt) for opt in opts)]
            distinct_optima.append(pd.DataFrame(temp, index=[f"{strategy}_{i}" for i in range(len(temp))]).T)
        distinct_optima = pd.concat(distinct_optima, axis=1)
        everything.append(distinct_optima)

    return everything

# nv = 20
na = 5
repetitions = 20
ties = False

run = False
if run:
    ees = []
    for nv in tqdm(range(1, 23)):
        ees.append(do_stuff(na, nv, ties, repetitions))

noes = []
nors = []
for ee in ees:
    noe = []
    nor = []
    for do in ee:
        noe.append(len([col for col in do.columns if "exhaustive" in col]))
        nor.append(len([col for col in do.columns if "random_gp_seed" in col]))
    noes.append(np.array(noe))
    nors.append(np.array(nor))

fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
fig.suptitle(f"{repetitions} repetitions, {na} alternatives.")

ax = axes[0]
sns.boxplot(noes[1:], ax=ax)
ax.set_title(f"Number of optima")
ax.set_xlabel("number of voters")
ax.set_ylabel("number of optima")
ax.set_xticklabels(range(2, 23))
ax.axhline(1, ls=":", color="black")

ax = axes[1]
sns.boxplot([nor/noe for nor, noe in zip(nors[1:], noes[1:])], ax=ax)
ax.set_title(f"Fraction of retrieved optima")
ax.set_xlabel("number of voters")
ax.set_ylabel("fraction of optima")
ax.set_xticklabels(range(2, 23))

plt.tight_layout()
plt.show()

#%% Comparison of different methods to explore all of the optima

reload(ru)

na = 7
nv = 20
ties = False
consensus_kind = "total_order"

np.random.seed(np.random.randint(0, 1000))
V = pd.Series(list(string.ascii_uppercase[:na]), name="object")
dr = pd.DataFrame([np.random.choice(np.arange(na), na, replace=ties) for _ in range(nv)], columns=V).T

seeds = range(nv)
strategies = ["exhaustive", "seed", "voters"]
optima, obj, runtime = {}, {}, {}
for s in strategies:
    optima[s], obj[s], runtime[s] = ru.kemeny_optima(dr, s, consensus_kind=consensus_kind, seeds=seeds)

for s in strategies:
    print("-"*4 + " " + s + " " + "-"*4)
    print(f"Number of optima                : {optima[s].shape[1]}")
    print(f"Max objective - min objective   : {max(obj[s])-min(obj[s])}")
    print(f"Runtime                         : {runtime[s]:.02s}")

#%% Repeated comparisons of methods to find optima

repetitions = 20
seeds = range(nv)
strategies = ["exhaustive", "seed", "voters"]
rep_optima, rep_obj, rep_runtime = [], [], []
for _ in tqdm(range(repetitions)):
    # reset the seed
    np.random.seed(np.random.randint(0, 1000))
    V = pd.Series(list(string.ascii_uppercase[:na]), name="object")
    dr = pd.DataFrame([np.random.choice(np.arange(na), na, replace=ties) for _ in range(nv)], columns=V).T

    optima, obj, runtime = {}, {}, {}
    for s in strategies:
        optima[s], obj[s], runtime[s] = ru.kemeny_optima(dr, s, consensus_kind=consensus_kind, seeds=seeds)

    rep_optima.append(optima)
    rep_obj.append(obj)
    rep_runtime.append(runtime)

rep_numopt = []
for optima in rep_optima:
    rep_numopt.append({s: opt.shape[1] for s, opt in optima.items()})
df_numopt = pd.DataFrame(rep_numopt)
df_fracopt = df_numopt.divide(df_numopt.exhaustive, axis=0)

rep_maxmin = []
for obj in rep_obj:
    rep_maxmin.append({s: max(ob)-min(ob) for s, ob in obj.items()})
df_maxmin = pd.DataFrame(rep_maxmin)

df_runtime = pd.DataFrame(rep_runtime)

#%% Plot
fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=100, sharex="all")
fig.suptitle(f"{repetitions} repetitions, {na} alternatives, {nv} voters")

axes = axes.flatten()

ax = axes[0]
ax.set_title("Number of optima found")
ax.set_xlabel("strategy")
ax.set_ylabel("number of optima")
sns.boxplot(df_numopt, ax=ax)

ax = axes[1]
ax.set_title("Fraction of optima found")
ax.set_xlabel("strategy")
ax.set_ylabel("fraction of optima")
sns.boxplot(df_fracopt, ax=ax)

ax = axes[2]
ax.set_title("Runtime")
ax.set_xlabel("strategy")
ax.set_ylabel("runtime")
sns.boxplot(df_runtime, ax=ax)


plt.show()

#%% Number of optima vs number of voters

na = 5
minnv = 2
maxnv = 26
seeds = range(nv)
strategies = ["exhaustive", "voters"]
rep_optima, rep_obj, rep_runtime = [], [], []
for _ in tqdm(range(minnv, maxnv)):
    # reset the seed
    np.random.seed(np.random.randint(0, 1000))
    V = pd.Series(list(string.ascii_uppercase[:na]), name="object")
    dr = pd.DataFrame([np.random.choice(np.arange(na), na, replace=ties) for _ in range(nv)], columns=V).T

    optima, obj, runtime = {}, {}, {}
    for s in strategies:
        optima[s], obj[s], runtime[s] = ru.kemeny_optima(dr, s, consensus_kind=consensus_kind, seeds=seeds)

    rep_optima.append(optima)
    rep_obj.append(obj)
    rep_runtime.append(runtime)
rep_numopt = []
for optima in rep_optima:
    rep_numopt.append({s: opt.shape[1] for s, opt in optima.items()})
df_numopt = pd.DataFrame(rep_numopt, index=range(minnv, maxnv))
df_fracopt = df_numopt.divide(df_numopt.exhaustive, axis=0)

rep_maxmin = []
for obj in rep_obj:
    rep_maxmin.append({s: max(ob)-min(ob) for s, ob in obj.items()})
df_maxmin = pd.DataFrame(rep_maxmin, index=range(minnv, maxnv))

df_runtime = pd.DataFrame(rep_runtime, index=range(minnv, maxnv))

#%% Plot

fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100, sharex="all")
fig.suptitle(f"{repetitions} repetitions, {na} alternatives")

sns.barplot(df_numopt, ax=ax)
ax.set_xlabel("number of voters")
ax.set_ylabel("number of optima")


plt.show()


#%% Study on real data, main8_final.csv
"""
Remarks: 
    - ties are possible even with the average score 
    - missing values from missing runs
    
The aggregation procedure is slow (~70 seconds per aggregation)
The rankings are uninformative: for one of them I got only 3 classes, most are zeros
"""
reload(ru)

try:
    df
except NameError:
    df = pd.read_csv(os.path.join(u.RESULT_FOLDER, "main8_final.csv")).drop_duplicates()

folder = os.path.join(os.path.dirname(u.RESULT_FOLDER), "main8_aggregation_optima")

run = False
if run:
    cs, drs, objs, runtime = {}, {}, {}, {}
    for scoring, model in tqdm(list(product(df.scoring.unique(), df.model.unique()))):
        df2 = df.query("model == @model and scoring == @scoring")

        dr = pd.DataFrame({
            dataset: ru.score2rf(df2.query("dataset == @dataset").groupby("encoder").mean().cv_score,
                                 ascending=False)
            for dataset in df2.dataset.unique()
        })
        dr.columns = range(dr.shape[1])

        # --- One optimum
        # start_time = time.time()
        # c = ru.kemeny_aggregation_gurobi_ties(dr)
        # r = time.time() - start_time

        # cs[scoring, model] = c
        # runtime[(scoring, model)] = r

        # --- Voters optima
        optima, obj, r = ru.kemeny_optima_ties_voters(dr, verbose=True)

        # --- Store results
        drs[(scoring, model)] = dr
        cs[(scoring, model)] = optima
        objs[(scoring, model)] = obj
        runtime[(scoring, model)] = r

        dr.to_csv(os.path.join(folder, f"dr_{scoring}_{model}.csv"))
        optima.to_csv(os.path.join(folder, f"optima_{scoring}_{model}.csv"))
        pd.Series(obj).to_csv(os.path.join(folder, f"objective_values_{scoring}_{model}.csv"))

save = False
if save:
    for k, dr, c in zip(drs.keys(), drs.values(), cs.values()):
        dr.to_csv(os.path.join(folder, f"rankings_{k}.csv"))
        c.to_csv(os.path.join(folder, f"consensus_{k}.csv"))


#%% Analysis

folder = os.path.join(os.path.dirname(u.RESULT_FOLDER), "main8_aggregation_optima")

model = "KNeighborsClassifier"
scoring = "roc_auc_score"

drd = pd.read_csv(os.path.join(folder, f"dr_{scoring}_{model}.csv"))
od = pd.read_csv(os.path.join(folder, f"objective_values_{scoring}_{model}.csv"))
optd = pd.read_csv(os.path.join(folder, f"optima_{scoring}_{model}.csv"))



#%% Nemenyi aggregation is not transitive
reload(ru)

alpha = 0.05

pk = ["dataset", "model", "scoring", "fold", "encoder"]
df = pd.read_csv(os.path.join(u.RESULT_FOLDER, "main8_final.csv")).drop_duplicates()

# !!! duplicates run are still there: instead, average on pks
df = df.groupby(pk).mean().reset_index()

ranking_nemenyi = {}
for model, scoring in product(df.model.unique(), df.scoring.unique()):
    df2 = df.query("model == @model and scoring == @scoring")
    dr = pd.DataFrame({
        dataset: ru.score2rf(df2.query("dataset == @dataset").groupby("encoder").mean().cv_score,
                             ascending=False)
        for dataset in df2.dataset.unique()
    })
    dr.columns = range(dr.shape[1])

    # average ranks
    ar = dr.mean(axis=1)

    # friedman & nemenyi. Not sure what to do with ties.
    if stats.friedmanchisquare(*[dr.loc[encoder] for encoder in dr.index])[1] >= alpha:
        ranking_nemenyi[(model, scoring)] = "Friedman could not be rejected"
        continue

    significant_difference = (sp.posthoc_nemenyi_friedman(dr.T.to_numpy()) < alpha).to_numpy().astype(int)
    strict_incidence = ru.rf2mat(ru.score2rf(ar, ascending=True), kind="incidence")
    significant_strict_incidence = strict_incidence * significant_difference
    try:
        ranking_nemenyi[(model, scoring)] = ru.mat2rf(significant_strict_incidence, alternatives=dr.index)
    except:
        ranking_nemenyi[(model, scoring)] = "Matrix is not transitive"
#%%

alpha = 0.20
repetitions = 1000

na = 10
nv = 2
ties = True

rrr = []
rmat = []
failed_friedman = 0
failed_nemenyi = []
ddd = []
alphas = []
for _ in tqdm(range(repetitions)):

    np.random.seed(np.random.randint(0, 10000))
    V = pd.Series(list(string.ascii_uppercase[:na]), name="object")
    dr = pd.DataFrame([np.random.choice(np.arange(na), na, replace=ties) for _ in range(nv)], columns=V).T

    ddd.append(dr)

    # average ranks
    ar = dr.mean(axis=1)

    a = stats.friedmanchisquare(*[dr.loc[o] for o in dr.index])[1]
    alphas.append(a)
    # friedman & nemenyi. Not sure what to do with ties.
    if a >= alpha:
        failed_friedman += 1
        continue

    significant_difference = (sp.posthoc_nemenyi_friedman(dr.T.to_numpy()) < alpha).to_numpy().astype(int)
    strict_incidence = ru.rf2mat(ru.score2rf(ar, ascending=True), kind="incidence")
    significant_strict_incidence = strict_incidence * significant_difference
    try:
        rrr.append(ru.mat2rf(significant_strict_incidence, alternatives=dr.index))
        rmat.append(significant_strict_incidence)
    except:
        failed_nemenyi.append(dr)

print(f"Failed Friedman {failed_friedman}")
print(f"Intransitive Nemenyi: {len(failed_nemenyi)}")
print(f"min alpha: {min(alphas)}")
print(f"maximum possible comparisons: {max(m.sum().sum() for m in rmat)}")


