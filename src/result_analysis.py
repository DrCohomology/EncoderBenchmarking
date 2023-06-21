"""
17.03.23
Result analysis, structured as in the paper
"""

import contextlib
import json
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from collections import defaultdict
from functools import reduce
from importlib import reload
from itertools import product
from pathlib import Path
from scipy.stats import kendalltau, iqr
from scikit_posthocs import posthoc_nemenyi_friedman
from tqdm import tqdm
from typing import List, Union

import src.encoders as e
import src.utils as u
import src.results_concatenator as rc
import src.rank_utils as ru

# Setup plotting with LaTeX
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='Times New Roman')

# %% 0a. Define all experiments
rlibs = None
std = [e.BinaryEncoder(), e.CatBoostEncoder(), e.CountEncoder(), e.DropEncoder(), e.MinHashEncoder(), e.OneHotEncoder(),
       e.OrdinalEncoder(), e.RGLMMEncoder(rlibs=rlibs), e.SumEncoder(), e.TargetEncoder(), e.WOEEncoder()]
cvglmm = [e.CVRegularized(e.RGLMMEncoder(rlibs=rlibs), n_splits=ns) for ns in [2, 5, 10]]
cvte = [e.CVRegularized(e.TargetEncoder(), n_splits=ns) for ns in [2, 5, 10]]
buglmm = [e.CVBlowUp(e.RGLMMEncoder(rlibs=rlibs), n_splits=ns) for ns in [2, 5, 10]]
bute = [e.CVBlowUp(e.TargetEncoder(), n_splits=ns) for ns in [2, 5, 10]]
dte = [e.Discretized(e.TargetEncoder(), how="minmaxbins", n_bins=nb) for nb in [2, 5, 10]]
binte = [e.PreBinned(e.TargetEncoder(), thr=thr) for thr in [1e-3, 1e-2, 1e-1]]
ste = [e.MeanEstimateEncoder(m=m) for m in [1e-1, 1, 10]]
encoders = reduce(lambda x, y: x + y, [std, cvglmm, cvte, buglmm, bute, dte, binte, ste])
encoders = set(u.get_acronym(str(x), underscore=False) for x in encoders)

scorings = ["ACC", "AUC", "F1"]

datasets_small_keys = set(u.DATASETS_SMALL.keys())
datasets_small = set(u.DATASETS_SMALL.values())
datasets_all = set(u.DATASETS.values())

models_fulltuning = ["DTC", "SVC", "KNC", "LR"]
models_modeltuning = ["DTC", "KNC", "LR"]
models_notuning = ["DTC", "SVC", "KNC", "LR", "LGBMC"]

exp_fulltuning = set(product(encoders, datasets_small, models_fulltuning, ["full"], scorings))
exp_modeltuning = set(product(encoders, datasets_all, models_modeltuning, ["model"], scorings))
exp_notuning = set(product(encoders, datasets_all, models_notuning, ["no"], scorings))

exp_total = exp_fulltuning.union(exp_modeltuning).union(exp_notuning)

# %% 0b. Loading and cleaning

mains = [pd.read_csv(Path(f"{u.RESULTS_DIR}/main{x}_final.csv")) for x in [6, 8, 9]]

# --- There are extra runs for full tuning, remove them
mains[0] = mains[0].drop(mains[0][(mains[0].model != "LGBMClassifier") &
                                  (~ mains[0].dataset.isin(set(u.DATASETS_SMALL.keys())))].index)

df = pd.concat(mains, ignore_index=True).drop(columns="scaler")
df["time"] = df["time"].fillna(df["tuning_time"])

pk = ["encoder", "dataset", "fold", "model", "tuning", "scoring"]
pk_exp = ["encoder", "dataset", "model", "tuning", "scoring"]

# --- Mappings to collapse non-unique evaluations, to keep the best values achieved
mappings = {
    "cv_score": "max",
    "tuning_score": "max",
    "time": "min",
    "model__max_depth": "min",
    "model__n_neighbors": "min",
    "model__n_estimators": "min",
    "model__C": "min",
    "model__gamma": "min"
}
df = df.groupby(pk).aggregate(mappings).reset_index()

# --- Shorter names for models, scorings, and datasets
df.model = df.model.map(lambda x: u.get_acronym(x, underscore=False))
df.scoring = df.scoring.map({"accuracy_score": "ACC",
                             "roc_auc_score": "AUC",
                             "f1_score": "F1"})
df.dataset = df.dataset.map(lambda x: u.DATASETS[x])

# --- Move LGBM from full tuning to no tuning
df.loc[df.model == "LGBMC", "tuning"] = "no"

df.to_csv(Path(u.RESULTS_DIR, "final.csv"), index=False)

# %% 1a. Missing evaluations analysis

pk = ["encoder", "dataset", "fold", "model", "tuning", "scoring"]
pk_exp = ["encoder", "dataset", "model", "tuning", "scoring"]

exp_completed = set(df.groupby(pk_exp).groups)

missing = pd.DataFrame(list(exp_total - exp_completed), columns=pk_exp).sort_values(pk_exp).reset_index(drop=True)

# --- Analysis of missing evaluations
missing_fraction = {}
for col in missing.columns:
    miss = missing.groupby(col).size()
    total = df.groupby(col).size() / 5 + miss  # remove folds
    missing_fraction[col] = pd.concat([miss, total], axis=1).fillna(0).sort_values(0).astype(int)

# %% 1a2. Open the logs

errors = []
for log in tqdm(glob.glob(f"{u.RESULTS_DIR}/**/**/*.json", recursive=True)):
    if Path(log).name.startswith("1"):
        continue
    with open(log, "r") as fr:
        errors.append(json.load(fr))
errors = pd.DataFrame(errors)

# %% 1a3. Analyze the error logs
errors = errors.loc[~ errors.duplicated()]





# %% 1b. Different rankings -> Different evaluations

df, rf = u.load_df_rf()

model = "LR"
scoring = "AUC"
tuning = "no"

df = df.query("model == @model and scoring == @scoring and tuning == @tuning")
rf = rf.loc(axis=1)[:, model, tuning, scoring]
rf = rf / rf.max()

rdiff = rf.max() - rf.min()
qdiff = df.groupby(["dataset", "model", "tuning", "scoring", "encoder"]).cv_score.mean() \
    .groupby(["dataset", "model", "tuning", "scoring"]).agg(["max", "min"])
qdiff = qdiff["max"] - qdiff["min"]
qdiff.index = qdiff.index.set_levels([idx.astype(str) for idx in qdiff.index.levels])

dd = pd.concat([rdiff, qdiff], axis=1)

plt.scatter(dd[0], dd[1])
plt.xlabel("rank difference")
plt.ylabel("performance difference")
plt.show()

# weird examples with 1 in variation
# df.query("dataset == 43922 and model == 'SVC' and tuning == 'full' and scoring == 'F1'").groupby("encoder").cv_score.mean()
 pass

# %% 2a. Store rank functions

pk_noencoder = ["dataset", "model", "tuning", "scoring"]

run = False
if run:
    rfd = {}  # cross-validated ranks
    cv_scored = {}  # cross-validated scores
    cv_score_stdd = {}  # cross-validated scores
    for dataset, model, tuning, scoring in tqdm(list(df.groupby(pk_noencoder).groups)):
        score = df.query("dataset == @dataset "
                         "and model == @model "
                         "and tuning == @tuning "
                         "and scoring == @scoring").groupby("encoder").cv_score.agg(["mean", "std"])
        cv_scored[(dataset, model, tuning, scoring)] = score["mean"]
        cv_score_stdd[(dataset, model, tuning, scoring)] = score["std"]
        rfd[(dataset, model, tuning, scoring)] = ru.score2rf(score["mean"], ascending=False)
    rf = pd.DataFrame(rfd)
    cv_score = pd.DataFrame(cv_scored)
    cv_score_std = pd.DataFrame(cv_score_stdd)

    rf.columns.name = ("dataset", "model", "tuning", "scoring")
    rf.to_csv(u.RANKINGS_DIR / "rank_function_from_average_cv_score.csv")
    cv_score.to_csv(Path(u.RANKINGS_DIR, "average_cv_score.csv"))
    cv_score_std.to_csv(Path(u.RANKINGS_DIR, "std_cv_score.csv"))

# %% 2b. Compute and store correlation metrics
"""
Just made way more efficient by removing unnecessary comparisons. Now ~2 mins.
rf.columns.levels ~= [dataset, model, tuning, scoring]
"""

rf = u.load_rf()

run = False
if run:
    # sensitivity to change in model (everything else fixed)
    taub_model, agrbest_model, agrworst_model, rho_model = u.pairwise_similarity_wide_format(rf,
                                                                                             simfuncs=[ru.kendall_tau,
                                                                                                       ru.agreement_best,
                                                                                                       ru.agreement_worst,
                                                                                                       ru.spearman_rho],
                                                                                             shared_levels=[0, 2, 3])

    # sensitivity to change in tuning (everything else fixed)
    taub_tuning, agrbest_tuning, agrworst_tuning, rho_tuning = u.pairwise_similarity_wide_format(rf,
                                                                                                 simfuncs=[
                                                                                                     ru.kendall_tau,
                                                                                                     ru.agreement_best,
                                                                                                     ru.agreement_worst,
                                                                                                     ru.spearman_rho],
                                                                                                 shared_levels=[0, 1,
                                                                                                                3])

    # sensitivity to change in scoring (everything else fixed)
    taub_scoring, agrbest_scoring, agrworst_scoring, rho_scoring = u.pairwise_similarity_wide_format(rf,
                                                                                                     simfuncs=[
                                                                                                         ru.kendall_tau,
                                                                                                         ru.agreement_best,
                                                                                                         ru.agreement_worst,
                                                                                                         ru.spearman_rho],
                                                                                                     shared_levels=[0,
                                                                                                                    1,
                                                                                                                    2])

    taub = reduce(lambda x, y: x.fillna(y), [taub_model, taub_scoring, taub_tuning])
    agrbest = reduce(lambda x, y: x.fillna(y), [agrbest_model, agrbest_scoring, agrbest_tuning])
    agrworst = reduce(lambda x, y: x.fillna(y), [agrworst_model, agrworst_scoring, agrworst_tuning])
    rho = reduce(lambda x, y: x.fillna(y), [rho_model, rho_tuning, rho_scoring])

    taub.to_csv(u.RANKINGS_DIR / "pw_kendall_tau_b_nan=omit.csv")
    agrbest.to_csv(u.RANKINGS_DIR / "pw_agrbest.csv")
    agrworst.to_csv(u.RANKINGS_DIR / "pw_agrworst.csv")
    rho.to_csv(u.RANKINGS_DIR / "pw_rho.csv")

# %% 2b2. Taub p-value

rf = u.load_rf()

ptaub_model = u.pairwise_similarity_wide_format(rf,
                                                simfuncs=[ru.kendall_tau_p],
                                                shared_levels=[0, 2, 3])[0]

# sensitivity to change in tuning (everything else fixed)
ptaub_tuning = u.pairwise_similarity_wide_format(rf,
                                                 simfuncs=[ru.kendall_tau_p],
                                                 shared_levels=[0, 1, 3])[0]

# sensitivity to change in scoring (everything else fixed)
ptaub_scoring = u.pairwise_similarity_wide_format(rf,
                                                  simfuncs=[ru.kendall_tau_p],
                                                  shared_levels=[0, 1, 2])[0]

ptaub = reduce(lambda x, y: x.fillna(y), [ptaub_model, ptaub_scoring, ptaub_tuning])
ptaub.to_csv(u.RANKINGS_DIR / "pw_kendall_tau_b_p_nan=omit.csv")

# %% 2c. Plot factor sensitivity - factors together
"""
Model sensitivity is the change in ranking when everything is fixed but the model
Approach 1: for every DTS, compute average (on M) correlation (excluding self-correlation and nans). 
    Note that agrbest and agrworst do not have nans (not even when the rankings are degenerate). 
    The plots are then boxplots of the correlation metrics, in which we control for other variables
        (for instance, we are able to see if one dataset has consistent rankings of encoders for different models)
        This second level of grouping SHOULD be equivalent to just considering both factors at once.    
    Use plot_type='heatmap' option

Approach 2: for every DTS, compute the correlation matrix (corr(Ri, Rj)_ij, where Ri and Rj are rankings 
    for models i and j resp. Then, aggregate over DTS, i.e., get the 3-tensor or rank correlation (Model1, Model2, DTS)
    and boxplot over DTS. This will give a matrix of boxplots      
    Use plot_type='boxplots' option
"""


reload(u)
reload(ru)

sims = u.load_similarity_dataframes()
taub = sims["pw_kendall_tau_b_p_nan=omit.csv"]
ptaub = sims['pw_kendall_tau_b_nan=omit.csv']
rho = sims["pw_rho.csv"]
agrworst = sims["pw_agrworst.csv"]
agrbest = sims["pw_agrbest.csv"]

model = "KNC"

rho = rho.loc(axis=0)[:, model, :, :].loc(axis=1)[:, model, :, :].droplevel("model", axis=0).droplevel("model", axis=1)
agrbest = agrbest.loc(axis=0)[:, model, :, :].loc(axis=1)[:, model, :, :].droplevel("model", axis=0).droplevel("model", axis=1)


factors = ["tuning", "scoring"]
similarities = ["rho", "agrbest"]

# sns.set(font_scale=0.8)
sns.set_style("ticks")
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='Times New Roman')

fig, axes = plt.subplots(1, len(factors), figsize=(5.5, 5.5/len(factors)))
for (ax, factor) in zip(axes.flatten(), factors):
    # df_sim = u.join_wide2long({"taub": taub, "ptaub": ptaub, "agrbest": agrbest, "agrworst": agrworst, "rho": rho},
    #                           comparison_level=factor)
    df_sim = u.join_wide2long({"agrbest": agrbest, "rho": rho},
                              comparison_level=factor)
    u.heatmap_longformat_multisim(df_sim, similarities, factor, fontsize=7, annot_fontsize=7,
                                  save_plot=False, show_plot=False, ax=ax)

# plt.savefig(u.FIGURES_DIR / "heatmap_allfactors_rho_agrbest.pdf")
plt.show()

# %% 2c1. Plot factor sensitivity - FINAL

reload(u)

sims = u.load_similarity_dataframes()
taub = sims["pw_kendall_tau_b_p_nan=omit.csv"]
ptaub = sims['pw_kendall_tau_b_nan=omit.csv']
rho = sims["pw_rho.csv"]
agrworst = sims["pw_agrworst.csv"]
agrbest = sims["pw_agrbest.csv"]

factors = ["model", "tuning", "scoring"]
similarities = ["rho", "agrbest"]

sns.set(font_scale=0.8)
sns.set_style("ticks")
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='Times New Roman')

factors = ["scoring"]

for factor in factors:
    df_sim = u.join_wide2long({"taub": taub, "ptaub": ptaub, "agrbest": agrbest, "agrworst": agrworst, "rho": rho},
                              comparison_level=factor)
    title = None
    if factor == "model":
        # figsize = (1.8, 2)
        # title = "(b) ML model"
        figsize = (1.8, 1.8)
    elif factor == "tuning":
        # figsize = (2.3, 1.2)
        # title = "(c) Tuning strategy"
        # tx = -1.5
        # ty = 0.35
        # adjust_left = 0.6
        # adjust_right = 1
        figsize = (1.2, 1.2)

    elif factor == "scoring":
        # figsize = (2.0, 1.1)
        # title = "(d) Scoring"
        # tx = -1.5
        # ty = 0.35
        # adjust_left = 0.7
        # adjust_right = 1
        figsize = (1.1, 1.1)

    else:
        raise AssertionError("Issue")

    u.heatmap_longformat_multisim(df_sim, similarities, factor, fontsize=8, annot_fontsize=8,
                                  figsize=figsize,
                                  save_plot=True, show_plot=True, title=title,)
                                  # tx=tx, ty=ty, adjust_left=adjust_left, adjust_right=adjust_right)

print("Showtime")
# %% 3a. Interpretation sensitivity - Get the aggregation functions (ru.Aggregator cannot support missing evaluations)
df, rf = u.load_df_rf()

# 45 minutes for all strategies (Kemeny is the bottleneck)
"""
!!! Gurobi is failing because 'model is too large'. It did not do this before though.
"""
run = False
if run:
    a = ru.Aggregator(df, rf)
    a.aggregate(verbose=True, ignore_strategies=["nemenyi rank", "kemeny rank"])
    a.to_csv(u.RANKINGS_DIR / "tmp" / "aggregated_ranks_from_average_cv_score_no_nemenyi_no_kemeny.csv")

# --- Kemeny aggregation is the bottleneck aggregation
run_kemeny = False
if run_kemeny:
    a = ru.Aggregator(df, rf)
    a.aggregate(verbose=True, strategies=["kemeny rank"])
    a.to_csv(u.RANKINGS_DIR / "tmp" / "aggregated_ranks_from_average_cv_score_kemeny.csv")

# --- Test multiple alphas, then concatenate everything together
run_nemenyi = False
if run_nemenyi:
    for alpha in [0.01, 0.05, 0.1]:
        a = ru.Aggregator(df, rf)
        a.aggregate(verbose=True, strategies=["nemenyi rank"], alpha=alpha)
        a.to_csv(u.RANKINGS_DIR / "tmp" / f"aggregate_ranks_from_average_cv_score_nemenyi_{alpha}.csv")

concatenate = False
if concatenate:
    aggrf = pd.read_csv(u.RANKINGS_DIR / "tmp" / "aggregated_ranks_from_average_cv_score_no_nemenyi_no_kemeny.csv", index_col=0,
                        header=[0, 1, 2, 3])
    agg_nemenyi = [aggrf]
    for alpha in [0.01, 0.05, 0.1]:
        agg_nemenyi.append(
            pd.read_csv(u.RANKINGS_DIR / "tmp" / f"aggregate_ranks_from_average_cv_score_nemenyi_{alpha}.csv",
                        index_col=0, header=[0, 1, 2, 3]))
    agg_kemeny = [
        pd.read_csv(u.RANKINGS_DIR / "tmp" / f"aggregated_ranks_from_average_cv_score_kemeny.csv",
                    index_col=0, header=[0, 1, 2, 3])
    ]

    aggrf = pd.concat(agg_nemenyi + agg_kemeny, axis=1)
    aggrf.to_csv(u.RANKINGS_DIR / "aggregated_ranks_from_average_cv_score.csv")

# %% 3b. Get and store correlation between aggregated rankings
aggrf = u.load_aggrf().rename(columns=u.AGGREGATION_NAMES, level="interpretation")
agg_taub, agg_ptaub, agg_rho, agg_agrbest, agg_agrworst = u.pairwise_similarity_wide_format(aggrf,
                                                                                            simfuncs=[ru.kendall_tau,
                                                                                                      ru.kendall_tau_p,
                                                                                                      ru.spearman_rho,
                                                                                                      ru.agreement_best,
                                                                                                      ru.agreement_worst],
                                                                                            shared_levels=slice(-1))

agg_taub.to_csv(u.RANKINGS_DIR / "pw_AGG_kendall_tau_b_nan=omit.csv")
agg_ptaub.to_csv(u.RANKINGS_DIR / "pw_AGG_kendall_tau_b_p_nan=omit.csv")
agg_rho.to_csv(u.RANKINGS_DIR / "pw_AGG_spearman_rho_nan=omit.csv")
agg_agrbest.to_csv(u.RANKINGS_DIR / "pw_AGG_agrbest.csv")
agg_agrworst.to_csv(u.RANKINGS_DIR / "pw_AGG_agrworst.csv")

# %% !!! 3c. Aggregation sensitivity plots - OUTDATED

reload(u)

aggsims = u.load_agg_similarities()
agg_taub = aggsims["pw_AGG_kendall_tau_b_nan=omit.csv"]
agg_ptaub = aggsims["pw_AGG_kendall_tau_b_p_nan=omit.csv"]
agg_rho = aggsims["pw_AGG_spearman_rho_nan=omit.csv"]
agg_agrbest = aggsims["pw_AGG_agrbest.csv"]
agg_agrworst = aggsims["pw_AGG_agrworst.csv"]

df_sim = u.join_wide2long({"taub": agg_taub, "ptaub": agg_ptaub, "rho": agg_rho, "agrbest": agg_agrbest,
                           "agrworst": agg_agrworst}, comparison_level="interpretation")

similarities = ["rho", "agrbest"]
for similarity in similarities:
    u.plot_long_format_similarity_dataframe(df_sim.round(1), similarity, comparison_level="interpretation",
                                            plot_type="heatmap",
                                            color="black", figsize_inches=(3, 3), fontsize=8, annot_fontsize=7,
                                            save_plot=False, show_plot=True, draw_points=False)

# %% 3d. Plot aggregation sensitivity - FINAL

reload(u)

aggsims = u.load_agg_similarities()
agg_taub = aggsims["pw_AGG_kendall_tau_b_nan=omit.csv"]
agg_ptaub = aggsims["pw_AGG_kendall_tau_b_p_nan=omit.csv"]
agg_rho = aggsims["pw_AGG_spearman_rho_nan=omit.csv"]
agg_agrbest = aggsims["pw_AGG_agrbest.csv"]
agg_agrworst = aggsims["pw_AGG_agrworst.csv"]

df_sim = u.join_wide2long({"taub": agg_taub, "ptaub": agg_ptaub, "rho": agg_rho, "agrbest": agg_agrbest,
                           "agrworst": agg_agrworst}, comparison_level="interpretation")

sns.set_style("ticks")
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='Times New Roman')

fig, ax = plt.subplots(1, 1, figsize=(3, 3.3))
u.heatmap_longformat_multisim(df_sim, ["rho", "agrbest"], "interpretation", fontsize=7, annot_fontsize=7,
                              save_plot=True, show_plot=True, ax=ax, summary_statistic="mean", title=None)

# %% 3d1. Average similarity

aggsims = u.load_agg_similarities()
agg_taub = aggsims["pw_AGG_kendall_tau_b_nan=omit.csv"]
agg_ptaub = aggsims["pw_AGG_kendall_tau_b_p_nan=omit.csv"]
agg_rho = aggsims["pw_AGG_spearman_rho_nan=omit.csv"]
agg_agrbest = aggsims["pw_AGG_agrbest.csv"]
agg_agrworst = aggsims["pw_AGG_agrworst.csv"]

sim = agg_agrbest.to_numpy()
np.fill_diagonal(sim, np.nan)
print(np.nanmean(sim))


# %% 3d2. Test for median_similarity --- removable
comparison_level="scoring"; cl = [f"{comparison_level}_1", f"{comparison_level}_2"]; similarity = "rho"
median_similarity = df_sim[cl + [similarity]].groupby(cl).median().reset_index() \
    .pivot(index=cl[0], columns=cl[1]) \
    .droplevel([0], axis=1)\
    .rename(index=u.FACTOR_LATEX[comparison_level], columns=u.FACTOR_LATEX[comparison_level])

tmp = median_similarity.to_numpy()
np.fill_diagonal(tmp, 0)
tmp = pd.DataFrame(tmp, index=median_similarity.index, columns=median_similarity.columns)



# %% 4a. Sensitivity on number of datasets - Compute
"""
For all aggregation strategies except Kemeny, which takes way too long. 
For a given sample size, draw two non-overlapping samples of datasest and compute the correlation between the 
    aggregated rankings. 
"""

df, rf = u.load_df_rf()

tuning = "no"
df_ = df.query("tuning == @tuning")
rf_ = rf.loc(axis=1)[:, :, tuning, :].copy()

sample_df_sim = u.load_sample_similarity_dataframe(tuning=tuning)
run = True
if run:
    # whenever we add experiments, start from the value of seed
    seed = 0
    sample_sizes = [5, 10, 15, 20, 25]
    repetitions = 20
    mat_corrs = []
    sample_aggregators = defaultdict(lambda: [])
    for sample_size in tqdm(sample_sizes):
        inner_mat_corrs = []
        inner_sample_aggregators = []
        for _ in tqdm(range(repetitions)):
            seed += 1
            a = ru.SampleAggregator(df_, rf_, sample_size, seed=seed, bootstrap=True).aggregate(ignore_strategies=["kemeny rank"],
                                                                                                verbose=False)

            tmp_taub, tmp_agrbest, tmp_agrworst, tmp_rho = u.pairwise_similarity_wide_format(a.aggrf,
                                                                                             simfuncs=[ru.kendall_tau,
                                                                                                       ru.agreement_best,
                                                                                                       ru.agreement_worst,
                                                                                                       ru.spearman_rho])
            agg_sample_long = u.join_wide2long(dict(zip(["taub", "agrbest", "agrworst", "rho"],
                                                        [tmp_taub, tmp_agrbest, tmp_agrworst, tmp_rho])),
                                               comparison_level="sample")

            inner_mat_corrs.append(agg_sample_long.assign(sample_size=sample_size).query("sample_1 < sample_2"))
            sample_aggregators[sample_size].append(a)
        mat_corrs.append(pd.concat(inner_mat_corrs, axis=0))
    # if mat_corr is already defined, make it bigger! We are adding experiments
    mat_corr = pd.concat(mat_corrs, axis=0)
    mat_corr = mat_corr.join(
        mat_corr.groupby(["interpretation", "sample_size"])[["taub", "agrbest", "agrworst", "rho"]].std(),
        on=["interpretation", "sample_size"], rsuffix="_std")
    sample_df_sim = pd.concat([sample_df_sim, mat_corr], axis=0)

# rename to AGGREGATION_NAMES but allow correctly formatted names
sample_df_sim.interpretation = sample_df_sim.interpretation.map(lambda x: defaultdict(lambda: x, u.AGGREGATION_NAMES)[x])
sample_df_sim.to_csv(u.RANKINGS_DIR / f"sample_sim_{tuning}.csv", index=False)

# %% 4a1. Sensitivity on number of datasets - Plots
reload(u)

sample_df_sim = u.load_sample_similarity_dataframe(tuning="no")

# sample_df_sim = sample_df_sim.query("model == 'LR'")

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
u.lineplot_longformat_sample_sim(sample_df_sim, similarity="rho", save_plot=False, show_plot=False,
                                 hue="interpretation",
                                 estimator="mean",
                                 errorbar="sd",
                                 ax=ax
                                 )
# ax.legend().remove()
plt.show()
print("Showtime")

# %% 4a2a. Sensitivity - multiple plots and custom legend - TOP

reload(u)

sns.set(font_scale=0.8)
sns.set_style("ticks")
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='Times New Roman')

fig, axes = plt.subplots(1, 3, figsize=(5.5, 2), sharey="all")
sim = "rho"
hue = "model"
handles = labels = None
for ax, tuning in zip(axes, ["no", "model", "full"]):
    sample_df_sim = u.load_sample_similarity_dataframe(tuning=tuning)

    u.lineplot_longformat_sample_sim(sample_df_sim, similarity=sim, save_plot=False, show_plot=False,
                                     hue=hue,
                                     estimator="mean",
                                     ax=ax,
                                     )
    if tuning == "no":
        handles, labels = ax.get_legend_handles_labels()

    ax.legend().remove()

plt.subplots_adjust(top=3/4)
plt.figlegend(
    handles=handles,
    labels=labels,
    bbox_to_anchor=(0, 3/4+0.02, 1, 0.2),
    loc="lower left",
    mode="expand",
    borderaxespad=1,
    ncol=5
)

sns.despine(trim=True)

# plt.savefig(u.FIGURES_DIR / f"top_sample_{sim}_{hue}.pdf", dpi=600)
plt.show()

print("Showtime")

# %% 4a2b. Sensitivity - multiple plots and custom legend - BOTTOM

reload(u)

fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.5), sharey="all")
sim = "agrbest"
hue = "model"
for ax, tuning in zip(axes, ["no", "model", "full"]):
    sample_df_sim = u.load_sample_similarity_dataframe(tuning=tuning)

    u.lineplot_longformat_sample_sim(sample_df_sim, similarity=sim, save_plot=False, show_plot=False,
                                     hue=hue,
                                     estimator="mean",
                                     ax=ax,
                                     )
    ax.legend().remove()

sns.despine(trim=True)

plt.savefig(u.FIGURES_DIR / f"bottom_sample_{sim}_{hue}.pdf", dpi=600)
plt.show()


# %% 4a2c. Sensitivity - 2 x 3 matrix

reload(u)

sns.set(font_scale=0.8)
sns.set_style("ticks")
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='Times New Roman')

fig = plt.figure(figsize=(5.5, 3))
gs = fig.add_gridspec(2, 3)

hue = "interpretation"

for isim, sim in enumerate(["rho", "agrbest"]):
    for itun, tuning in enumerate(["no", "model", "full"]):
        sample_df_sim = u.load_sample_similarity_dataframe(tuning=tuning)

        xb = (sim == "agrbest")
        yl = (tuning == "no")

        with sns.axes_style("ticks", {"xtick.bottom": True, "ytick.left": True}):
            ax = fig.add_subplot(gs[isim, itun])

            u.lineplot_longformat_sample_sim(sample_df_sim, similarity=sim, save_plot=False, show_plot=False,
                                             hue=hue,
                                             estimator="mean",
                                             ax=ax,
                                             )
        if not xb:
            ax.set_xlabel(None)
            ax.set_xticklabels([])
        if not yl:
            ax.set_ylabel(None)
            ax.set_yticklabels([])

        ax.grid(axis="y", zorder=-1, linewidth=0.4)

        if tuning == "no":
            handles, labels = ax.get_legend_handles_labels()

        ax.legend().remove()

        if sim == "rho":
            ax.set_title(f"{tuning} tuning")

    plt.tight_layout(w_pad=3, h_pad=1)

    ## hue = model
    # plt.subplots_adjust(top=0.86)
    # plt.figlegend(
    #     handles=handles,
    #     labels=labels,
    #     bbox_to_anchor=(0, 0.86+0.02, 1, 0.2),
    #     loc="lower left",
    #     mode="expand",
    #     borderaxespad=1,
    #     ncol=5,
    #     frameon=False
    # )

    ## hue = interpretation
    plt.subplots_adjust(top=0.8)
    plt.figlegend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(0, 0.8+0.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=1,
        ncol=5,
        frameon=False
    )

    # hue = scoring
    # plt.subplots_adjust(top=0.86)
    # plt.figlegend(
    #     handles=handles,
    #     labels=labels,
    #     bbox_to_anchor=(0, 0.86+0.02, 1, 0.2),
    #     loc="lower center",
    #     # mode="expand",
    #     borderaxespad=1,
    #     ncol=3,
    #     frameon=False
    # )

sns.despine(trim=True)

plt.savefig(u.FIGURES_DIR / f"sample_{hue}.pdf", dpi=600)
plt.show()

print("Showtime")

# %% 6a. Rank of encoders in grid of models

rf = u.load_rf()

folder = u.FIGURES_DIR



rf_melt = rf.melt(ignore_index=False).reset_index()
rf_melt.columns = ["encoder", "dataset", "model", "tuning", "scoring", "rank"]

grid = sns.FacetGrid(rf_melt, col="scoring", row="model",
                     margin_titles=True, sharey=False)

grid.set_titles(row_template="{row_name}", col_template="{col_name}")

grid.map_dataframe(u.sorted_boxplot_horizontal, x="rank", y="encoder",
                   palette="crest", showfliers=False, linewidth=0.2, showcaps=False,
                   medianprops=dict(color="red", linewidth=0.4))
# grid.set_xticklabels(rotation=90)

grid.despine(top=True, trim=True)

grid.fig.set_size_inches(7.25, 10)
grid.fig.tight_layout()
# grid.savefig(folder / f"encoder_rank_boxplot_matrix.pdf", dpi=600)

# plt.show()

print("Done")

# %% 6b. Rank of encoders ALL

reload(u)

rf = u.load_rf()

rf_melt = rf.melt(ignore_index=False).reset_index()
rf_melt.columns = ["encoder", "dataset", "model", "tuning", "scoring", "rank"]
rf_melt.encoder = rf_melt.encoder.map(u.ENCODER_LATEX)

sns.set(font_scale=0.8)
sns.set_style("ticks", {"ytick.left": False})
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='Times New Roman')


fig, ax = plt.subplots(1, 1, figsize=(1.8, 4.4))
ax = u.sorted_boxplot_horizontal(data=rf_melt, y="encoder", x="rank", order_by="mean",
                                 # palette=sns.light_palette("grey", n_colors=len(rf.index)),
                                 color="lightgrey",
                                 showfliers=False,
                                 linewidth=1, showcaps=False,
                                 showmeans=True,
                                 meanprops={"marker": "o",
                                            "markeredgecolor": "red",
                                            "markersize": 2},
                                 medianprops={"linestyle": "-"
                                 },
                                 ax=ax)
ax.set(xlabel=None, ylabel=None)
ax.set_xlim(0, 32)
ax.set_xticks([0, 10, 20, 30])
ax.grid(axis="x", zorder=-1, linewidth=0.4)
# ax.set_title("(c) All models")

sns.despine(left=True, trim=True)
plt.tight_layout(w_pad=0.5)
plt.savefig(u.FIGURES_DIR / f"boxplot_rank_all.pdf", dpi=600)
plt.show()

# %% 6b1. Rank of encoders fixed model

reload(u)

rf = u.load_rf()

model = "KNC"
rf = rf.loc(axis=1)[:, model, :, :]

rf_melt = rf.melt(ignore_index=False).reset_index()
rf_melt.columns = ["encoder", "dataset", "model", "tuning", "scoring", "rank"]
rf_melt.encoder = rf_melt.encoder.map(u.ENCODER_LATEX)

sns.set(font_scale=0.8)
sns.set_style("ticks", {"ytick.left": False})
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='Times New Roman')


fig, ax = plt.subplots(1, 1, figsize=(1.8, 4.4))
ax = u.sorted_boxplot_horizontal(data=rf_melt, y="encoder", x="rank", order_by="mean",
                                 # palette=sns.light_palette("grey", n_colors=len(rf.index)),
                                 color="lightgrey",
                                 showfliers=False,
                                 linewidth=1, showcaps=False,
                                 showmeans=True,
                                 meanprops={"marker": "o",
                                            "markeredgecolor": "red",
                                            "markersize": 2},
                                 medianprops={"linestyle": "-"
                                 },
                                 ax=ax)
ax.set(xlabel=None, ylabel=None)
ax.set_xlim(0, 32)
ax.set_xticks([0, 10, 20, 30])
ax.grid(axis="x", zorder=-1, linewidth=0.4)

# if model == "DTC":
#     title = "(a) Decision tree"
# elif model == "LR":
#     title = "(b) Logistic regression"
# elif model == "all":
#     title = "(c) All models"
# else:
#     raise ValueError("rip")
# ax.set_title(title)

sns.despine(left=True, trim=True)
plt.tight_layout(w_pad=0.5)

plt.savefig(u.FIGURES_DIR / f"boxplot_rank_{model}.pdf", dpi=600)
plt.show()

# %% 6b2. Rank of encoders - 1 x 3
reload(u)

rf = u.load_rf()

rf_melt = rf.melt(ignore_index=False).reset_index()
rf_melt.columns = ["encoder", "dataset", "model", "tuning", "scoring", "rank"]
rf_melt.encoder = rf_melt.encoder.map(u.ENCODER_LATEX)

sns.set(font_scale=0.8)
sns.set_style("ticks", {"ytick.left": False})
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
mpl.rc('font', family='Times New Roman')

models = ["DTC", "LR", "all"]
fig, axes = plt.subplots(1, 3, figsize=(5.5, 4.4))
for ax, model in zip(axes, models):

    rf = u.load_rf() if model == "all" else u.load_rf().loc(axis=1)[:, model, :, :]

    rf_melt = rf.melt(ignore_index=False).reset_index()
    rf_melt.columns = ["encoder", "dataset", "model", "tuning", "scoring", "rank"]
    rf_melt.encoder = rf_melt.encoder.map(u.ENCODER_LATEX)

    u.sorted_boxplot_horizontal(data=rf_melt, y="encoder", x="rank", order_by="mean",
                                color="lightgrey",
                                showfliers=False,
                                linewidth=1, showcaps=False,
                                showmeans=True,
                                meanprops={"marker": "o",
                                            "markeredgecolor": "red",
                                            "markersize": 2},
                                medianprops={"linestyle": "-"},
                                ax=ax)
    ax.set(xlabel=None, ylabel=None)
    ax.set_xlim(0, 32)
    ax.set_xticks([0, 10, 20, 30])
    ax.grid(axis="x", zorder=-1, linewidth=0.4)

    if model == "DTC":
        title = "Decision tree"
    elif model == "LR":
        title = "Logistic regression"
    elif model == "all":
        title = "All models"
    else:
        raise ValueError("rip")
    ax.set_title(title)

sns.despine(left=True, trim=True)
plt.tight_layout(w_pad=2)
plt.savefig(u.FIGURES_DIR / f"boxplot_rank.pdf", dpi=600)
plt.show()


# %% 6b2. Distribution of average ranks

rf = u.load_rf()

rf = rf / rf.max()

rf_melt = rf.melt(ignore_index=False).reset_index()
rf_melt.columns = ["encoder", "dataset", "model", "tuning", "scoring", "rank"]
rf_melt.encoder = rf_melt.encoder.map(u.ENCODER_LATEX)

sns.set_style("ticks")

fig, axes = plt.subplots(2, 3, figsize=(5.5, 3), sharex="all", sharey="all")

for ax, model in zip(axes.flatten(), ["all"] + rf_melt.model.unique().tolist()):
    if model == "all":
        tmp = rf_melt.groupby("encoder").mean()
    else:
        tmp = rf_melt.query("model == @model").groupby("encoder").mean()

    sns.histplot(data=tmp, x="rank",
                 stat="density", binwidth=0.05,
                 ax=ax)

    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1])

    ax.set_title(model)


sns.despine(trim=True)
plt.tight_layout(pad=0.5)
plt.show()

print("Showtime")

# %% 6c. Statistical tests for quality and rank difference

df, rf = u.load_df_rf()
a = ru.BaseAggregator(df=df, rf=rf)
# a.aggregate(strategies=["nemenyi rank"])

baseline = "OHE"

nemtest = (posthoc_nemenyi_friedman(a.rf.T.reset_index(drop=True)) < 0.05).astype(int).to_numpy()
nemtest = pd.DataFrame(nemtest, index=rf.index, columns=rf.index)
print(f"Nemenyi non-significant difference with {baseline}: ", set(nemtest.loc[baseline][nemtest.loc[baseline] == 0].index))

df_ = df.groupby(["encoder", "dataset", "model", "tuning", "scoring"]).cv_score.mean().reset_index()
ttest = []
for scoring in ["ACC", "AUC", "F1"]:
    df__ = df_.query("scoring == @scoring")
    for e1, e2 in product(df_.encoder.unique(), repeat=2):
        if e1 >= e2:
            continue

        df__1 = df__.query("encoder == @e1")
        df__2 = df__.query("encoder == @e2")

        df___ = pd.merge(df__1, df__2, how="inner", on=["dataset", "model", "tuning"])
        best, pval = ru.compare_with_ttest(df___.cv_score_x, df___.cv_score_y, corrected=False)
        ttest.append([scoring, e1, e2, best, pval])

ttest = pd.DataFrame(ttest, columns=["scoring", "encoder_1", "encoder_2", "best_encoder", "pval"])
for scoring in ["ACC", "AUC", "F1"]:
    ttest_ = ttest.query("scoring == @scoring and "
                         "(encoder_1 == @baseline or encoder_2 == @baseline) and "
                         "best_encoder == 0")
    # "((encoder_1 == @baseline and best_encoder == 2) or "
    # "(encoder_2 == @baseline and best_encoder == 1))")
    print(f"ttest non-significant for {scoring} and {baseline}: ",
          set(ttest_.encoder_1.unique()).union(ttest_.encoder_2.unique()))


# %% 6c1. Significance of rank difference for one model

df, rf = u.load_df_rf()

model = "KNC"
df = df.query("model == @model")
rf = rf.loc(axis=1)[:, model, :, :]

a = ru.BaseAggregator(df=df, rf=rf)
# a.aggregate(strategies=["nemenyi rank"])

baseline = "WOEE"

nemtest = (posthoc_nemenyi_friedman(a.rf.T.reset_index(drop=True)) < 0.05).astype(int).to_numpy()
nemtest = pd.DataFrame(nemtest, index=rf.index, columns=rf.index)
print(f"Nemenyi non-significant difference with {baseline}: ", set(nemtest.loc[baseline][nemtest.loc[baseline] == 0].index))

df_ = df.groupby(["encoder", "dataset", "model", "tuning", "scoring"]).cv_score.mean().reset_index()
ttest = []
for scoring in ["ACC", "AUC", "F1"]:
    df__ = df_.query("scoring == @scoring")
    for e1, e2 in product(df_.encoder.unique(), repeat=2):
        if e1 >= e2:
            continue

        df__1 = df__.query("encoder == @e1")
        df__2 = df__.query("encoder == @e2")

        df___ = pd.merge(df__1, df__2, how="inner", on=["dataset", "model", "tuning"])
        best, pval = ru.compare_with_ttest(df___.cv_score_x, df___.cv_score_y, corrected=False)
        ttest.append([scoring, e1, e2, best, pval])

ttest = pd.DataFrame(ttest, columns=["scoring", "encoder_1", "encoder_2", "best_encoder", "pval"])
for scoring in ["ACC", "AUC", "F1"]:
    ttest_ = ttest.query("scoring == @scoring and "
                         "(encoder_1 == @baseline or encoder_2 == @baseline) and "
                         "best_encoder == 0")
    # "((encoder_1 == @baseline and best_encoder == 2) or "
    # "(encoder_2 == @baseline and best_encoder == 1))")
    print(f"ttest non-significant for {scoring} and {baseline}: ",
          set(ttest_.encoder_1.unique()).union(ttest_.encoder_2.unique()))



# %% 6d. Plot overall performance

df = u.load_df()
scoring = "AUC"
df_auc = df.query("scoring == @scoring")

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
g = u.sorted_boxplot_horizontal(data=df_auc, x="cv_score", y="encoder",
                                palette=sns.light_palette("grey", n_colors=df_auc.nunique()["encoder"]),
                                showfliers=False, linewidth=1, showcaps=False,
                                medianprops=dict(color="red", linewidth=1))
plt.tight_layout()
plt.savefig(u.FIGURES_DIR / "boxplot_quality_all_experiments.pdf")
plt.show()

# %% 6e Plot tuning effect

df = u.load_df()

pk = ["encoder", "dataset", "fold", "model", "tuning", "scoring"]
pk.remove("fold")
df_auc = df.groupby(pk).cv_score.mean().reset_index()

df_full = df_auc.query("tuning == 'full'")
df_model = df_auc.query("tuning == 'model'")
df_no = df_auc.query("tuning == 'no'")

pk.remove("tuning")

df_full_no = pd.merge(df_full, df_no, on=pk, how="inner")[pk + ["cv_score_x", "cv_score_y"]]
df_full_no["full_VS_no"] = df_full_no.cv_score_x - df_full_no.cv_score_y

df_full_model = pd.merge(df_full, df_model, on=pk, how="inner")[pk + ["cv_score_x", "cv_score_y"]]
df_full_model["full_VS_model"] = df_full_model.cv_score_x - df_full_model.cv_score_y

df_model_no = pd.merge(df_model, df_no, on=pk, how="inner")[pk + ["cv_score_x", "cv_score_y"]]
df_model_no["model_VS_no"] = df_model_no.cv_score_x - df_model_no.cv_score_y


# factor = "model"
# fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharex=False, clear=True)
# for ax, (comparison, df_) in zip(axes, zip(["full_VS_no", "full_VS_model", "model_VS_no"],
#                                            [df_full_no, df_full_model, df_model_no])):
#     sns.boxplot(data=df_, x=comparison, y=factor,
#                 palette=sns.light_palette("grey", n_colors=df.nunique()[factor]),
#                 showfliers=False, linewidth=1, showcaps=False,
#                 medianprops=dict(color="red", linewidth=1),
#                 ax=ax)
#
#     # ax.set_xlim((-0.05, 0.05))
#     ax.set_xlabel(f"{scoring}_{comparison}")

factors = ["model", "scoring"]
df_model_no.rename(columns={"model_VS_no": f"gain"}, inplace=True)
fig, axes = plt.subplots(1, len(factors), figsize=(5.5, 3))
for factor, ax in zip(factors, axes):
    sns.boxplot(data=df_model_no, x=f"gain", y=factor,
                    palette=sns.light_palette("grey", n_colors=df.nunique()[factor]),
                    showfliers=False, linewidth=1, showcaps=False,
                    medianprops=dict(color="red", linewidth=1),
                    ax=ax)


    sns.despine()
plt.tight_layout(pad=0.5)
# plt.savefig(u.FIGURES_DIR / "boxplot_tuningeffect_model_no.pdf")
plt.show()

# %% 7. Discriminating power of aggregation strategies

rankings_folder = Path(u.RESULTS_DIR).parent / "Rankings"
aggrf = pd.read_csv(rankings_folder / "aggregated_ranks_from_average_cv_score.csv", index_col=0, header=[0, 1, 2, 3])

agg_power = aggrf.max(axis=0).reset_index()
agg_power.columns = ["model", "tuning", "scoring", "strategy", "power"]
agg_power.power = (1 + agg_power.power) / len(aggrf.index)

# compact names
new_names = {
    "rank_mean": "RM",
    "rank_median": "RMd",
    "rank_numbest": "RB",
    "rank_num_worst": "RW",
    "rank_numworst": "RW",
    "rank_kemeny": "RK",
    "rank_nemenyi": "RN",
    "rank_nemenyi_0.01": "RN01",
    "rank_nemenyi_0.05": "RN05",
    "rank_nemenyi_0.10": "RN10",
    "qual_mean": "QM",
    "qual_median": "QMd",
    "qual_thrbest_0.95": "QT",
    "qual_rescaled_mean": "QR"
}
strategy_palette = {
    "QM": "#ff0000",
    "QMd": "#d40909",
    "QR": "#af0d0d",
    "QT": "#8e0a0a",
    "RB": "#0096ff",
    "RM": "#0a79c7",
    "RMd": "#0e5e96",
    "RW": "#114467",
    "RN": "#00303c",
    "RN01": "#00303c",
    "RN05": "#00303c",
    "RN10": "#00303c",
    "RK": "#ef0115"
}

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "times new roman",
})

sns.set_theme(style="whitegrid", font_scale=0.5)
sns.despine(trim=True, left=True)
sns.set(font_scale=1)

agg_power.strategy = agg_power.strategy.map(new_names)


def index_sorted_reversed_by_median_then_max(df, groupby_col, target_col):
    return df.groupby(groupby_col)[target_col].agg(["median", "max"]) \
        .sort_values(by=["median", "max"], ascending=False).index


fig, ax = plt.subplots(1, 1)
sns.boxplot(agg_power,
            order=index_sorted_reversed_by_median_then_max(agg_power, groupby_col="strategy", target_col="power"),
            x="strategy", y="power", palette=strategy_palette, ax=ax)
ax.set_ylabel("fraction of tiers")

fig.tight_layout()
# fig.savefig(Path(r"C:\Users\federicom\Desktop\Retreat 2023\Poster") / "aggregation_strategy_power.svg")

plt.show()
# %% 8a. Comparison with Pargent (2022)
"""
Discrepancies: 

Datasets:
    MISSING click_prediction_small (41434): Pargent treates the id columns as categorical, but they are numerical in our version
    ID open_payments: they: 41442, we: 42178
    MISSING road-safety-drivers-sex (41447): this dataset does not exist. 
        45038 has the same name, but it has a different number of entries and there are 3 binary features out of 32
        
Models:
    MISSING Lasso regression (we have LogReg)
    MISSING RF
    REPLACE xgboost with LGBM (very similar)
    REPLACE SVM + information gain filter for features WITH SVM
    REPLACE 15NN with 5NN
    
Tunings:
    They say they use limited tuning, but they only tune Lasso and tune SVM with model (?) tuning 
    
Aggregations: 
    To get the rankings, they use: E1 > E2 if t-test rejects --- we use plain cross-validated average
        reasons for our choice: 
            1. t-tests' hypotheses will be rejected by random chance Demsar (2006) 
            2. pairwise t-tests yield non-transitive rankings (that's why we are not doing it)
    They use Kemeny - BUT: their implementation from Hornik (2007) uses the symmetric distance as distance
        it might be equivalent to other distances adapted to tied rankings
        it does not take into account partiality of the rankings (maybe they don't have this problem)
        it yields a TOTAL ORDER, i.e., they CANNOT account for ties in the consensus
    
Other: 
    High cardinality threshold
    Imputation for missing levels - we have default -1
"""


PARGENT22 = {
    "datasets_int": [42178, 981, 4135, 1590, 1114, 41162, 42738, 41224],
    "datasets_str": [str(x) for x in [42178, 981, 4135, 1590, 1114, 41162, 42738, 41224]],
    "models": ["LGBMC", "SVC", "KNC"],
    "tunings": ["no"],
    "scorings": ["AUC"],
    "aggregations": ['kemeny rank']
}

df, rf = u.load_df_rf()

df = df.query("dataset in @PARGENT22['datasets_int'] "
              "and model in @PARGENT22['models'] "
              "and tuning in @PARGENT22['tunings']"
              "and scoring in @PARGENT22['scorings']")
rf = rf.loc(axis=1)[PARGENT22["datasets_str"], PARGENT22["models"], PARGENT22["tunings"], PARGENT22["scorings"]]

a = ru.Aggregator(df, rf)
a.aggregate(strategies=PARGENT22["aggregations"], verbose=True)

# %% 8b. Comparison with Cerda (2022)
"""
Discrepancies: 

Datasets:
    only adult (curated) is in both studies. They focus more on multiclass and regression
Models:
    XGBoost -> LGBM
Scorings:
    precision -> F1
Tunings:

Aggregations: 

Other: 
"""

CERDA22 = {
    "datasets_int": [1590],
    "datasets_str": [str(x) for x in [1590]],
    "models": ["LGBMC"],
    "tunings": ["no"],
    "scorings": ["F1"],
    "aggregations": ['rescaled mean quality'],
    "encoders": ["OHE", "MHE"]
}

df, rf = u.load_df_rf()

df = df.query("dataset in @CERDA22['datasets_int'] "
              "and model in @CERDA22['models'] "
              "and tuning in @CERDA22['tunings']"
              "and scoring in @CERDA22['scorings']")
rf = rf.loc(axis=1)[CERDA22["datasets_str"], CERDA22["models"], CERDA22["tunings"], CERDA22["scorings"]]

# a = ru.Aggregator(df, rf)
# a.aggregate(strategies=CERDA22["aggregations"], verbose=True)

df_ = df.query("encoder in @CERDA22['encoders']")



# %% 8c. Comparison with Valdez-Valenzuela (2021)
"""
Discrepancies: 

Datasets:
    They say "supplementary material" but this does not exist on IEEE. 

Models:
    MISSING neural network
Scorings:
    precision -> F1
Tunings:

Aggregations: 

Other: 
"""

# 1590 adult
# 40945 titanic
# 31 credit-g
# 42178 telco

VALDEZ21 = {
    "datasets_int": [1590],
    "datasets_str": [str(x) for x in [1590]],
    "models": ["LGBMC", "LR", "SVC"],
    "tunings": ["no"],
    "scorings": ["ACC"],
}

df, rf = u.load_df_rf()

df = df.query("dataset in @VALDEZ21['datasets_int'] "
              "and model in @VALDEZ21['models'] "
              "and tuning in @VALDEZ21['tunings']"
              "and scoring in @VALDEZ21['scorings']")
rf = rf.loc(axis=1)[VALDEZ21["datasets_str"], VALDEZ21["models"], VALDEZ21["tunings"], VALDEZ21["scorings"]]

# a = ru.Aggregator(df, rf)
# a.aggregate(strategies=VALDEZ21["aggregations"], verbose=True)

df_ = df.groupby(["encoder", "dataset", "model", "tuning", "scoring"]).cv_score.mean().reset_index()

for model in VALDEZ21["models"]:
    df__ = df_.query("model == @model")
    # print(f"{model} best :", df__.loc[df__.cv_score == df__.cv_score.max()].iloc[0].encoder)
    # print(f"{model} worst:", df__.loc[df__.cv_score == df__.cv_score.min()].iloc[0].encoder)
    print(df__.sort_values("cv_score")[["model", "encoder", "cv_score"]])



# %% 8d. Comparison with Wright (2019)
"""
Discrepancies: 

Datasets:

Models:

Tunings:

Aggregations: 

Other: 
"""

WRIGHT19 = {
    "datasets_int": [50],
    "datasets_str": [str(x) for x in [50]],
    "models": ["LGBMC"],
    "tunings": ["no"],
    "scorings": ["ACC"],
    "encoders": ["TE", "DE", "OHE"]
    }

df, rf = u.load_df_rf()

df = df.query("dataset in @WRIGHT19['datasets_int'] "
              "and model in @WRIGHT19['models'] "
              "and tuning in @WRIGHT19['tunings']"
              "and scoring in @WRIGHT19['scorings']")
rf = rf.loc(axis=1)[WRIGHT19["datasets_str"], WRIGHT19["models"], WRIGHT19["tunings"], WRIGHT19["scorings"]]

for encoder in WRIGHT19["encoders"]:
    df_ = df.query("encoder == @encoder").groupby(["encoder", "model", "dataset", "scoring"]).cv_score.agg(["mean", "max"]).reset_index()
    print(df_)

# %% 8e. Comparison with Dahouda (2021)

"""
Discrepancies: 

Datasets:

Models:

Tunings:

Aggregations: 

Other: 
"""

DAHOUDA21 = {
    "datasets_int": [1461],
    "datasets_str": [str(x) for x in [1461]],
    "models": ["LR"],
    "tunings": ["no"],
    "scorings": ["ACC"],
    "encoders": ["TE", "BE", "OHE"]
}

df, rf = u.load_df_rf()

df = df.query("dataset in @DAHOUDA21['datasets_int'] "
              "and model in @DAHOUDA21['models'] "
              "and tuning in @DAHOUDA21['tunings']"
              "and scoring in @DAHOUDA21['scorings']")
rf = rf.loc(axis=1)[DAHOUDA21["datasets_str"], DAHOUDA21["models"], DAHOUDA21["tunings"], DAHOUDA21["scorings"]]

for encoder in DAHOUDA21["encoders"]:
    df_ = df.query("encoder == @encoder").groupby(["encoder", "model", "dataset", "scoring"]).cv_score.agg(["mean", "std"]).reset_index()
    print(df_)
# %% 8f. Comparison with Cerda (2018)
"""
Discrepancies: 

Datasets:
    only adult (curated) is in both studies. They focus more on multiclass and regression
Models:
    XGBoost -> LGBM
Scorings:
    precision -> F1
Tunings:

Aggregations: 

Other: 
"""

CERDA18 = {
    "datasets_int": [42738],
    "datasets_str": [str(x) for x in [42738]],
    "models": ["LGBMC"],
    "tunings": ["no"],
    "scorings": ["F1"],
    "encoders": ["OHE", "ME01E", "ME1E", "ME10E"]
}

df, rf = u.load_df_rf()

df = df.query("dataset in @CERDA18['datasets_int'] "
              "and model in @CERDA18['models'] "
              "and tuning in @CERDA18['tunings']"
              "and scoring in @CERDA18['scorings']")
rf = rf.loc(axis=1)[CERDA18["datasets_str"], CERDA18["models"], CERDA18["tunings"], CERDA18["scorings"]]

# a = ru.Aggregator(df, rf)
# a.aggregate(strategies=CERDA18["aggregations"], verbose=True)

df_ = df.query("encoder in @CERDA18['encoders']").groupby(["encoder", "dataset", "model"]).cv_score.agg(["mean", "max"]).reset_index()











