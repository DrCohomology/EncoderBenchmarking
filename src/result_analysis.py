"""
17.03.23
Result analysis, structured as in the paper
"""

import contextlib
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
sns.set_theme(style="whitegrid", font_scale=1)
sns.despine(trim=True, left=True)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "times new roman",
})

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
ste = [e.MEstimate(m=m) for m in [1e-1, 1, 10]]
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

exp_completed = set(df.groupby(pk_exp).groups)

missing = pd.DataFrame(list(exp_total - exp_completed), columns=pk_exp).sort_values(pk_exp).reset_index(drop=True)
"""
There is an extra run for f1 score
"""
# a = df.groupby(["encoder", "dataset", "model", "tuning"]).size()
# print(f"Extra f1 run for \n {a.loc[a!=15]}")

# --- Analysis of missing evaluations
missing_fraction = {}
for col in missing.columns:
    miss = missing.groupby(col).size()
    total = df.groupby(col).size() / 5 + miss  # remove folds
    missing_fraction[col] = pd.concat([miss, total], axis=1).fillna(0).sort_values(0).astype(int)

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
    cv_score.to_csv(Path(rankings_folder, "average_cv_score.csv"))
    cv_score_std.to_csv(Path(rankings_folder, "std_cv_score.csv"))

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

# %% !!! 2c. Factor sensitivity
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

factors = ["model"]
similarities = ["rho"]
# for factor in factors:
#     df_sim = u.join_wide2long({"taub": taub, "ptaub": ptaub, "agrbest": agrbest, "agrworst": agrworst, "rho": rho},
#                               comparison_level=factor)
#
# for similarity in similarities:
#     u.plot_long_format_similarity_dataframe(df_sim, similarity, comparison_level=factor, plot_type="heatmap",
#                                             color="black",
#                                             save_plot=False, show_plot=True, draw_points=False,
#                                             figsize_inches=(1.75, 2.5))

similarity = "rho"
cmap = "rocket"

similarity_ = similarity
similarity = u.SIMILARITY_LATEX[similarity]

fig, axes = plt.subplots(1, 3, figsize=(5.5, 3.5), gridspec_kw={'width_ratios': [1, 1, 1]})
ypos = -0.6

# ---- model
ax = axes[0]
comparison_level = "model"
df_sim = u.join_wide2long({"taub": taub, "ptaub": ptaub, "agrbest": agrbest, "agrworst": agrworst, "rho": rho},
                          comparison_level=comparison_level)
df_sim = df_sim.rename(columns=u.SIMILARITY_LATEX)
cl = [f"{comparison_level}_1", f"{comparison_level}_2"]
median_similarity = df_sim[cl + [similarity]].groupby(cl).median().reset_index() \
    .pivot(index=cl[0], columns=cl[1]) \
    .droplevel([0], axis=1)

ax = sns.heatmap(median_similarity, annot=True, ax=ax,
                 vmin=-1 if similarity_ in {"taub", "rho"} else 0,
                 vmax=1,
                 cmap=cmap,
                 square=True, cbar=False, annot_kws={"fontsize": 8})
ax.set(xlabel=None, ylabel=None)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
    horizontalalignment='right',
    fontweight='light',
    fontsize=10
)
ax.set_yticklabels([])
ax.set_title("(a)", y=ypos)

# ---- tuning
ax = axes[1]

comparison_level = "tuning"
df_sim = u.join_wide2long({"taub": taub, "ptaub": ptaub, "agrbest": agrbest, "agrworst": agrworst, "rho": rho},
                          comparison_level=comparison_level)
df_sim = df_sim.rename(columns=u.SIMILARITY_LATEX)
cl = [f"{comparison_level}_1", f"{comparison_level}_2"]
median_similarity = df_sim[cl + [similarity]].groupby(cl).median().reset_index() \
    .pivot(index=cl[0], columns=cl[1]) \
    .droplevel([0], axis=1)

ax = sns.heatmap(median_similarity, annot=True, ax=ax,
                 vmin=-1 if similarity_ in {"taub", "rho"} else 0,
                 vmax=1,
                 cmap=cmap,
                 square=True, cbar=False)
ax.set(xlabel=None, ylabel=None)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
    horizontalalignment='right',
    fontweight='light',
    fontsize=10
)
ax.set_yticklabels([])
ax.set_title("(b)", y=ypos)

# ---- scoring
ax = axes[2]

comparison_level = "scoring"
df_sim = u.join_wide2long({"taub": taub, "ptaub": ptaub, "agrbest": agrbest, "agrworst": agrworst, "rho": rho},
                          comparison_level=comparison_level)
df_sim = df_sim.rename(columns=u.SIMILARITY_LATEX)
cl = [f"{comparison_level}_1", f"{comparison_level}_2"]
median_similarity = df_sim[cl + [similarity]].groupby(cl).median().reset_index() \
    .pivot(index=cl[0], columns=cl[1]) \
    .droplevel([0], axis=1)

ax = sns.heatmap(median_similarity, annot=True, ax=ax,
                 vmin=-1 if similarity_ in {"taub", "rho"} else 0,
                 vmax=1,
                 cmap=cmap,
                 square=True, cbar=False, cbar_kws={"shrink": .45})
ax.set(xlabel=None, ylabel=None)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
    horizontalalignment='right',
    fontweight='light',
    fontsize=10
)
ax.set_yticklabels([])
ax.set_title("(c)", y=ypos)

# ---- colorbar
# ax = axes[3]
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
# sm.set_array([])
#
# plt.colorbar(sm, cax=ax, shrink=0.4, fraction=5)

plt.tight_layout(pad=0.5, rect=[0, 0.2, 1, 1])
plt.show()


# %% 2c2. Factor sensitivity multisimilarity

reload(u)
reload(ru)

sims = u.load_similarity_dataframes()
taub = sims["pw_kendall_tau_b_p_nan=omit.csv"]
ptaub = sims['pw_kendall_tau_b_nan=omit.csv']
rho = sims["pw_rho.csv"]
agrworst = sims["pw_agrworst.csv"]
agrbest = sims["pw_agrbest.csv"]

factors = ["model", "tuning", "scoring"]
similarities = ["rho", "agrbest"]

fig, axes = plt.subplots(1, len(factors), figsize=(5.5, 5.5/len(factors)))
for (ax, factor) in zip(axes, factors):
    df_sim = u.join_wide2long({"taub": taub, "ptaub": ptaub, "agrbest": agrbest, "agrworst": agrworst, "rho": rho},
                              comparison_level=factor)
    u.heatmap_longformat_multisim(df_sim, similarities, factor, fontsize=8, annot_fontsize=8,
                                  save_plot=False, show_plot=False, ax=ax)

plt.savefig(u.FIGURES_DIR / "heatmap_allfactors_['rho', 'agrbest'].pdf")
plt.show()


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

# %% 3d. Aggregation sensitivity plots - two measures in the same heatmap

reload(u)

aggsims = u.load_agg_similarities()
agg_taub = aggsims["pw_AGG_kendall_tau_b_nan=omit.csv"]
agg_ptaub = aggsims["pw_AGG_kendall_tau_b_p_nan=omit.csv"]
agg_rho = aggsims["pw_AGG_spearman_rho_nan=omit.csv"]
agg_agrbest = aggsims["pw_AGG_agrbest.csv"]
agg_agrworst = aggsims["pw_AGG_agrworst.csv"]

df_sim = u.join_wide2long({"taub": agg_taub, "ptaub": agg_ptaub, "rho": agg_rho, "agrbest": agg_agrbest,
                           "agrworst": agg_agrworst}, comparison_level="interpretation")

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
u.heatmap_longformat_multisim(df_sim, ["rho", "agrbest"], "interpretation", fontsize=8, annot_fontsize=8,
                              save_plot=True, show_plot=True, ax=ax, summary_statistic="mean")

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
    seed = len(sample_df_sim)
    sample_sizes = [50]
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
# sample_df_sim.to_csv(u.RANKINGS_DIR / f"sample_sim_{tuning}.csv", index=False)

# %% 4a1. Sensitivity on number of datasets - Plots
reload(u)

# sample_df_sim = u.load_sample_similarity_dataframe(tuning="model")

u.lineplot_longformat_sample_sim(sample_df_sim, similarity="rho", save_plot=False, show_plot=True,
                                 hue="model",
                                 # errorbar=lambda x: (x.mean()-x.std()/10, x.mean()+x.std()/10),
                                 estimator="mean",
                                 )


# %% !!! 5. Most sensitive datasets - datasets with high performance difference (OUTDATED - NOT IN FULL PAPER)
"""
High-variability datasets are those for which the number of times the iqr (on e, fixed d, m, s) of encoder performances 
    is greater than the median (on datasets) of such iqrs + the iqr of iqrs, and this has to happen at least 
    the median + iqr times. 
So: 
    1) fix dataset, model, tuning, scoring: compute iqr of encoder performance
    2) for every dataset: count the number of mst combinations for which the corresponding iqr is in the top 25%
    3) keep the datasets for which such count is in the top 25% 
    
!!! NO LGBM YET
"""

rankings_folder = Path(u.RESULTS_DIR).parent / "Rankings"
df = pd.read_csv(Path(u.RESULTS_DIR, "final.csv"))
rf = pd.read_csv(rankings_folder / "rank_function_from_average_cv_score.csv", index_col=0, header=[0, 1, 2, 3])

factors = ["dataset", "model", "tuning", "scoring"]

b = df.groupby(factors).cv_score.agg(iqr).dropna()
b = b.loc[b >= b.median() + iqr(b)].reset_index()
b = b.groupby("dataset").size().sort_values()
b = b.loc[b >= b.median()]  # 90 is half the combinations of fold-model-tuning-scoring (NO LGBM YET)

# high-variability datasets
rf_hv = rf[b.index.astype(str)]

folder = Path(u.RESULTS_DIR).parent / "Full results"

sns.set_theme(style="whitegrid", font_scale=0.3)

melt_rf_hv = rf_hv.melt(ignore_index=False).reset_index()
melt_rf_hv.columns = ["encoder", "dataset", "model", "tuning", "scoring", "rank"]

grid = sns.FacetGrid(melt_rf_hv, col="scoring", row="model",
                     margin_titles=True, sharey=False)

grid.set_titles(row_template="{row_name}", col_template="{col_name}")

grid.map_dataframe(sorted_boxplot, x="rank", y="encoder",
                   palette="crest", showfliers=False, linewidth=0.2, showcaps=False,
                   medianprops=dict(color="red", linewidth=0.4))
# grid.set_xticklabels(rotation=90)

grid.despine(top=True, trim=True)

grid.fig.set_size_inches(7.25, 10)
grid.fig.tight_layout()
grid.savefig(folder / f"encoder_rank_boxplot_matrix_HV.svg", dpi=600)

# plt.show()

print("Done")
# %% 6a. Rank of encoders in grid

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
grid.savefig(folder / f"encoder_rank_boxplot_matrix.pdf", dpi=600)

# plt.show()

print("Done")

# %% 6b. Plot overall ranks

reload(u)

rf = u.load_rf()
# rf = rf.loc(axis=1)[:, "DTC", "no", "AUC"]
# rf = rf / rf.max(axis=0)

rf_melt = rf.melt(ignore_index=False).reset_index()
rf_melt.columns = ["encoder", "dataset", "model", "tuning", "scoring", "rank"]
rf_melt.encoder = rf_melt.encoder.map(u.ENCODER_LATEX)

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3))
ax = u.sorted_boxplot_vertical(data=rf_melt, y="rank", x="encoder",
                               palette=sns.light_palette("grey", n_colors=len(rf.index)), showfliers=False,
                               linewidth=1, showcaps=False, medianprops=dict(color="red", linewidth=1),
                               ax=ax)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
    horizontalalignment='right',
    fontweight='light',
    fontsize=10
)
ax.set(xlabel=None)

plt.tight_layout(pad=0.5)
plt.savefig(u.FIGURES_DIR / "boxplot_rank_all_experiments.pdf", dpi=600)
# plt.show()

# %% 6b1. Rank of encoders for fixed model

reload(u)

rf = u.load_rf()

model = "LR"
rf = rf.loc(axis=1)[:, model, :, :]

rf_melt = rf.melt(ignore_index=False).reset_index()
rf_melt.columns = ["encoder", "dataset", "model", "tuning", "scoring", "rank"]
rf_melt.encoder = rf_melt.encoder.map(u.ENCODER_LATEX)

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3))
ax = u.sorted_boxplot_vertical(data=rf_melt, y="rank", x="encoder",
                               palette=sns.light_palette("grey", n_colors=len(rf.index)), showfliers=False,
                               linewidth=1, showcaps=False, medianprops=dict(color="red", linewidth=1),
                               ax=ax)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
    horizontalalignment='right',
    fontweight='light',
    fontsize=10
)
ax.set(xlabel=None)

plt.tight_layout(pad=0.5)
plt.savefig(u.FIGURES_DIR / f"boxplot_rank_{model}.pdf", dpi=600)
plt.show()


# %% 6c. Significance of rank difference

df, rf = u.load_df_rf()
a = ru.BaseAggregator(df=df, rf=rf)
# a.aggregate(strategies=["nemenyi rank"])

baseline = "OHE"

test = (posthoc_nemenyi_friedman(a.rf.T.reset_index(drop=True)) < 0.05).astype(int).to_numpy()
test = pd.DataFrame(test, index=rf.index, columns=rf.index)
print(test.loc[baseline].sort_values())

# %% 6c1. Significance of rank difference for one model

df, rf = u.load_df_rf()
rf = rf.loc(axis=1)[:, "LR", :, :]

a = ru.BaseAggregator(df=df, rf=rf)
# a.aggregate(strategies=["nemenyi rank"])

baseline = "OHE"

test = (posthoc_nemenyi_friedman(a.rf.T.reset_index(drop=True)) < 0.05).astype(int).to_numpy()
test = pd.DataFrame(test, index=rf.index, columns=rf.index)
print(test.loc[baseline].sort_values())



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


PARGENT = {
    "datasets_int": [42178, 981, 4135, 1590, 1114, 41162, 42738, 41224],
    "datasets_str": [str(x) for x in [42178, 981, 4135, 1590, 1114, 41162, 42738, 41224]],
    "models": ["LGBMC", "SVC", "KNC"],
    "tunings": ["no"],
    "scorings": ["AUC"],
    "aggregations": ['kemeny rank']
}

df, rf = u.load_df_rf()

df = df.query("dataset in @PARGENT['datasets_int'] "
              "and model in @PARGENT['models'] "
              "and tuning in @PARGENT['tunings']"
              "and scoring in @PARGENT['scorings']")
rf = rf.loc(axis=1)[PARGENT["datasets_str"], PARGENT["models"], PARGENT["tunings"], PARGENT["scorings"]]

a = ru.Aggregator(df, rf)
a.aggregate(strategies=PARGENT["aggregations"], verbose=True)








