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
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "times new roman"
})

latex_name = {
    "taub": r"$\tau_b$",
    "taub_std": r"$\sigma(\tau_b)$",
    "agrbest": r"$\alpha_{best}$",
    "agrbest_std": r"$\sigma(\alpha_{best})$",
    "agrworst": r"$\alpha_{worst}$",
    "agrworst_std": r"$\sigma(\alpha_{worst})$",
}

#%% 0a. Define all experiments
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
encoders = reduce(lambda x, y: x+y, [std, cvglmm, cvte, buglmm, bute, dte, binte, ste])
encoders = set(u.get_acronym(str(x), underscore=False) for x in encoders)

scorings = ["ACC", "AUC", "F1"]

datasets_small = set(u.DATASETS_SMALL.values())
datasets_all = set(u.DATASETS.values())

models_fulltuning = ["DTC", "SVC", "KNC", "LR"]
models_modeltuning = ["DTC", "KNC", "LR"]
models_notuning = ["DTC", "SVC", "KNC", "LR", "LGBMC"]

exp_fulltuning = set(product(encoders, datasets_small, models_fulltuning, ["full"], scorings))
exp_modeltuning = set(product(encoders, datasets_all, models_modeltuning, ["model"], scorings))
exp_notuning = set(product(encoders, datasets_all, models_notuning, ["no"], scorings))

exp_total = exp_fulltuning.union(exp_modeltuning).union(exp_notuning)

#%% 0b. Loading and cleaning

mains = [pd.read_csv(Path(f"{u.RESULT_FOLDER}/main{x}_final.csv")) for x in [6, 8, 9]]

# --- There are extra runs for full tuning, remove them
mains[0] = mains[0].drop(mains[0][(mains[0].model != "LGBMClassifier") &
                                  (~ mains[0].dataset.isin(datasets_small))].index)

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

df.to_csv(Path(u.RESULT_FOLDER, "final.csv"), index=False)

#%% 1a. Missing evaluations analysis

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

#%% 1b. Overall results (EMPTY)
pass

#%% 2a. Store rank functions

pk_noencoder = ["dataset", "model", "tuning", "scoring"]

rankings_folder = Path(u.RESULT_FOLDER).parent / "Rankings"

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

    rf.to_csv(Path(rankings_folder, "rank_function_from_average_cv_score.csv"))
    cv_score.to_csv(Path(rankings_folder, "average_cv_score.csv"))
    cv_score_std.to_csv(Path(rankings_folder, "std_cv_score.csv"))

# df.query("dataset == 1235 and model == 'DTC' and tuning == 'model' and scoring == 'ACC' and encoder in ['BE', 'CBE']").groupby

# --  Make the names consistent with df: dataset -> openML.id; model -> acronym; scoring -> Kurzung
# sc_map = {"accuracy_score": "ACC", "roc_auc_score": "ROC", "f1_score": "F1"}
# rf.columns = rf.columns.map(lambda x: (u.DATASETS[x[0]], u.get_acronym(x[1], underscore=False), x[2], sc_map[x[3]]))
pass

#%% 2b. Compute and store correlation metrics
"""
Compute taub, agrbest, and agrworst for all pairs of rankings. 
It takes ~6 hrs, likely due to the dataframe implementation. A better way would be to compute a matrix and then make it 
    into a dataframe. In this case, it should take ~1 hr. Another optimization is not to consider every pair,
    but just the relevant ones
The remaining missing values are due to constant rank functions, which messes up kendall_tau
It is an obvious problem which was already known, kendall cannot work with tied rankings
However, I could not find any reference to this fact (~10 mins)
I shall just ignore them in the analysis (for now)
"""

# !!! Future work: implement my own and run with numba
# This is to be done after all experiments have been re-run as it takes a while
def kendall_taub():
    pass


rankings_folder = Path(u.RESULT_FOLDER).parent / "Rankings"

rf = pd.read_csv(rankings_folder / "rank_function_from_average_cv_score.csv",
                 index_col=0, header=[0, 1, 2, 3])

# Mtaub = np.zeros((len(rf.columns), len(rf.columns)))
# for (i1, col1), (i2, col2) in tqdm(list(product(enumerate(rf.columns), repeat=2))):
#     Mtaub[i1, i2] = kendalltau(rf[col1], rf[col2], variant="b", nan_policy="omit")[0]
#     if i1 == 10:
#         break

run = False
if run:
    # For every pair of columns, compute: taub, agreement on best, agreement on worst
    taub = pd.DataFrame(index=rf.columns, columns=rf.columns)
    agrbest = pd.DataFrame(index=rf.columns, columns=rf.columns)
    agrworst = pd.DataFrame(index=rf.columns, columns=rf.columns)
    for (i1, col1), (i2, col2) in tqdm(list(product(enumerate(rf.columns), repeat=2))):
        if i1 >= i2:
            continue
        taub.loc[col1, col2] = kendalltau(rf[col1], rf[col2], variant="b", nan_policy="omit")[0]
        agrbest.loc[col1, col2] = ru.agreement(rf[col1], rf[col2], best=True)
        agrworst.loc[col1, col2] = ru.agreement(rf[col1], rf[col2], best=False)

    # They are antisymmetric
    taub = taub.fillna(taub.T)
    agrbest = agrbest.fillna(agrbest.T)
    agrworst = agrworst.fillna(agrworst.T)

    for i in range(len(taub)):
        assert np.isnan(taub.iloc[i, i]) and np.isnan(agrbest.iloc[i, i]) and np.isnan(agrworst.iloc[i, i])

        taub.iloc[i, i] = 1
        agrbest.iloc[i, i] = 1
        agrworst.iloc[i, i] = 1

    # --  Make the names consistent with df and rf: dataset -> openML.id; model -> acronym; scoring -> Kurzung
    # sc_map = {"accuracy_score": "ACC", "roc_auc_score": "ROC", "f1_score": "F1"}
    # taub.columns = taub.index = taub.columns.map(lambda x: (str(u.DATASETS[x[0]]),
    #                                                         u.get_acronym(x[1], underscore=False),
    #                                                         x[2],
    #                                                         sc_map[x[3]]))
    # agrbest.columns = agrbest.index = agrbest.columns.map(lambda x: (str(u.DATASETS[x[0]]),
    #                                                                  u.get_acronym(x[1], underscore=False),
    #                                                                  x[2],
    #                                                                  sc_map[x[3]]))
    # agrworst.columns = agrworst.index = agrworst.columns.map(lambda x: (str(u.DATASETS[x[0]]),
    #                                                                     u.get_acronym(x[1], underscore=False),
    #                                                                     x[2],
    #                                                                     sc_map[x[3]]))

    taub.to_csv(rankings_folder / "pw_kendall_tau_b_nan=omit.csv")
    agrbest.to_csv(rankings_folder / "pw_agrbest.csv")
    agrworst.to_csv(rankings_folder / "pw_agrworst.csv")

#%% 2c1. Factor stability approach 1
"""
Model stability is the change in ranking when everything is fixed but the model
Approach 1: for every DTS, compute average (on M) correlation (excluding self-correlation and nans). 
    Note that agrbest and agrworst do not have nans (not even when the rankings are degenerate). 
    The plots are then boxplots of the correlation metrics, in which we control for other variables
        (for instance, we are able to see if one dataset has consistent rankings of encoders for different models)
        This second level of grouping SHOULD be equivalent to just considering both factors at once.    
"""


def index_sorted_by_median(df, groupby_col, target_col):
    return df.groupby(groupby_col)[target_col].median().sort_values().index


def get_avg_corr_model(df_corr):
    """
    Assumed that df_corr.columns is a MultiIndex [dataset, model, tuning, scoring].
    Ignores nans and the diagonal (ie correlation of item with itself)
    """
    t = pd.Series({idx: np.nanmean(np.tril(temp.xs(idx, level=[0, 2, 3], axis=1), -1))
                   for idx, temp in df_corr.groupby(level=[0, 2, 3])}).to_frame().reset_index()
    t.columns = ["dataset", "tuning", "scoring", "corr"]
    return t


# taub, agrbest, agrworst = load_correlation_dataframes()
#
# avg_corr = reduce(lambda l, r: l.merge(r, on=["dataset", "tuning", "scoring"], how="inner"),
#                   [get_avg_corr_model(taub), get_avg_corr_model(agrbest), get_avg_corr_model(agrworst)])
# avg_corr.columns = ["dataset", "tuning", "scoring", "taub", "agrbest", "agrworst"]
#
# fig, axes = plt.subplots(3, 4, figsize=(12, 12), sharey="col")
# fig.suptitle("Average correlation for fixed everything except model")
#
# for y, axrow in zip(["taub", "agrbest", "agrworst"], axes):
#     for x, ax in zip([None, "dataset", "tuning", "scoring"], axrow):
#         sns.boxplot(avg_corr, ax=ax, x=x, y=y,
#                     order=index_sorted_by_median(avg_corr, x, y) if x == "dataset" else None)
#         if x == "dataset":
#             ax.set_xticks([])
#
# plt.tight_layout()
# plt.show()

pass

#%% 2c2. Factor Stability approach 2 (no dataset - too many)
"""
Approach 2: for every DTS, compute the correlation matrix (corr(Ri, Rj)_ij, where Ri and Rj are rankings 
    for models i and j resp. Then, aggregate over DTS, i.e., get the 3-tensor or rank correlation (Model1, Model2, DTS)
    and boxplot over DTS. This will give a matrix of boxplots  
"""


def load_correlation_dataframes(folder):
    # load (indexed) matrices of rank correlations
    taub = pd.read_csv(folder / "pw_kendall_tau_b_nan=omit.csv", index_col=[0, 1, 2, 3], header=[0, 1, 2, 3])
    agrbest = pd.read_csv(folder / "pw_agrbest.csv", index_col=[0, 1, 2, 3], header=[0, 1, 2, 3])
    agrworst = pd.read_csv(folder / "pw_agrworst.csv", index_col=[0, 1, 2, 3], header=[0, 1, 2, 3])

    # fix: the dataset ID is not read as string
    taub.index = taub.index.set_levels(taub.index.levels[0].astype(str), level=0)
    agrbest.index = agrbest.index.set_levels(agrbest.index.levels[0].astype(str), level=0)
    agrworst.index = agrworst.index.set_levels(agrworst.index.levels[0].astype(str), level=0)

    # rename the index and column levels
    factors = ["dataset", "model", "tuning", "scoring"]

    for idx in [taub.index, taub.columns, agrbest.index, agrbest.columns, agrworst.index, agrworst.columns]:
        idx.rename(factors, inplace=True)

    return taub, agrbest, agrworst


# dataset, tuning, scoring, model1, model2, correlation
def melt_corr_matrix(df_corr, factor):
    """
    df_corr is an indexed square matrix of correlations between rankings obtained from different combinations of
        experimental factors
    Assume that df_corr.columns is a MultiIndex [dataset, model, tuning, scoring].
    Output a dataframe with schema (dataset, tuning, scoring, factor_1, factor_2, correlation)
    The output dataframe includes all pairwise comparisons
    """

    factors = ["dataset", "model", "tuning", "scoring"]
    try:
        factors.remove(factor)
    except ValueError:
        raise ValueError(f"{factor} is not a valid experimental factor")

    l = []
    for idx, temp in df_corr.groupby(level=factors):
        # cross section
        t = temp.xs(idx, level=factors, axis=1)
        # change names
        t.index = t.index.rename({factor: f"{factor}_1"})
        t.columns.name = f"{factor}_2"
        # stack: indexed matrix -> dataframe
        t = t.stack().reorder_levels(factors + [f"{factor}_1", f"{factor}_2"])
        l.append(t)

    return pd.concat(l, axis=0).rename("corr").to_frame().reset_index()


def compute_mat_corr(taub, agrbest, agrworst, factor):
    """
    Takes as input three indexed correlation matrices between combinations of experimental factors
        Their columns are MultiIndex objects [dataset, model, tuning, scoring].
    Outputs a dataframe with schema ({other_factor_1}, {other_factor_2}, {other_factor_3}, factor_1, factor_2, taub, agrbest, agrworst)
        other_factor_n is the name of another factor in [dataset, model, tuning, scoring] which is not factor
    The output dataframe includes all pairwise comparisons
    """

    factors = ["dataset", "model", "tuning", "scoring"]
    try:
        factors.remove(factor)
    except ValueError:
        raise ValueError(f"{factor} is not a valid experimental factor")
    factors.extend([f"{factor}_1", f"{factor}_2"])

    mat_corr = reduce(
        lambda l, r: l.merge(r, on=factors, how="inner"),
        [melt_corr_matrix(taub, factor), melt_corr_matrix(agrbest, factor), melt_corr_matrix(agrworst, factor)])
    mat_corr.columns = factors + ["taub", "agrbest", "agrworst"]

    return mat_corr


def plot_mat_corr(mat_corr, corr_metric, factor, color, folder, latex_name,
                  show_plot=False, draw_points=True, ylim=None):
    """
    mat_corr has schema (dataset, tuning, scoring, factor_1, factor_2, taub, agrbest, agrworst),
        each entry represents the correlation between the ranking produced with model1 and model2, with the rest fixed
    """

    sns.set_theme(style="whitegrid", font_scale=0.5)
    sns.despine(trim=True, left=True)

    # rename columns to their latex_name for pertty plotting
    mat_corr = mat_corr.rename(columns=latex_name)
    corr_metric_ = corr_metric
    corr_metric = latex_name[corr_metric]

    # IEEE column width = 3.25 inches; dpi is 300, 600 for black&white
    grid = sns.FacetGrid(mat_corr, col=f"{factor}_2", row=f"{factor}_1",
                         margin_titles=True, sharex="all", sharey="all",
                         aspect=0.4, ylim=ylim)
    grid.set_titles(row_template="{row_name}", col_template="{col_name}")

    if draw_points:
        grid.map_dataframe(sns.stripplot, y=corr_metric, color=color, size=1)
    grid.map_dataframe(sns.boxplot, y=corr_metric, color=color, boxprops=dict(alpha=0.5), fliersize=0.5)

    grid.fig.set_size_inches(3.25, 5)
    grid.fig.tight_layout()
    grid.savefig(folder / f"{factor}_stability_matrix_{corr_metric_}.svg", dpi=600)

    if show_plot:
        plt.show()


folder = Path(u.RESULT_FOLDER).parent / "Stability"

test = False
factors = ["model", "tuning", "scoring"] if not test else ["tuning"]
corr_metrics = ["taub", "agrbest", "agrworst"] if not test else ["taub"]

for factor in factors:
    taub, agrbest, agrworst = load_correlation_dataframes(Path(u.RESULT_FOLDER).parent / "Rankings")
    mat_corr = compute_mat_corr(taub, agrbest, agrworst, factor)

    for corr_metric in corr_metrics:
        plot_mat_corr(mat_corr, corr_metric, factor, "black", folder, latex_name, show_plot=test, draw_points=False,
                      ylim=(-1, 1) if corr_metric == "taub" else (0, 1))

#%% 3a. Interpretation stability - Get the aggregation functions (ru.Aggregator cannot support missing evaluations)
"""
Aggregation strategies:
    quality
        mean, median, thrbest, rescaled mean, (ttest)
    rank
        mean, median, numbest, numworst, kemeny, nemenyi
        
Note that df is supposed to be alrady filtered for a specific model, scoring, and tuning. 
!!! This can be changed in the future.        
"""


class BaseAggregator(object):
    """
    Aggregated scores (self.df) and/or rankings (self.rf) into a single ranking computed with an aggregation strategy
    Accepted aggregation strategies are they keys of self.supported_strategies
    The aggregation output is a dataframeof rankings, self.aggrf, with index the encoders and columns the aggregation
        strategy
    It is called BaseAggregator because it does not act on df nor rf, assuming they are ready to be aggregated
    """

    def __init__(self, df: pd.DataFrame, rf: pd.DataFrame):
        self.df = df
        self.rf = rf
        self.aggrf = pd.DataFrame(index=rf.index)

        self.supported_strategies = {
            "mean rank": self._aggr_rank_mean,
            "median rank": self._aggr_rank_median,
            "numbest rank": self._aggr_rank_numbest,
            "numworst rank": self._aggr_rank_numworst,
            "kemeny rank": self._aggr_rank_kemeny,
            "nemenyi rank": self._aggr_rank_nemenyi,
            "mean quality": self._aggr_qual_mean,
            "median quality": self._aggr_qual_median,
            "thrbest quality": self._aggr_qual_thrbest,
            "rescaled mean quality": self._aggr_qual_rescaled_mean,
        }
        # True if low = better (i.e., if it behaves like a rank function)
        self.ascending = {
            "rank_mean": True,
            "rank_median": True,
            "rank_numbest": False,
            "rank_numworst": True,
            "rank_kemeny": True,
            "rank_nemenyi": True,
            "qual_mean": False,
            "qual_median": False,
            "qual_thrbest": False,
            "qual_rescaled_mean": False
        }


    def _aggr_rank_mean(self, **kwargs):
        self.aggrf["rank_mean"] = self.rf.mean(axis=1)

    def _aggr_rank_median(self, **kwargs):
        self.aggrf["rank_median"] = self.rf.median(axis=1)

    def _aggr_rank_numbest(self, **kwargs):
        self.aggrf["rank_numbest"] = (self.rf == self.rf.min(axis=0)).sum(axis=1)

    def _aggr_rank_numworst(self, **kwargs):
        self.aggrf["rank_numworst"] = (self.rf == self.rf.max(axis=0)).sum(axis=1)

    def _aggr_rank_kemeny(self, **solver_params):
        self.aggrf["rank_kemeny"] = ru.kemeny_aggregation_gurobi_ties(self.rf.set_axis(range(self.rf.shape[1]), axis=1),
                                                                      **solver_params)

    def _aggr_rank_nemenyi(self, alpha=0.05):
        """
        Issues with no answer (yet):
            is the test "good" when the rankings involved have different number of tiers?
            does it support missing values?
            transitivity is not guaranteed, however, it seems to be always transitive

        Compute the outranking (domination) matrix and the matrix of significant differences according to Nemenyi pw tests,
        then multiply them together to get the significative differences matrix, and rebuild a rank function from it
        """
        self.aggrf["rank_nemenyi"] = ru.mat2rf(ru.rf2mat(ru.score2rf(self.rf.mean(axis=1)), kind="domination") *
                                               (posthoc_nemenyi_friedman(self.rf.T.reset_index(drop=True)) < alpha)
                                               .astype(int).to_numpy(),
                                               alternatives=self.rf.index)

    def _aggr_qual_mean(self, **kwargs):
        self.aggrf["qual_mean"] = self.df.groupby("encoder").cv_score.agg(np.nanmean)

    def _aggr_qual_median(self, **kwargs):
        self.aggrf["qual_median"] = self.df.groupby("encoder").cv_score.median(np.nanmedian)

    def _aggr_qual_thrbest(self, thr=0.95, **kwargs):
        """
        Count the number of datasets on which an encoder achieves quality >= thr*best
            best is the best performance on a dataset
        """
        self.aggrf["qual_thrbest"] = self.df.groupby(["dataset", "encoder"]).cv_score.mean().to_frame().reset_index()\
                                            .join(df.groupby("dataset").cv_score.max(), on="dataset", rsuffix="_best")\
                                            .query("cv_score >= @thr*cv_score_best").groupby("encoder").size()\
                                            .reindex(self.rf.index).fillna(0)

    def _aggr_qual_rescaled_mean(self, **kwargs):
        """
        iqr == 0 means that all encoders are equal on the dataset, i.e., we should ignore the comparison anyway
        """
        d1 = self.df.groupby(["dataset", "encoder"]).cv_score.mean().to_frame().reset_index()\
                 .join(self.df.groupby(["dataset"]).cv_score.agg(np.nanmin), on="dataset", rsuffix="_worst")\
                 .join(self.df.groupby(["dataset"]).cv_score.agg(iqr), on="dataset", rsuffix="_iqr")
        d1["cv_score_rescaled"] = (d1["cv_score"] - d1["cv_score_worst"]) / d1["cv_score_iqr"]
        self.aggrf["qual_rescaled_mean"] = d1.query("cv_score_iqr != 0").groupby("encoder").cv_score_rescaled.mean()

    def aggregate(self, strategies: Union[set, list, str] = "all", ignore_strategies=tuple(), **kwargs):

        if strategies == "all":
            strategies = self.supported_strategies.keys()
        for strategy in set(strategies) - set(ignore_strategies):
            self.supported_strategies[strategy](**kwargs)

        # Transform the scores into rankings
        for col in self.aggrf:
            self.aggrf[col] = ru.score2rf(self.aggrf[col], ascending=self.ascending[col])


class Aggregator(object):
    """
    Aggregator that subsets the columns of df and rf to select model, tuning, and scoring
        As default behaviour, loops over all combinations
    For each combination, it aggregates scores (self.df) and/or rankings (self.rf) into
        a ranking computed with an aggregation strategy
    Accepted aggregation strategies are they keys of self.supported_strategies
    The aggregation output is a dataframeof rankings, self.aggrf, with index the encoders and columns the aggregation
        strategy

    df is the dataframe of experimental evaluations and has schema:
        'encoder', 'dataset', 'fold', 'model', 'tuning', 'scoring', 'cv_score', 'tuning_score', 'time',
        'model__max_depth', 'model__n_neighbors', 'model__n_estimators', 'model__C', 'model__gamma'
    rf is the dataframe of rank functions (obtained from instance by getting the average cv_score and then ranking) and
        has schema:
        index = encoders
        columns = all combinations of dataset, model, tuning, scoring
    """

    def __init__(self, df: pd.DataFrame, rf: pd.DataFrame):
        self.df = df
        self.rf = rf

        self.combinations = self.df.groupby(["model", "tuning", "scoring"]).size().index
        self.base_aggregators = {(m, t, s): BaseAggregator(df.query("model == @m and tuning == @t and scoring == @s"),
                                                          rf.loc(axis=1)[:, m, t, s])
                                 for m, t, s in self.combinations}
        self.aggrf = pd.DataFrame(index=rf.index)


    def aggregate(self, strategies: Union[list, set, str] = "all", ignore_strategies: Union[tuple, list] = tuple(),
                  verbose: bool = False, **kwargs):
        comb_iter = tqdm(list(self.combinations)) if verbose else self.combinations
        for model, tuning, scoring in comb_iter:
            a = self.base_aggregators[(model, tuning, scoring)]
            a.aggregate(strategies, ignore_strategies=ignore_strategies, **kwargs)
            a.aggrf.columns = pd.MultiIndex.from_product([[model], [tuning], [scoring], a.aggrf.columns])

        self.aggrf = pd.concat([a.aggrf for a in self.base_aggregators.values()], axis=1)

        return self


rankings_folder = Path(u.RESULT_FOLDER).parent / "Rankings"
df = pd.read_csv(Path(u.RESULT_FOLDER, "final.csv"))
rf = pd.read_csv(rankings_folder / "rank_function_from_average_cv_score.csv",
                 index_col=0, header=[0, 1, 2, 3])
# 45 minutes for all strategies (Kemeny is the bottleneck)
run = False
if run:
    a = Aggregator(df, rf)
    a.aggregate(verbose=True)
    a.aggrf.to_csv(rankings_folder / "aggregated_ranks_from_average_cv_score.csv")

#%% 3b. Get and store correlation between aggregated rankings



def square_correlation_matrix(aggrf, shared_levels, corr_function):
    """
    Corr function is assumed to be symmetric and 1 if the inputs coincide
    """

    factor_combinations = aggrf.columns.sort_values()

    out = pd.DataFrame(index=factor_combinations, columns=factor_combinations)
    for (i1, col1), (i2, col2) in product(enumerate(factor_combinations), repeat=2):
        if i1 >= i2:
            continue
        # compare only if they have equal model, tuning, scoring
        if col1[shared_levels] != col2[shared_levels]:
            continue
        out.loc[col1, col2] = corr_function(aggrf[col1], aggrf[col2])

    # Make symmetric, add diagonal
    out = out.fillna(out.T)
    for i in range(len(out)):
        out.iloc[i, i] = 1.0

    return out


def compute_sqms_analysis(aggrf, shared_levels):

    compact_names = {
        "rank_mean": "RM",
        "rank_median": "RMd",
        "rank_numbest": "RB",
        "rank_num_worst": "RW",
        "rank_numworst": "RW",
        "rank_kemeny": "RK",
        "rank_nemenyi": "RN",
        "qual_mean": "QM",
        "qual_median": "QMd",
        "qual_thrbest": "QT",
        "qual_rescaled_mean": "QR"
    }

    agg_taub = square_correlation_matrix(aggrf, shared_levels,
                                         lambda x, y: kendalltau(x, y, variant="b", nan_policy="omit")[0])\
        .rename(index=compact_names, columns=compact_names)
    agg_agrbest = square_correlation_matrix(aggrf, shared_levels,
                                            lambda x, y: ru.agreement(x, y, best=True))\
        .rename(index=compact_names, columns=compact_names)
    agg_agrworst = square_correlation_matrix(aggrf, shared_levels,
                                             lambda x, y: ru.agreement(x, y, best=False))\
        .rename(index=compact_names, columns=compact_names)

    return agg_taub, agg_agrbest, agg_agrworst


def compute_correlation_matrix(aggrf, shared_levels):
    """"
    common_levels are the column levels that have to be equal in order to compute the correlation between two cols
    """

    factor_combinations = aggrf.columns.sort_values()

    # For every pair of columns, compute: taub, agreement on best, agreement on worst
    agg_taub = pd.DataFrame(index=factor_combinations, columns=factor_combinations)
    agg_agrbest = pd.DataFrame(index=factor_combinations, columns=factor_combinations)
    agg_agrworst = pd.DataFrame(index=factor_combinations, columns=factor_combinations)
    for (i1, col1), (i2, col2) in list(product(enumerate(aggrf.columns), repeat=2)):
        if i1 >= i2:
            continue
        # compare the aggregated rankings only if they have equal model, tuning, scoring
        if col1[shared_levels] != col2[shared_levels]:
            continue
        agg_taub.loc[col1, col2] = kendalltau(aggrf[col1], aggrf[col2], variant="b", nan_policy="omit")[0]
        agg_agrbest.loc[col1, col2] = ru.agreement(aggrf[col1], aggrf[col2], best=True)
        agg_agrworst.loc[col1, col2] = ru.agreement(aggrf[col1], aggrf[col2], best=False)

    # They are symmetric
    agg_taub = agg_taub.fillna(agg_taub.T)
    agg_agrbest = agg_agrbest.fillna(agg_agrbest.T)
    agg_agrworst = agg_agrworst.fillna(agg_agrworst.T)

    for i in range(len(agg_taub)):
        agg_taub.iloc[i, i] = 1
        agg_agrbest.iloc[i, i] = 1
        agg_agrworst.iloc[i, i] = 1

    # compact names
    new_names = {
        "rank_mean": "RM",
        "rank_median": "RMd",
        "rank_numbest": "RB",
        "rank_num_worst": "RW",
        "rank_numworst": "RW",
        "rank_kemeny": "RK",
        "rank_nemenyi": "RN",
        "qual_mean": "QM",
        "qual_median": "QMd",
        "qual_thrbest": "QT",
        "qual_rescaled_mean": "QR"
    }

    agg_taub = agg_taub.rename(index=new_names, columns=new_names)
    agg_agrbest = agg_agrbest.rename(index=new_names, columns=new_names)
    agg_agrworst = agg_agrworst.rename(index=new_names, columns=new_names)

    return agg_taub, agg_agrbest, agg_agrworst


aggrf = pd.read_csv(rankings_folder / "aggregated_ranks_from_average_cv_score.csv", index_col=0, header=[0, 1, 2, 3])
agg_taub, agg_agrbest, agg_agrworst = compute_sqms_analysis(aggrf, shared_levels=slice(-1))

agg_taub.to_csv(rankings_folder / "pw_AGG_kendall_tau_b_nan=omit.csv")
agg_agrbest.to_csv(rankings_folder / "pw_AGG_agrbest.csv")
agg_agrworst.to_csv(rankings_folder / "pw_AGG_agrworst.csv")
#%% 3c. Interpretation stability boxplot


def load_correlation_dataframes_interpretation(folder):
    # load (indexed) matrices of rank correlations
    agg_taub = pd.read_csv(folder / "pw_AGG_kendall_tau_b_nan=omit.csv", index_col=[0, 1, 2, 3], header=[0, 1, 2, 3])
    agg_agrbest = pd.read_csv(folder / "pw_AGG_agrbest.csv", index_col=[0, 1, 2, 3], header=[0, 1, 2, 3])
    agg_agrworst = pd.read_csv(folder / "pw_AGG_agrworst.csv", index_col=[0, 1, 2, 3], header=[0, 1, 2, 3])

    # fix: the dataset ID is not read as string
    agg_taub.index = agg_taub.index.set_levels(agg_taub.index.levels[0].astype(str), level=0)
    agg_agrbest.index = agg_agrbest.index.set_levels(agg_agrbest.index.levels[0].astype(str), level=0)
    agg_agrworst.index = agg_agrworst.index.set_levels(agg_agrworst.index.levels[0].astype(str), level=0)

    # rename the index and column levels
    factors = ["model", "tuning", "scoring", "strategy"]

    for idx in [agg_taub.index, agg_taub.columns, agg_agrbest.index, agg_agrbest.columns, agg_agrworst.index,
                agg_agrworst.columns]:
        idx.rename(factors, inplace=True)

    return agg_taub, agg_agrbest, agg_agrworst


def melt_corr_matrix_interpretation(df_corr, factor):
    """
    df_corr is an indexed square matrix of correlations between rankings obtained from different combinations of
        experimental factors
    Assume that df_corr.columns is a MultiIndex [model, tuning, scoring, strategy].
    Output a dataframe with schema (model, tuning, scoring, strategy_1, strategy_2, correlation)
    The output dataframe includes all pairwise comparisons
    """

    factors = ["model", "tuning", "scoring", "strategy"]
    try:
        factors.remove(factor)
    except ValueError:
        raise ValueError(f"{factor} is not a valid experimental factor")

    l = []
    for idx, temp in df_corr.groupby(level=factors):
        # cross section
        t = temp.xs(idx, level=factors, axis=1)
        # change names
        t.index = t.index.rename({factor: f"{factor}_1"})
        t.columns.name = f"{factor}_2"
        # stack: indexed matrix -> dataframe
        t = t.stack().reorder_levels(factors + [f"{factor}_1", f"{factor}_2"])
        l.append(t)

    return pd.concat(l, axis=0).rename("corr").to_frame().reset_index()


def compute_mat_corr_interpretation(agg_taub, agg_agrbest, agg_agrworst, factor):
    """
    Takes as input three indexed correlation matrices between combinations of experimental factors
        Their columns are MultiIndex objects [dataset, model, tuning, scoring].
    Outputs a dataframe with schema ({other_factor_1}, {other_factor_2}, {other_factor_3}, factor_1, factor_2, taub, agrbest, agrworst)
        other_factor_n is the name of another factor in [dataset, model, tuning, scoring] which is not factor
    The output dataframe includes all pairwise comparisons
    """

    factors = ["model", "tuning", "scoring", "strategy"]
    try:
        factors.remove(factor)
    except ValueError:
        raise ValueError(f"{factor} is not a valid experimental factor")
    factors.extend([f"{factor}_1", f"{factor}_2"])

    mat_corr = reduce(
        lambda l, r: l.merge(r, on=factors, how="inner"),
        [melt_corr_matrix_interpretation(agg_taub, factor),
         melt_corr_matrix_interpretation(agg_agrbest, factor),
         melt_corr_matrix_interpretation(agg_agrworst, factor)])
    mat_corr.columns = factors + ["taub", "agrbest", "agrworst"]

    return mat_corr


folder = Path(u.RESULT_FOLDER).parent / "Stability"
factor = "strategy"
agg_taub, agg_agrbest, agg_agrworst = load_correlation_dataframes_interpretation(Path(u.RESULT_FOLDER).parent / "Rankings")
mat_corr = compute_mat_corr_interpretation(agg_taub, agg_agrbest, agg_agrworst, factor)

test = False
corr_metrics = ["taub", "agrbest", "agrworst"] if not test else ["taub"]

for corr_metric in corr_metrics:
    plot_mat_corr(mat_corr, corr_metric, factor, "black", folder, latex_name, show_plot=test, draw_points=False,
                  ylim=(-1, 1) if corr_metric == "taub" else (0, 1))

print("Done")

#%% 4a. Stability in number of datasets - Compute
"""
For all interpretation strategies except Kemeny, which takes way too long. 
For a given sample size, draw two non-overlapping samples of datasest and compute the correlation between the 
    aggregated rankings. 
Repeat (to get deviation)
!!! no tuning is still missing LGBM results (classified as full tuning)
"""


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def sample_non_overlapping(S, n_samples, sample_size, seed=0):
    if n_samples * sample_size > len(S):
        raise ValueError("Not enough elements in S.")
    S_ = set(S)
    with temp_seed(seed):
        out = []
        for _ in range(n_samples):
            Snew = set(np.random.choice(list(S_), sample_size, replace=False))
            out.append(Snew)
            S_ = S_ - Snew
    return out


def melt_corr_matrix_sample(df_corr, factor):
    """
    df_corr is an indexed square matrix of correlations between rankings obtained from different combinations of
        experimental factors
    Assume that df_corr.columns is a MultiIndex [model, tuning, scoring, strategy].
    Output a dataframe with schema (model, tuning, scoring, strategy_1, strategy_2, correlation)
    The output dataframe includes all pairwise comparisons
    """

    factors = ["sample", "model", "tuning", "scoring", "strategy"]
    try:
        factors.remove(factor)
    except ValueError:
        raise ValueError(f"{factor} is not a valid experimental factor")

    l = []
    for idx, temp in df_corr.groupby(level=factors):
        # cross section
        t = temp.xs(idx, level=factors, axis=1)
        # change names
        t.index = t.index.rename({factor: f"{factor}_1"})
        t.columns.name = f"{factor}_2"
        # stack: indexed matrix -> dataframe
        t = t.stack().reorder_levels(factors + [f"{factor}_1", f"{factor}_2"])
        l.append(t)

    return pd.concat(l, axis=0).rename("corr").to_frame().reset_index()


def compute_mat_corr_sample(sample_taub, sample_agrbest, sample_agrworst, factor):
    """
    Takes as input three indexed correlation matrices between combinations of experimental factors
        Their columns are MultiIndex objects [dataset, model, tuning, scoring].
    Outputs a dataframe with schema ({other_factor_1}, {other_factor_2}, {other_factor_3}, factor_1, factor_2, taub, agrbest, agrworst)
        other_factor_n is the name of another factor in [dataset, model, tuning, scoring] which is not factor
    The output dataframe includes all pairwise comparisons
    """

    factors = ["sample", "model", "tuning", "scoring", "strategy"]
    try:
        factors.remove(factor)
    except ValueError:
        raise ValueError(f"{factor} is not a valid experimental factor")
    factors.extend([f"{factor}_1", f"{factor}_2"])

    mat_corr = reduce(
        lambda l, r: l.merge(r, on=factors, how="inner"),
        [melt_corr_matrix_sample(sample_taub, factor),
         melt_corr_matrix_sample(sample_agrbest, factor),
         melt_corr_matrix_sample(sample_agrworst, factor)])
    mat_corr.columns = factors + ["taub", "agrbest", "agrworst"]

    return mat_corr


class SampleAggregator(object):

    def __init__(self, df, rf, sample_size, seed=0):
        """
        df and rf are already restricted to values corresponding to no tuning

        """

        self.sample_size = sample_size
        self.seed = seed

        self.df = df.query("tuning == 'no'")
        self.rf = rf.loc(axis=1)[:, :, "no", :]

        self.a1 = None  # Aggregator
        self.a2 = None  # Aggregator

        self.aggrf = pd.DataFrame()  # Aggregated rankings

        self.datasets_sample_1, self.datasets_sample_2 = sample_non_overlapping(self.df.dataset.unique(), n_samples=2,
                                                                                sample_size=self.sample_size,
                                                                                seed=self.seed)
        self.df1 = self.df.query("dataset in @self.datasets_sample_1")
        self.df2 = self.df.query("dataset in @self.datasets_sample_2")
        self.rf1 = self.rf.loc(axis=1)[[str(x) for x in self.datasets_sample_1], :, :, :]
        self.rf2 = self.rf.loc(axis=1)[[str(x) for x in self.datasets_sample_2], :, :, :]

    def aggregate(self, verbose=False, strategies: Union[list, str, set] = "all",
                  ignore_strategies: Union[list, tuple] = tuple(), **kwargs):

        self.a1 = Aggregator(self.df1, self.rf1).aggregate(strategies=strategies,
                                                           ignore_strategies=ignore_strategies, verbose=verbose,
                                                           **kwargs)
        self.a2 = Aggregator(self.df2, self.rf2).aggregate(strategies=strategies,
                                                           ignore_strategies=ignore_strategies, verbose=verbose,
                                                           **kwargs)

        factors = ["sample", "model", "tuning", "scoring", "strategy"]
        self.aggrf = pd.concat([pd.concat({str(self.datasets_sample_1): self.a1.aggrf}, names=factors, axis=1),
                               pd.concat({str(self.datasets_sample_2): self.a2.aggrf}, names=factors, axis=1)],
                               axis=1)

        return self


rankings_folder = Path(u.RESULT_FOLDER).parent / "Rankings"
df = pd.read_csv(Path(u.RESULT_FOLDER, "final.csv"))
rf = pd.read_csv(rankings_folder / "rank_function_from_average_cv_score.csv", index_col=0, header=[0, 1, 2, 3])

# As we want to use all 50 datasets, filter for tuning
df_ = df.query("tuning == 'no'")
rf_ = rf.loc(axis=1)[:, :, "no", :]
factors = ["sample", "model", "tuning", "scoring", "strategy"]

global_mat_corr = pd.read_csv(rankings_folder / "pw_corr_SAMPLE_with_strategy_std.csv")
run = False
if run:
    # whenever we add experiments, start from the value of seed
    seed = len(global_mat_corr)
    sample_sizes = [5, 10, 15, 20, 25]
    repetitions = 20
    mat_corrs = []
    sample_aggregators = defaultdict(lambda: [])
    for sample_size in tqdm(sample_sizes):
        inner_mat_corrs = []
        inner_sample_aggregators = []
        for _ in tqdm(range(repetitions)):
            seed += 1

            a = SampleAggregator(df, rf, sample_size, seed=seed).aggregate(ignore_strategies=["kemeny rank"],
                                                                           verbose=False)

            mc = compute_mat_corr_sample(*compute_sqms_analysis(a.aggrf, shared_levels=slice(1, None)),
                                         factor="sample")

            inner_mat_corrs.append(mc.assign(sample_size=sample_size).query("sample_1 < sample_2"))
            sample_aggregators[sample_size].append(a)
        mat_corrs.append(pd.concat(inner_mat_corrs, axis=0))
    # if mat_corr is already defined, make it bigger! We are adding experiments
    mat_corr = pd.concat(mat_corrs, axis=0)
    mat_corr = mat_corr.join(mat_corr.groupby(["strategy", "sample_size"])[["taub", "agrbest", "agrworst"]].std(),
                             on=["strategy", "sample_size"], rsuffix="_std")
    global_mat_corr = pd.concat([global_mat_corr, mat_corr], axis=0)

global_mat_corr.to_csv(rankings_folder / "pw_corr_SAMPLE_with_strategy_std.csv", index=False)

#%% 4b. Stability in number of datasets - Plots

mat_corr = pd.read_csv(rankings_folder / "pw_corr_SAMPLE_with_strategy_std.csv")
folder = Path(u.RESULT_FOLDER).parent / "Stability"

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
    "RK": None
}

corr_metric = "agrbest"

fig, ax = plt.subplots(1, 1, sharey="all")

sns.barplot(mat_corr, x="sample_size", y=corr_metric, hue="strategy", palette=strategy_palette, ax=ax)
sns.move_legend(ax, loc="upper right", bbox_to_anchor=(1.4, 1))
ax.set_ylabel(latex_name[corr_metric])

sns.despine(top=True, trim=True)

fig.set_size_inches(7.25, 5)
fig.tight_layout()
fig.savefig(folder / f"dataset_sample_stability_{corr_metric}.svg", dpi=600)

#%% 5. Most sensitive datasets - datasets with high performance difference (OUTDATED - NOT IN FULL PAPER)
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


def sorted_boxplot(data, x, y, **kwargs):
    sns.boxplot(data, x=x, y=y,
                order=index_sorted_by_median(data, groupby_col="encoder", target_col="rank"),
                **kwargs)


rankings_folder = Path(u.RESULT_FOLDER).parent / "Rankings"
df = pd.read_csv(Path(u.RESULT_FOLDER, "final.csv"))
rf = pd.read_csv(rankings_folder / "rank_function_from_average_cv_score.csv", index_col=0, header=[0, 1, 2, 3])

factors = ["dataset", "model", "tuning", "scoring"]

b = df.groupby(factors).cv_score.agg(iqr).dropna()
b = b.loc[b >= b.median() + iqr(b)].reset_index()
b = b.groupby("dataset").size().sort_values()
b = b.loc[b >= b.median()]  # 90 is half the combinations of fold-model-tuning-scoring (NO LGBM YET)

# high-variability datasets
rf_hv = rf[b.index.astype(str)]

folder = Path(u.RESULT_FOLDER).parent / "Full results"

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
#%% 6. Rank of encoders plots

folder = Path(u.RESULT_FOLDER).parent / "Full results"

sns.set_theme(style="whitegrid", font_scale=0.3)

melt_rf = rf.melt(ignore_index=False).reset_index()
melt_rf.columns = ["encoder", "dataset", "model", "tuning", "scoring", "rank"]

grid = sns.FacetGrid(melt_rf, col="scoring", row="model",
                     margin_titles=True, sharey=False)

grid.set_titles(row_template="{row_name}", col_template="{col_name}")

grid.map_dataframe(sorted_boxplot, x="rank", y="encoder",
                   palette="crest", showfliers=False, linewidth=0.2, showcaps=False,
                   medianprops=dict(color="red", linewidth=0.4))
# grid.set_xticklabels(rotation=90)

grid.despine(top=True, trim=True)

grid.fig.set_size_inches(7.25, 10)
grid.fig.tight_layout()
grid.savefig(folder / f"encoder_rank_boxplot_matrix.svg", dpi=600)

# plt.show()

print("Done")

#%% 6a. Plot for one model and scoring

latex_encoders = {enc: enc for enc in df.encoder.unique()}
latex_encoders.update({
    "BUCV10RGLMME": r"BUCV$_{10}$RGLMME",
    "BUCV2RGLMME": r"BUCV$_{2}$RGLMME",
    "BUCV5RGLMME": r"BUCV$_{5}$RGLMME",
    "BUCV10TE": r"BUCV$_{10}$TE",
    "BUCV2TE": r"BUCV$_{2}$TE",
    "BUCV5TE": r"BUCV$_{5}$TE",
    "CV10RGLMME": r"CV$_{10}$RGLMME",
    "CV2RGLMME": r"CV$_{2}$RGLMME",
    "CV5RGLMME": r"CV$_{5}$RGLMME",
    "CV10TE": r"CV$_{10}$TE",
    "CV2TE": r"CV$_{2}$TE",
    "CV5TE": r"CV$_{5}$TE",
    "DTEM10": r"D$_{10}$TE",
    "DTEM2": r"D$_{2}$TE",
    "DTEM5": r"D$_{5}$TE",
    "ME01E": r"MEE$_{0.1}$",
    "ME1E": r"MEE$_{1}$",
    "ME10E": r"MEE$_{10}$",
    "PBTE0001": r"PB$_{0.001}$TE",
    "PBTE001": r"PB$_{0.01}$TE",
    "PBTE01": r"PB$_{0.1}$TE"
})


folder = Path(u.RESULT_FOLDER).parent / "Full results"

sns.set_theme(style="whitegrid", font_scale=0.5)

model = "LGBMC"
scoring = "AUC"

melt_rf = rf.melt(ignore_index=False).reset_index()
melt_rf.columns = ["encoder", "dataset", "model", "tuning", "scoring", "rank"]
melt_rf = melt_rf.query("model == @model and scoring == @scoring")
melt_rf.encoder = melt_rf.encoder.map(latex_encoders)

fig, ax = plt.subplots(1, 1, figsize=(2.5, 3.25), dpi=600)
sorted_boxplot(data=melt_rf, x="rank", y="encoder",
               palette=sns.light_palette("grey", n_colors=melt_rf.encoder.nunique()),
               showfliers=True, linewidth=1, showcaps=True,
               medianprops=dict(color="red", linewidth=1), ax=ax)

fig.tight_layout()
fig.savefig(folder / f"encoder_rank_{model}_{scoring}.svg", dpi=600)

plt.close("all")
plt.show()

#%% 7. Performance effect of tuning (EMPTY)
pass
#%% 8.

