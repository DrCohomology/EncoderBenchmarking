"""
Comparison and ranking utilities
"""
import gurobipy as gp
import numpy as np
import pandas as pd

from collections import defaultdict
from pathlib import Path
from scipy.stats import iqr
from scikit_posthocs import posthoc_nemenyi_friedman
from tqdm.notebook import tqdm
from typing import Iterable, Union

import src.utils as u
import src.relation_utils as rlu
import src.rank_metrics as rm

class BaseAggregator(object):
    """
    Aggregate all qualities, saved in 'self.df', and/or rankings, saved in 'self.rf', into a consensus ranking for
        each aggregation strategy used.

    self.df is the dataframe of experimental evaluations and has columns:
        'encoder', 'dataset', 'fold', 'model', 'tuning', 'scoring', 'cv_score', 'tuning_score', 'time',
        'model__max_depth', 'model__n_neighbors', 'model__n_estimators', 'model__C', 'model__gamma'
    self.rf is the dataframe of rankings and has:
        index: encoders
        columns: all combinations of dataset, model, tuning, scoring

    The output of aggregation is 'self.aggrf', which has:
        index: encoders
        columns: aggregation strategies
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
        self.increasing = {
            "rank_mean": True,
            "rank_median": True,
            "rank_numbest": False,
            "rank_numworst": True,
            "rank_kemeny": True,
            "rank_nemenyi_0.01": True,
            "rank_nemenyi_0.05": True,
            "rank_nemenyi_0.10": True,
            "qual_mean": False,
            "qual_median": False,
            "qual_thrbest_0.95": False,
            "qual_rescaled_mean": False
        }

    def _aggr_rank_mean(self, **kwargs):
        """
        Aggregate with the average rank.
        """
        self.aggrf["rank_mean"] = self.rf.mean(axis=1)

    def _aggr_rank_median(self, **kwargs):
        """
        Aggregate with the median rank.
        """
        self.aggrf["rank_median"] = self.rf.median(axis=1)

    def _aggr_rank_numbest(self, **kwargs):
        """
        Aggregate with the number of times an encoder is in the best tier.
        """
        self.aggrf["rank_numbest"] = (self.rf == self.rf.min(axis=0)).sum(axis=1)

    def _aggr_rank_numworst(self, **kwargs):
        """
        Aggregate with the number of times an encoder is in the worst tier.
        """
        self.aggrf["rank_numworst"] = (self.rf == self.rf.max(axis=0)).sum(axis=1)

    def _aggr_rank_kemeny(self, **solver_params):
        """
        Aggregated solving the MILP Kemeny optimization problem.
        Refer to 'kemeny_aggregation_gurobi_ties' for details.
        """
        self.aggrf["rank_kemeny"] = kemeny_aggregation_gurobi_ties(self.rf.set_axis(range(self.rf.shape[1]), axis=1),
                                                                   **{k: v for k, v in solver_params.items()
                                                                       if k in ["Seed", "Start"]})

    def _aggr_rank_nemenyi(self, alpha=0.05):
        """
        Aggregate according to pairwise Nemenyi tests performed for every pair of alternatives.
        Aggregation is performed by multiplying elementwise the matrix of relation
            "average rank of a1 <= average rank of a2" and the matrix of relation
            "the average rank of a1 is significatively differetnt from that of a2".
            The resulting matrix is then translated into a ranking.
        Although, in principle, the matrix is not necessarily transitive, we never encountered that problem.
        """
        self.aggrf[f"rank_nemenyi_{alpha:.02f}"] = rlu.mat2rf(
             rlu.rf2mat(rlu.score2rf(self.rf.mean(axis=1)), kind="domination") *
             (posthoc_nemenyi_friedman(self.rf.T.reset_index(drop=True)) < alpha).astype(int).to_numpy(),
             alternatives=self.rf.index
        )

    def _aggr_qual_mean(self, **kwargs):
        """
        Aggregate with the average quality of an encoder.
        """
        self.aggrf["qual_mean"] = self.df.groupby("encoder").cv_score.agg(np.nanmean)

    def _aggr_qual_median(self, **kwargs):
        """
        Aggregate with the median quality of an encoder.
        """
        self.aggrf["qual_median"] = self.df.groupby("encoder").cv_score.median(np.nanmedian)

    def _aggr_qual_thrbest(self, thr=0.95, **kwargs):
        """
        Aggregate with the number of times an encoder has quality >= 'thr'*best_quality_on_dataset
        """
        self.aggrf[f"qual_thrbest_{thr}"] = (
            self.df.groupby(["dataset", "encoder"]).cv_score.mean().to_frame().reset_index()
                   .join(self.df.groupby("dataset").cv_score.max(), on="dataset", rsuffix="_best")
                   .query("cv_score >= @thr*cv_score_best").groupby("encoder").size()
                   .reindex(self.rf.index).fillna(0)
        )

    def _aggr_qual_rescaled_mean(self, **kwargs):
        """
        Rescale the quality
        """
        d1 = self.df.groupby(["dataset", "encoder"]).cv_score.mean().to_frame().reset_index()\
                 .join(self.df.groupby(["dataset"]).cv_score.agg(np.nanmin), on="dataset", rsuffix="_worst")\
                 .join(self.df.groupby(["dataset"]).cv_score.agg(iqr), on="dataset", rsuffix="_iqr")
        d1["cv_score_rescaled"] = (d1["cv_score"] - d1["cv_score_worst"]) / d1["cv_score_iqr"]
        # iqr == 0 means that all encoders are equal on the dataset, i.e., we should ignore the comparison anyway
        self.aggrf["qual_rescaled_mean"] = d1.query("cv_score_iqr != 0").groupby("encoder").cv_score_rescaled.mean()

    def aggregate(self, strategies: Iterable = "all", ignore_strategies: Iterable = tuple(), **kwargs):
        """
        Aggregate.
        """
        if strategies == "all":
            strategies = self.supported_strategies.keys()
        for strategy in set(strategies) - set(ignore_strategies):
            self.supported_strategies[strategy](**kwargs)

        # Transform the scores into rankings
        for col in self.aggrf:
            self.aggrf[col] = rlu.score2rf(self.aggrf[col], increasing=self.increasing[col])

        return self


class Aggregator(object):
    """
    Aggregate qualities, saved in 'self.df', and/or rankings, saved in 'self.rf', into a consensus ranking for
        each aggregation strategy used. The aggregation, unlike for BaseAggregator, does not happen across all
        rankings/qualities, but rather in subsets of rankings/qualities obtained by fixing some experimental
        factors --- model, tuning, and scoring.

    self.df is the dataframe of experimental evaluations and has columns:
        'encoder', 'dataset', 'fold', 'model', 'tuning', 'scoring', 'cv_score', 'tuning_score', 'time',
        'model__max_depth', 'model__n_neighbors', 'model__n_estimators', 'model__C', 'model__gamma'
    self.rf is the dataframe of rankings and has:
        index: encoders
        columns: all combinations of dataset, model, tuning, scoring

    The output of aggregation is 'self.aggrf', which has:
        index: encoders
        columns: all combinations of model, tuning, scoring, and aggregation strategies
    """

    def __init__(self, df: pd.DataFrame, rf: pd.DataFrame):
        self.df = df
        self.rf = rf

        self.combinations = self.df.groupby(["model", "tuning", "scoring"]).size().index
        self.base_aggregators = {(m, t, s): BaseAggregator(df.query("model == @m and tuning == @t and scoring == @s"),
                                                           rf.loc(axis=1)[:, m, t, s])
                                 for m, t, s in self.combinations}
        self.aggrf = pd.DataFrame(index=rf.index)

    def to_csv(self, path: Path):
        """
        Saves 'self.aggrf' into a csv file.
        """
        if self.aggrf.empty:
            raise Exception("self.aggrf is not populated yet. Called self.aggregate before saving.")
        return self.aggrf.to_csv(path)

    def aggregate(self, strategies: Union[list, set, str] = "all", ignore_strategies: Union[tuple, list] = tuple(),
                  verbose: bool = False, **kwargs):

        comb_iter = tqdm(list(self.combinations)) if verbose else self.combinations
        for model, tuning, scoring in comb_iter:
            a = self.base_aggregators[(model, tuning, scoring)]
            a.aggregate(strategies, ignore_strategies=ignore_strategies, **kwargs)
            a.aggrf.columns = pd.MultiIndex.from_product([[model], [tuning], [scoring], a.aggrf.columns],
                                                         names=["model", "tuning", "scoring", "aggregation"])

        self.aggrf = pd.concat([a.aggrf for a in self.base_aggregators.values()], axis=1)

        return self


class SampleAggregator(object):

    def __init__(self, df, rf, sample_size, seed=0, bootstrap=False):
        """
        df and rf are already restricted to values corresponding to no tuning

        """

        self.sample_size = sample_size
        self.seed = seed

        self.df = df
        self.rf = rf

        self.a1 = None  # Aggregator
        self.a2 = None  # Aggregator

        self.aggrf = pd.DataFrame()  # Aggregated rankings

        self.datasets_sample_1, self.datasets_sample_2 = u.get_disjoint_samples(self.df.dataset.unique(), n_samples=2,
                                                                                sample_size=self.sample_size,
                                                                                seed=self.seed, bootstrap=bootstrap)
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

        factors = ["sample", "model", "tuning", "scoring", "aggregation"]
        self.aggrf = pd.concat([pd.concat({str(self.datasets_sample_1): self.a1.aggrf}, names=factors, axis=1),
                               pd.concat({str(self.datasets_sample_2): self.a2.aggrf}, names=factors, axis=1)],
                               axis=1).reorder_levels([1, 2, 3, 4, 0], axis=1)

        return self


def kemeny_aggregation_gurobi_ties(rf: pd.DataFrame, **solver_params):
    """
    Based on the mixed-integer optimization problem defined in [1], with distance adapted for ties and missing values
        from [2].

    [1] Yoo, Y., & Escobedo, A. R. (2021). A new binary programming formulation and social choice property for
        Kemeny rank aggregation. Decision Analysis, 18(4), 296-320.
    [2] Moreno-Centeno, E., & Escobedo, A. R. (2016). Axiomatic aggregation of incomplete rankings.
        IIE Transactions, 48(6), 475-488.

    """

    ms = rlu.dr2mat(rf, kind="yoo")  # ms[i] = ms[i, :, :] is the adjacency matrix of the i-th ranking
    nv, na, _ = ms.shape  # number of voters, number of alternatives

    # If any rank is missing, use the cost matrix from [1], 'c'
    if np.isnan(ms).any():
        # weight = how many alternatives were ranked for voter i: minimum number of nan columns in ms[i]
        w = na - np.isnan(ms).sum(axis=1).min(axis=1)
        c = np.nansum([ms[i] / (w[i] * (w[i]-1)) for i in range(nv)], axis=0)
    else:
        c = np.sum(ms, axis=0)

    # MILP formulation
    model = gp.Model()
    model.setParam("OutputFlag", 0)
    median = model.addMVar(shape=(na, na), vtype=gp.GRB.BINARY)

    # We do not care for other parameters (at the moment)
    for k, v in solver_params.items():
        if k == "Seed":
            model.setParam(k, v)
        elif k == "Start":
            median.setAttr(k, v)
        else:
            raise ValueError(f"Parameter {k} with value {v} is not a valid input parameter.")

    model.setObjective(gp.quicksum(gp.quicksum((2 * median - np.ones((na, na))) * c)), gp.GRB.MAXIMIZE)
    # multiple calls to addConstr are preferrable to a single call to addConstrs
    for cs in rlu.get_constraints(median, "yoo_weak_order"):
        model.addConstr(cs, name="")

    model.update()
    model.optimize()

    return rlu.mat2rf(median.X, alternatives=rf.index)


def replicability_analysis(df, rf, tuning, seed=0, sample_sizes=range(5, 26, 5), repetitions=100, append_to_existing=True, save=True):
    df_ = df.query("tuning == @tuning")
    rf_ = rf.loc(axis=1)[:, :, tuning, :].copy()

    if append_to_existing:
        sample_df_sim = u.load_sample_similarity_dataframe(tuning=tuning)
    else:
        sample_df_sim = pd.DataFrame()

    # whenever we add experiments, start from the value of seed
    mat_corrs = []
    sample_aggregators = defaultdict(lambda: [])
    for sample_size in tqdm(sample_sizes):
        inner_mat_corrs = []
        inner_sample_aggregators = []
        for _ in tqdm(range(repetitions)):
            seed += 1
            a = SampleAggregator(df_, rf_, sample_size, seed=seed, bootstrap=True).aggregate(
                ignore_strategies=["kemeny rank"],
                verbose=False)

            tmp_jaccard, tmp_rho = u.pairwise_similarity_wide_format(a.aggrf,
                                                                     simfuncs=[rm.jaccard_best,
                                                                               rm.spearman_rho])
            agg_sample_long = u.join_wide2long(dict(zip(["jaccard", "rho"],
                                                        [tmp_jaccard, tmp_rho])),
                                               comparison_level="sample")

            inner_mat_corrs.append(agg_sample_long.assign(sample_size=sample_size).query("sample_1 < sample_2"))
            sample_aggregators[sample_size].append(a)
        mat_corrs.append(pd.concat(inner_mat_corrs, axis=0))
    # if mat_corr is already defined, make it bigger! We are adding experiments
    mat_corr = pd.concat(mat_corrs, axis=0)
    mat_corr = mat_corr.join(
        mat_corr.groupby(["aggregation", "sample_size"])[["jaccard", "rho"]].std(),
        on=["aggregation", "sample_size"], rsuffix="_std")
    sample_df_sim = pd.concat([sample_df_sim, mat_corr], axis=0)

    # rename to AGGREGATION_NAMES but allow correctly formatted names
    sample_df_sim.aggregation = sample_df_sim.aggregation.map(lambda x: defaultdict(lambda: x, u.AGGREGATION_NAMES)[x])

    if save:
        sample_df_sim.to_csv(u.RANKINGS_DIR / f"sample_sim_{tuning}.csv", index=False)


# def kemeny_aggregation_cvxpy(dr, solver=cp.GUROBI, consensus_kind="total_order", solver_opts=None):
#     """
#     Returns the median consensus according to the symmetric distance. See Hornik and Meyer (2007).
#     based on cvxpy module
#     consensus_kind =
#         weak_order: totality and transitivity
#         total_order: antisymmetry and transitivity (note that we do not care about i == j)
#         strict_order: antisymmetry and transitivity
#     """
#     ms = dr2mat(dr, kind="preference")
#
#     # this function cannot handle missgin values
#     assert not np.isnan(ms).any()
#
#     nv, na, _ = ms.shape    # num_voters, num_alternatives, num_alternatives
#     c = np.sum(ms, axis=0)  # c matrix in the paper, with diagonal 0
#     median = cp.Variable(shape=(na, na), boolean=True)
#
#     # --- run the optimization, which returns an incidence matrix
#     p = cp.Problem(
#         cp.Maximize(cp.sum(cp.multiply(c, median))),
#         get_constraints(median, consensus_kind)
#     )
#     p.solve(solver=solver, solver_opts=solver_opts)
#
#     return mat2rf(np.array(median.value, dtype=int), alternatives=dr.index, kind="incidence")
