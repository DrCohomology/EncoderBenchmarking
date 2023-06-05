"""
Comparison and ranking utilities
"""

import cvxpy as cp
import gurobipy as gp
import mip
import numpy as np
import pandas as pd
import time
import warnings

from collections import defaultdict
from functools import reduce
from itertools import product, permutations
from pathlib import Path
from scipy.stats import kendalltau, t, iqr, spearmanr
from scikit_posthocs import posthoc_nemenyi_friedman
from tqdm import tqdm
from typing import Iterable, Union

import src.utils as u


def t_test(v1, v2, alpha=0.05, corrected=True):
    """
    Test whether one of v1 or v2 is statistically greater than the other.
    Assume v1 and v2 are results from a cross validation
    Add Bengio correction term (2003)
    """
    if len(v1) != len(v2):
        raise ValueError("The inputs should have equal length.")

    n = len(v1)

    diff = v1 - v2
    avg = diff.mean()
    std = diff.std()
    if std == 0:
        return 0, 0, 0, 0

    # test training ratio
    ttr = 1 / (n - 1)

    adjstd = np.sqrt(1 / n + ttr) * std if corrected else np.sqrt(1 / n) * std
    tstat = avg / adjstd

    df = n - 1
    crit = t.ppf(1.0 - alpha, df)
    p = (1.0 - t.cdf(np.abs(tstat), df)) * 2.0
    return p, tstat, df, crit


def compare_with_ttest(v1, v2, alpha=0.05, corrected=True):
    """
    returns 0 if the two are indistinguishable
    returns 1 if v1 is "greater than" v2
    returns 2 if v2 is "greater than" v1

    """

    if np.array(v1 == v2).all():
        return 0, 1

    p, s1, s2, s3 = t_test(v1, v2, alpha=alpha, corrected=corrected)

    if p >= alpha:
        return 0, p
    else:
        if (v1 - v2).mean() > 0:
            return 1, p
        else:
            return 2, p


def get_dominating(R):
    """
    Retrieves the indices of non-dominated elements and "removes" them from the matrix.
    At the end of the recursive process, the result is a list of nested sets.
    """
    if np.linalg.norm(R - np.ones_like(R)) == 0:
        return [set(range(len(R)))]

    non_dominated = []
    R = R.copy()
    for i in range(len(R)):
        if R[i, :].sum() == len(R):
            non_dominated.append(i)
    # having found the index of non-dominated rows, we can make the corresponding columns ininfluent
    # ie make the other indices dominate also the dominating indices
    for i in non_dominated:
        R[i, :] = np.ones_like(R[i, :])
        R[:, i] = np.ones_like(R[:, i])
    final_rank = [set(non_dominated)]
    final_rank.extend(get_dominating(R))
    return final_rank


def filter_rank(r):
    """
    Transforms a list of nested sets into a list of increments
    """
    return [r[0]] + [r[i]-r[i-1] for i in range(1, len(r))]


def test_totality(R):

    """
    Tests for totality AND reflexivity of a relation.
    """
    if len(R.shape) >= 3 or R.shape[0] != R.shape[1]:
        raise ValueError(f"The input should be a square matrix but has shape {R.shape}.")
    return reduce(
        lambda x, y: x and y,
        [
            R[i, j] + R[j, i] >= 1
            for i, j in product(range(len(R)), repeat=2)
        ]
    )


def test_transitivity(R):
    if len(R.shape) >= 3 or R.shape[0] != R.shape[1]:
        raise ValueError(f"The input should be a square matrix but has shape {R.shape}.")
    return reduce(
        lambda x, y: x and y,
        [
            R[i, j] + R[j, k] - R[i, k] <= 1
            for i, j, k in product(range(len(R)), repeat=3)
        ]
    )


def agreement(col1: pd.Series, col2: pd.Series, best: bool = True):
    """
    Agreement measured with Jaccard similarity of the best or worst tier.

    best:
        True: agreement on the best tiers, i.e., the tiers of alternatives that achieve minimum rank in their
            respective rankings
        False: agreement on the worst tiers, i.e., the tiers of alternatives that achieve maximum rank in their
            respective rankings

    """
    o1 = col1.min() if best else col1.max()
    o2 = col2.min() if best else col2.max()
    b1 = set(col1[col1 == o1].index)
    b2 = set(col2[col2 == o2].index)
    return len(b1.intersection(b2)) / len(b1.union(b2))


def agreement_best(col1: pd.Series, col2: pd.Series):
    return agreement(col1, col2, best=True)


def agreement_worst(col1: pd.Series, col2: pd.Series):
    return agreement(col1, col2, best=False)


def kendall_tau(x: Iterable, y: Iterable, variant="b", nan_policy="omit"):
    return kendalltau(x, y, variant=variant, nan_policy=nan_policy)[0]


def kendall_tau_p(x: Iterable, y: Iterable, variant="b", nan_policy="omit"):
    return kendalltau(x, y, variant=variant, nan_policy=nan_policy)[1]


def spearman_rho(x: Iterable, y: Iterable, variant="b", nan_policy="omit"):
    try:
        return spearmanr(x, y, nan_policy=nan_policy)[0]
    except Exception as error:
        print(error)
        return np.nan

# ---  Rank aggregation basics

def d_kendall(r1, r2):
    """
    r1 and r2 are rank functions, ie r1[i] is the rank of the i-th element
    """

    assert len(r1) == len(r2)

    c = 0
    for i in range(len(r1)):
        for j in range(i):
            temp = np.sign((r1[j]-r1[i])*(r2[j]-r2[i])) - 1
            if not np.isnan(temp):
                c -= temp
    return c/2


def d_kendall_set(c, dr):
    return np.sum([d_kendall(c, dr[col]) for col in dr.columns])


def score2rf(score: pd.Series, ascending=True):
    """
    Ascending =
        True: lower score = better rank (for instance, if score is the result of a loss function or a ranking itself)
        False: greater score = better rank (for instance, if score is the result of a score such as roc_auc_score)
    """
    c = 1 if ascending else -1
    order_map = {
        s: sorted(score.unique(), key=lambda x: c * x).index(s) for s in score.unique()
    }
    return score.map(order_map)


def rf2mat(r: np.ndarray, kind="preference"):
    """
    rank function (identified as a vector) to domination (outranking) matrix
    kind =
        preference: computes the antisymmetric preference matrix Mij = int(Ri < Rj), Mji = -Mij
        {domination, outranking}: computes the domination (outranking) matrix Mij = 1 iff (Ri <= Rj) else 0
        incidence: computes the incidence matrix of a strict linear order Mij = 1 iff R1 < Rj
        ranking: as preference, but Mij = 1 if Ri <= Rj, -1 if Ri > Rj, 0 if i=j
            Adapted from Yoo (2021)
    """
    na = len(r)     # num alternatives
    mat = - np.zeros((na, na))
    for i, j in product(range(na), repeat=2):
        if j > i:
            continue
        if kind == 'preference':
            mat[i, j] = np.sign((r[j] - r[i]))
            mat[j, i] = - mat[i, j]
        elif kind in {"domination", "outranking"}:
            mat[i, j] = int(r[i] <= r[j])
            mat[j, i] = int(r[j] <= r[i])
        elif kind == "incidence":
            mat[i, j] = int(r[i] < r[j])
            mat[j, i] = int(r[j] < r[i])
        elif kind == "ranking":
            if i == j:
                mat[i, j] = 0
            elif r[i] == r[j]:
                mat[i, j] = mat[j, i] = 1
            else:
                mat[i, j] = np.sign((r[j] - r[i]))
                mat[j, i] = - mat[i, j]
        else:
            raise ValueError(f"kind={kind} is not accepted.")

    return mat


def mat2rf(mat: np.ndarray, alternatives):
    """
    Use dominance to retrieve the rank function
    """
    if "transitivity" not in get_relation_properties(mat):
        raise ValueError("mat2rf. Dominance is only usable if the matrix is transitive.")
    return score2rf(pd.Series(np.sum(mat, axis=0), index=alternatives))


def dr2mat(dr: pd.DataFrame, kind="preference"):
    """
    dr.index is the set of alternatves
    dr.columns are the voters
    """
    # voter X (alternative X alternative)
    ms = np.zeros((dr.shape[1], dr.shape[0], dr.shape[0]))
    for col in dr:
        ms[col] = rf2mat(dr[col].to_numpy(), kind=kind)

    return ms


def get_constraints(median: np.array, consensus_kind="total_order"):
    """
    consensus_kind =
        weak_order: totality and transitivity (reflexivity?)
        total_order: antisymmetry and transitivity (note that we do not care about i == j)
        strict_order: antisymmetry and transitivity
        yoo_weak_order: Adapted from Yoo (2021): acyclicity, totality,
        all: return all constraints
    """
    na = median.shape[0]

    # ---  constraints
    totality = [
        median[i, j] + median[j, i] >= 1
        for i, j in product(range(na), repeat=2) if i < j
    ]
    reflexivity = [
        median[i, i] == 1
        for i in range(na)
    ]
    antisymmetry = [
        median[i, j] + median[j, i] == 1
        for i, j in product(range(na), repeat=2) if i < j
    ]
    transitivity = [
        median[i, j] + median[j, k] - median[i, k] <= 1
        for i, j, k in product(range(na), repeat=3) if i != j != k != i
    ]
    acyclicity = [
        median[i, j] - median[k, j] - median[i, k] >= -1
        for i, j, k in product(range(na), repeat=3) if i != j != k != i
    ]

    if consensus_kind == "total_order":
        return reflexivity + antisymmetry + transitivity
    elif consensus_kind == "weak_order":
        return totality + transitivity
    elif consensus_kind == "strict_order":
        return antisymmetry + transitivity
    elif consensus_kind == "yoo_weak_order":
        return acyclicity + totality
    else:
        raise ValueError(f"consensus_kind = {consensus_kind} is not a valid value.")


def get_relation_properties(mat: np.array):
    """
    model is a domination matrix
    """

    if len(mat.shape) >= 3 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"The input should be a square matrix but has shape {mat.shape}.")

    na = mat.shape[0]

    def check(l):
        return len(l) == sum(l)

    properties = {
        "totality": check([mat[i, j] + mat[j, i] >= 1 for i, j in product(range(na), repeat=2) if i < j]),
        "reflexivity": check([mat[i, i] == 1 for i in range(na)]),
        "antisymmetry": check([mat[i, j] + mat[j, i] == 1 for i, j in product(range(na), repeat=2) if i < j]),
        "transitivity": check([mat[i, j] + mat[j, k] - mat[i, k] <= 1
                               for i, j, k in product(range(na), repeat=3) if i != j != k != i]),
        "acyclicity": check([mat[i, j] - mat[k, j] - mat[i, k] >= -1
                            for i, j, k in product(range(na), repeat=3) if i != j != k != i]),
    }
    return [p for p, satisfied in properties.items() if satisfied]


# --- Aggregation


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
            "rank_nemenyi_0.01": True,
            "rank_nemenyi_0.05": True,
            "rank_nemenyi_0.10": True,
            "qual_mean": False,
            "qual_median": False,
            "qual_thrbest_0.95": False,
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
        self.aggrf["rank_kemeny"] = kemeny_aggregation_gurobi_ties(self.rf.set_axis(range(self.rf.shape[1]), axis=1),
                                                                   **{k: v for k, v in solver_params.items()
                                                                       if k in ["Seed", "Start"]})

    def _aggr_rank_nemenyi(self, alpha=0.05):
        """
        Issues with no answer (yet):
            is the test "good" when the rankings involved have different number of tiers?
            does it support missing values?
            transitivity is not guaranteed, however, it seems to be always transitive

        Compute the outranking (domination) matrix and the matrix of significant differences according to Nemenyi pw tests,
        then multiply them together to get the significative differences matrix, and rebuild a rank function from it
        """
        self.aggrf[f"rank_nemenyi_{alpha:.02f}"] = mat2rf(
             rf2mat(score2rf(self.rf.mean(axis=1)), kind="domination") *
             (posthoc_nemenyi_friedman(self.rf.T.reset_index(drop=True)) < alpha).astype(int).to_numpy(),
             alternatives=self.rf.index
        )

    def _aggr_qual_mean(self, **kwargs):
        self.aggrf["qual_mean"] = self.df.groupby("encoder").cv_score.agg(np.nanmean)

    def _aggr_qual_median(self, **kwargs):
        self.aggrf["qual_median"] = self.df.groupby("encoder").cv_score.median(np.nanmedian)

    def _aggr_qual_thrbest(self, thr=0.95, **kwargs):
        """
        Count the number of datasets on which an encoder achieves quality >= thr*best
            best is the best performance on a dataset
        """
        self.aggrf[f"qual_thrbest_{thr}"] = (
            self.df.groupby(["dataset", "encoder"]).cv_score.mean().to_frame().reset_index()
                   .join(self.df.groupby("dataset").cv_score.max(), on="dataset", rsuffix="_best")
                   .query("cv_score >= @thr*cv_score_best").groupby("encoder").size()
                   .reindex(self.rf.index).fillna(0)
        )

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
            self.aggrf[col] = score2rf(self.aggrf[col], ascending=self.ascending[col])

        return self


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


    def to_csv(self, path: Path):
        """
        Saves self.aggrf to a dataframe in the specified path
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
                                                         names=["model", "tuning", "scoring", "interpretation"])

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

        factors = ["sample", "model", "tuning", "scoring", "interpretation"]
        self.aggrf = pd.concat([pd.concat({str(self.datasets_sample_1): self.a1.aggrf}, names=factors, axis=1),
                               pd.concat({str(self.datasets_sample_2): self.a2.aggrf}, names=factors, axis=1)],
                               axis=1).reorder_levels([1, 2, 3, 4, 0], axis=1)

        return self


def kemeny_aggregation_cvxpy(dr, solver=cp.GUROBI, consensus_kind="total_order", solver_opts=None):
    """
    Returns the median consensus according to the symmetric distance. See Hornik and Meyer (2007).
    based on cvxpy module
    consensus_kind =
        weak_order: totality and transitivity
        total_order: antisymmetry and transitivity (note that we do not care about i == j)
        strict_order: antisymmetry and transitivity
    """
    ms = dr2mat(dr, kind="preference")

    # this function cannot handle missgin values
    assert not np.isnan(ms).any()

    nv, na, _ = ms.shape    # num_voters, num_alternatives, num_alternatives
    c = np.sum(ms, axis=0)  # c matrix in the paper, with diagonal 0
    median = cp.Variable(shape=(na, na), boolean=True)

    # --- run the optimization, which returns an incidence matrix
    p = cp.Problem(
        cp.Maximize(cp.sum(cp.multiply(c, median))),
        get_constraints(median, consensus_kind)
    )
    p.solve(solver=solver, solver_opts=solver_opts)

    return mat2rf(np.array(median.value, dtype=int), alternatives=dr.index, kind="incidence")


def kemeny_aggregation_mip(dr, solver=mip.GRB, consensus_kind="total_order"):
    """
    Returns the median consensus according to the symmetric distance. See Hornik and Meyer (2007).
    based on MIP module
    consensus_kind =
        weak_order: totality and transitivity
        total_order: antisymmetry and transitivity (note that we do not care about i == j)
        strict_order: antisymmetry and transitivity
    """

    ms = dr2mat(dr, kind="preference")

    # this function cannot handle missgin values
    assert not np.isnan(ms).any()

    nv, na, _ = ms.shape
    c = np.sum(ms, axis=0)  # c matrix in the paper, with diagonal 0

    # --- optimization problem
    model = mip.Model(sense=mip.MAXIMIZE, solver_name=solver)
    model.verbose = 0
    model.sol_pool_size = 1000
    median = np.array([[model.add_var(var_type=mip.BINARY) for _ in range(na)] for _ in range(na)])
    model.objective = mip.maximize(mip.xsum(median[i, j] * c[i, j] for i, j in product(range(na), repeat=2)))

    # --- run the optimization, which returns an incidence matrix
    for cs in get_constraints(median, consensus_kind):
        model += cs

    status = model.optimize()
    return mat2rf(np.array([[md.x for md in median_row] for median_row in median], dtype=int), alternatives=dr.index)


def kemeny_aggregation_gurobi(dr, consensus_kind="total_order", **solver_params):
    """
        Returns the median consensus according to the symmetric distance. See Hornik and Meyer (2007).
        Gurobi-specific
        consensus_kind =
            weak_order: totality and transitivity
            total_order: antisymmetry and transitivity (note that we do not care about i == j)
            strict_order: antisymmetry and transitivity
        """

    ms = dr2mat(dr, kind="preference")

    # this function cannot handle missgin values
    assert not np.isnan(ms).any()

    nv, na, _ = ms.shape
    c = np.sum(ms, axis=0)  # c matrix in the paper, with diagonal 0

    # --- optimization problem
    model = gp.Model()
    model.setParam("OutputFlag", 0)
    median = model.addMVar(shape=(na, na), vtype=gp.GRB.BINARY)

    for k, v in solver_params.items():
        if k == "Seed":
            model.setParam(k, v)
        elif k == "Start":
            median.setAttr(k, v)
        else:
            raise ValueError(f"Parameter {k} with value {v} is not valid.")

    model.setObjective(gp.quicksum(gp.quicksum(median * c)), gp.GRB.MAXIMIZE)
    for cs in get_constraints(median, consensus_kind):
        model.addConstr(cs)

    model.update()
    model.optimize()

    return mat2rf(median.X, alternatives=dr.index, kind="incidence")


def kemeny_aggregation_gurobi_ties(rf, **solver_params):
    """
    Based on the optimization problem defined in:
    "A new binary programming formulation and social choice property for Kemeny rank aggregation" - Yoo 2021

    With support for missing values, cost c adapted from Yoo (2020), itself adapted from Moreno-Centeno and Escobedo (2016)
    """

    ms = dr2mat(rf, kind="ranking")
    nv, na, _ = ms.shape

    # If some rank is missing, use the c from Yoo (2021)
    if np.isnan(ms).any():
        # weight = how many alternatives were ranked for voter i: minimum number of nan columns in ms[i]
        w = na - np.isnan(ms).sum(axis=1).min(axis=1)
        c = np.nansum([ms[i] / (w[i] * (w[i]-1)) for i in range(nv)], axis=0)
    else:
        c = np.sum(ms, axis=0)

    # --- optimization problem
    model = gp.Model()
    model.setParam("OutputFlag", 0)
    median = model.addMVar(shape=(na, na), vtype=gp.GRB.BINARY)

    for k, v in solver_params.items():
        if k == "Seed":
            model.setParam(k, v)
        elif k == "Start":
            median.setAttr(k, v)
        else:
            raise ValueError(f"Parameter {k} with value {v} is not a valid input parameter")

    model.setObjective(gp.quicksum(gp.quicksum((2 * median - np.ones((na, na))) * c)), gp.GRB.MAXIMIZE)
    for cs in get_constraints(median, "yoo_weak_order"):
        model.addConstr(cs)

    model.update()
    model.optimize()

    return mat2rf(median.X, alternatives=rf.index)


# --- Optima of Kemeny aggregation


def get_distinct(optima: list[pd.Series], V: Iterable):
    """
    optima: list of optima as pd.Series
    V: iterable of objects/alternatives
    """
    assert len(optima) > 0
    return pd.DataFrame([pd.Series(opt, index=V) for opt in set(tuple(opt) for opt in optima)]).T


def kemeny_optima_exhaustive(dr, consensus_kind="total_order", **kwargs):
    """
    strategy =
        exhaustive: search through all possible permutations
        seed: random seed. Requires n_tries
        voters: use voters' rankings as initial guesses
    """
    try:
        assert consensus_kind == "total_order"
    except AssertionError:
        raise ValueError("The only consensus_kind implemented is 'total_order'")

    na = dr.shape[0]

    time_start = time.time()
    kds = defaultdict(lambda: [])
    for sigma in list(permutations(range(na))):
        kds[d_kendall_set(sigma, dr)].append(pd.Series(sigma, index=dr.index))

    optima = get_distinct(kds[min(kds)], dr.index)
    obj_values = [min(kds) for _ in range(len(optima))]
    runtime = time.time() - time_start

    return optima, obj_values, runtime


def kemeny_optima_seed(dr, seeds, consensus_kind="total_order", **kwargs):
    time_start = time.time()
    optima, obj_values = [], []
    for seed in seeds:
        optima.append(kemeny_aggregation_gurobi(dr, consensus_kind, Seed=seed))
        obj_values.append(d_kendall_set(optima[-1], dr))
    runtime = time.time() - time_start
    return get_distinct(optima, dr.index), obj_values, runtime


def kemeny_optima_voters(dr, consensus_kind="total_order", **kwargs):
    time_start = time.time()
    optima, obj_values = [], []
    for voter in dr:
        optima.append(kemeny_aggregation_gurobi(dr, consensus_kind, Start=rf2mat(dr[voter], "domination")))
        obj_values.append(d_kendall_set(optima[-1], dr))
    runtime = time.time() - time_start
    return get_distinct(optima, dr.index), obj_values, runtime


def kemeny_optima_ties_voters(dr, verbose=False, **kwargs):
    time_start = time.time()
    optima, obj_values = [], []

    iterable = tqdm(dr.columns) if verbose else dr.columns
    for voter in iterable:
        optima.append(kemeny_aggregation_gurobi_ties(dr, Start=rf2mat(dr[voter], "domination")))
        obj_values.append(d_kendall_set(optima[-1], dr))
    runtime = time.time() - time_start
    return get_distinct(optima, dr.index), obj_values, runtime


def kemeny_optima(dr, strategy, *args, **kwargs):
    assert strategy in ["exhaustive", "seed", "voters"]
    if strategy == "exhaustive":
        return kemeny_optima_exhaustive(dr, **kwargs)
    elif strategy == "seed":
        return kemeny_optima_seed(dr, *args, **kwargs)
    elif strategy == "voters":
        return kemeny_optima_voters(dr, **kwargs)
