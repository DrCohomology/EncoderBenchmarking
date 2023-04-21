"""
Comparison and ranking utilities
"""

import cvxpy as cp
import gurobipy as gp
import mip
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import time
import warnings

from collections import defaultdict
from functools import reduce
from itertools import product, permutations
from scipy.stats import kendalltau, t, iqr
from tqdm import tqdm
from typing import Iterable


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
    Tests for totality AND riflexivity of a relation.
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
    Agreement is measured with intersection over union
    """
    o1 = col1.min() if best else col1.max()
    o2 = col2.min() if best else col2.max()
    b1 = set(col1[col1 == o1].index)
    b2 = set(col2[col2 == o2].index)
    return len(b1.intersection(b2)) / len(b1.union(b2))


class Aggregator(object):

    def __init__(self, df: pd.DataFrame, scoring: str = None, model: str = None):
        """
        'scoring' and 'model' are resp. the scoring and model we aggregate ranks on.
        In other terms, is one of them is None, we will consider it as a separate

        """

        if "scoring" not in df.columns or "model" not in df.columns:
            raise ValueError("The input DataFrame should have columns 'scoring' and 'model'.")
        if scoring and scoring not in df.scoring.unique():
            raise ValueError(f"{scoring} is not a valid scoring function. Valid scorings are {list(df.scoring.unique())}")
        if model and model not in df.model.unique():
            raise ValueError(f"{model} is not a valid model. Valid models are {list(df.model.unique())}")

        self.scoring = scoring
        self.model = model
        self.original_df = df

        if scoring and model:
            self.df = self.original_df.loc[(self.original_df.scoring == self.scoring) & (self.original_df.model == self.model)][
                ["dataset", "encoder", "cv_score"]]
        elif scoring:
            self.df = self.original_df.loc[(df.scoring == self.scoring)][["model", "dataset", "encoder", "cv_score"]]
        elif model:
            self.df = self.original_df.loc[(df.model == self.model)][["scoring", "dataset", "encoder", "cv_score"]]
        else:
            self.df = self.original_df


        self.supported_strategies = [
            "all",
            "mean rank",
            "median rank",
            "numbest rank",
            "numworst rank",
            # "numKbest rank",
            # "hornik-meyer rank",
            "nemenyi rank",  # transitivity cannot be guaranteed
            "mean performance",
            "median performance",
            "thrbest performance",
            'rescaled mean performance'
        ]

        self.plain_domination_matrices = []
        self.ttest_domination_matrices = []
        self.ttest_pvals = []  # of the pairwise t-tests
        self.bayesian_equality_matrices = []  # according to Bayesian statistic in Benavoli 2017
        self.bayesian_domination_matrices = []

        self.final_ranks = {}

    # --- Utilities

    def _get_e2i(self):
        self.e2i = {E: i for (i, E) in enumerate(self.df.encoder.unique())}
        self.i2e = {v: k for (k, v) in self.e2i.items()}

    def _filter_kwargs(self, kwargs):
        """
        Filters the passed-in kwargs to ensue that each method of self gets only the correct ones.
        KNOWN ISSUE: if two methods have a parameter with the same name, both get the same value of such parameter.
        """
        fkw = {}
        for attr in (x for x in dir(self) if callable(getattr(self, x))):
            fkw[attr] = {}
            try:
                for k, v in kwargs.items():
                    if k in getattr(self, attr).__code__.co_varnames:
                        fkw[attr][k] = v
            except AttributeError:
                del fkw[attr]
        return fkw

    # --- Domination matrices

    def _get_plain_dommat(self, missing_evaluations="ignore"):

        # print(f"_get_plain_dommat: missing evaluations: {missing_evaluations}. If ignore, missing is interpreted as "
        #       f"equivalent.")

        n = len(self.e2i)
        for dataset in self.df.dataset.unique():
            df_ = self.df.query("dataset == @dataset")
            R = - np.ones((n, n))
            for E1, i1 in self.e2i.items():
                for E2, i2 in self.e2i.items():
                    if i1 > i2:
                        continue
                    cv1 = df_[df_.encoder == E1].cv_score.to_numpy()
                    cv2 = df_[df_.encoder == E2].cv_score.to_numpy()

                    if len(cv1) * len(cv2) == 0:
                        if missing_evaluations == "ignore":
                            R[i1, i2] = R[i2, i1] = 1
                            continue
                        else:
                            raise ValueError(f"One encoder of {E1}, {E2} has no evaluations.")
                    elif len(cv1) != len(cv2):
                        if missing_evaluations == "ignore":
                            R[i1, i2] = R[i2, i1] = 1
                            continue
                        else:
                            raise ValueError(f"Different number of evaluations for {E1} {len(cv1)} and {E2} {len(cv2)}.")

                    # E1, E2 same
                    if np.nanmean(cv1) == np.nanmean(cv2):
                        R[i1, i2] = R[i2, i1] = 1
                    # E1 better than E2
                    elif np.nanmean(cv1) > np.nanmean(cv2):
                        R[i1, i2] = 1
                        R[i2, i1] = 0
                    # E2 better than E1
                    else:
                        R[i1, i2] = 0
                        R[i2, i1] = 1
            try:
                assert (R != -1).all()
            except AssertionError:
                raise Exception(
                    f"Dataset {dataset}: when building the domination matrix some entries were left uninitialized ")

            self.plain_domination_matrices.append(R)

    def _get_ttest_dommat(self, alpha=0.1, corrected=True):
        """
        Run pairwise comparisons of cross-validated performances (hence the correction term).
        A domination matrix is M[i1, i2] = 1 iff (i1 is better than i2) or (the comparison is undecided ie pval>alpha).
        Update self.ttest_domination_matrices with a domination matrix for each dataset.
        """
        n = len(self.e2i)
        for dataset in self.df.dataset.unique():
            df_ = self.df.loc[self.df.dataset == dataset]
            R = - np.ones((n, n))
            pvals = -np.ones((n, n))
            for E1, i1 in self.e2i.items():
                for E2, i2 in self.e2i.items():
                    if i1 > i2:
                        continue
                    cv1 = df_[df_.encoder == E1].cv_score.to_numpy()
                    cv2 = df_[df_.encoder == E2].cv_score.to_numpy()

                    if len(cv1) * len(cv2) == 0:
                        raise ValueError("One of the two measurements is null for", E1, E2)
                    elif len(cv1) != len(cv2):
                        raise ValueError("The two measurements have different size", E1, E2)

                    # frequentist analysis
                    comp, p = compare_with_ttest(cv1, cv2, alpha=alpha, corrected=corrected)
                    pvals[i1, i2] = p
                    pvals[i2, i1] = p
                    # print(dataset, E1, E2, comp, p)

                    # E1 and E2 not comparable
                    if comp == 0:
                        R[i1, i2] = 1
                        R[i2, i1] = 1
                    # E1 > E2
                    elif comp == 1:
                        R[i1, i2] = 1
                        R[i2, i1] = 0
                    # E1 < E2
                    elif comp == 2:
                        R[i1, i2] = 0
                        R[i2, i1] = 1
                    else:
                        raise ValueError(f"Something went very wrong when comparing with t-test: {E1}, {E2}, comp={comp}")

            if (R == -1).sum() > 0:
                raise Exception(f"Dataset {dataset}: when building the domination matrix some entries were left uninitialized ")

            self.ttest_domination_matrices.append(R)
            self.ttest_pvals.append(pvals)

    def _get_bayesian_mat(self, rope=0.01):
        n = len(self.e2i)
        for dataset in self.df.dataset.unique():
            df_ = self.df.loc[self.df.dataset == dataset]
            Beq = - np.ones((n, n))  # Beq[i, j] = prob(true difference in average performance is less than rope)
            Bdom = -np.ones((n, n))  # Beq[i, j] = prob(true difference in average performance is NOT ZERO): as the rope breaks things, we ignore it here
            for E1, i1 in self.e2i.items():
                for E2, i2 in self.e2i.items():
                    if i1 > i2:
                        continue

                    cv1 = df_[df_.encoder == E1].cv_score.to_numpy()
                    cv2 = df_[df_.encoder == E2].cv_score.to_numpy()

                    if len(cv1) * len(cv2) == 0:
                        raise ValueError("One of the two measurements is null for", E1, E2)
                    elif len(cv1) != len(cv2):
                        raise ValueError("The two measurements have different size", E1, E2)

                    # ---- bayesian analysis, Benavoli 2017, corrected
                    if i1 == i2:
                        Beq[i1, i2] = 1
                        Bdom[i1, i2] = 0
                        continue

                    k = len(cv1)  # ATTENTION! In utils.ttest, this parameter is called n
                    average_difference = (cv1 - cv2).mean()
                    corrected_variance = (1 / k + 1 / (k - 1)) * (
                                cv1 - cv2).var()  # test/train is 1/(n-1) for n folds of cross validation
                    if corrected_variance == 0:
                        """
                        If variance == 0, the t-student collapses into a Dirac-delta, and Beq[i1, i2] = prob(E1 == E2) 
                        is 1 if the difference in averages is less than the rope, 0 otherwise
                        """
                        if np.abs(average_difference) <= rope:
                            Beq[i1, i2] = Beq[i2, i1] = 1
                            Bdom[i1, i2] = Bdom[i2, i1] = 0
                        else:
                            Beq[i1, i2] = Beq[i2, i1] = 0
                            Bdom[i1, i2] = 1 if average_difference > 0 else 0
                            Bdom[i2, i1] = 1 - Bdom[i1, i2]
                        continue
                    distribution_true_difference = t(df=k - 1, loc=average_difference, scale=corrected_variance)
                    Beq[i1, i2] = Beq[i2, i1] = distribution_true_difference.cdf(
                        rope) - distribution_true_difference.cdf(-rope)
                    Bdom[i1, i2] = 1 - distribution_true_difference.cdf(rope)
                    Bdom[i2, i1] = distribution_true_difference.cdf(-rope)
            if (Beq == -1).sum() > 0:
                raise Exception(
                    f"Dataset {dataset}: when building the equality probability matrix some entries were left uninitialized ")
            if (Bdom == -1).sum() > 0:
                raise Exception(
                    f"Dataset {dataset}: when building the Bayesian domination matrix some entries were left uninitialized ")

            self.bayesian_equality_matrices.append(Beq)
            self.bayesian_domination_matrices.append(Bdom)

    def _get_domination_matrices(self, how="all", **kwargs):
        """
        Gets the domination matrices for each dataset listed in a dataframe
        """

        # dollar[algorithm, dataset] = position in weak ordering
        try:
            self.e2i
        except AttributeError:
            self._get_e2i()

        fkw = self._filter_kwargs(kwargs)
        if how in ("plain", "all"):
            self._get_plain_dommat()
        if how in ("ttest", "all"):
            self._get_ttest_dommat(**fkw["_get_ttest_dommat"])
        if how in ("bayes", "all"):
            self._get_bayesian_mat(**fkw["_get_bayesian_mat"])

    # --- Rank function utilities

    def _get_rank_from_matrix(self, M):
        """
        Based on the results (in the paper) that two indices have the same dominance if and only if htey belong to the
        same tier.
        """
        # totality is not strictly necessary, transitivty is
        if not test_totality(M):
            print("M is not total, which can be due to an encoder missing evaluations for (dataset, scoring, model).")

        assert test_transitivity(M)

        scores = {
            E: M[i].sum() for (E, i) in self.e2i.items()
        }
        return self._get_rank_from_scores(scores, ascending=False)

    def _get_ranks(self, how="plain"):
        if how == "plain":
            self.ranks = [self._get_rank_from_matrix(R) for R in self.plain_domination_matrices]
        elif how == "ttest":
            self.ranks = [self._get_rank_from_matrix(R) for R in self.ttest_domination_matrices]
        else:
            raise ValueError("Invalid value of parameter 'how', ", how)

    @staticmethod
    def _get_rank_from_scores(scores, ascending=True):
        """
        If ascending is True, you assume that the lowest score corresponds to the lowest rank of 0.
        In other terms, the lower the score the better.
        This in the light of statistics computed on ranks, which give this sort of behaviour
        """

        c = 1 if ascending else -1
        order_map = {
            s: sorted(set(scores.values()), key=lambda x: c*x).index(s) for s in set(scores.values())
        }
        return dict(sorted([(E, order_map[s]) for (E, s) in scores.items()], key=lambda s: s[1]))

    # --- Rank function aggregation

    def _mean_rank_aggregation(self):
        scores = {
            E: np.mean([r[E] for r in self.ranks]) for E in self.e2i.keys()
        }
        self.final_ranks["mean rank"] = self.meanRank_rank = self._get_rank_from_scores(scores)

    def _median_rank_aggregation(self):
        scores = {
            E: np.median([r[E] for r in self.ranks]) for E in self.e2i.keys()
        }
        self.final_ranks["median rank"] = self.medianRank_rank = self._get_rank_from_scores(scores)

    def _numbest_rank_aggregation(self):
        scores = {
            E: sum(r[E] == 0 for r in self.ranks) for E in self.e2i.keys()
        }
        self.final_ranks["numbest rank"] = self.numbestRank_rank = self._get_rank_from_scores(scores, ascending=False)

    def _numworst_rank_aggregation(self):
        scores = {
            E: sum(r[E] == max(r.values()) for r in self.ranks) for E in self.e2i.keys()
        }
        self.final_ranks["numworst rank"] = self.numworstRank_rank = self._get_rank_from_scores(scores)

    def _numkbest_rank_aggregation(self, k):
        """Number of times an encoder is among the k-best CLASSES of encoders, i.e. not among the best k encoders"""
        scores = {
            E: sum(r[E] <= k-1 for r in self.ranks) for E in self.e2i.keys()
        }
        self.final_ranks[f"num{k}best rank"] = self._get_rank_from_scores(scores, ascending=False)
        self.__setattr__(f"num{k}bestRank_rank", self.final_ranks[f"num{k}best rank"])

    def _hornikmeyer_rank_aggregation(self, solver=cp.GLPK_MI, how="plain"):
        """Based on optimization problem formulated in Hornik and Meyer (2007). Takes the domination matrices and
        computes the centroid DM according to symmetric distance"""

        raise DeprecationWarning("Faster solvers are available.")

        if solver == cp.ECOS_BB:
            warnings.warn("ECOS_BB is deprecated as solver")

        if how == "plain":
            Rs = np.array(self.plain_domination_matrices)
        elif how == "ttest":
            Rs = np.array(self.ttest_domination_matrices)
        else:
            raise ValueError("Invalid value of parameter 'how', ", how)

        nR = len(self.e2i)
        centroid = cp.Variable(shape=(nR, nR), boolean=True)

        # formulate cost function and objective
        C = np.sum(2 * Rs - 1, axis=0)
        # as the sum has to be computed without the diagonal elements, we kill them in C (see Hornik and Meyer)
        for i in range(len(C)):
            C[i, i] = 0
        objective = cp.Maximize(cp.sum(cp.multiply(C, centroid)))

        # constraints
        totality = [
            centroid[i, j] + centroid[j, i] >= 1
            for i, j in product(range(nR), repeat=2) if i != j
        ]
        transitivity = [
            centroid[i, j] + centroid[j, k] - centroid[i, k] <= 1
            for i, j, k in product(range(nR), repeat=3) if i != j and j != k and i != k
        ]

        # problem
        prob = cp.Problem(objective, totality + transitivity)
        prob.solve(solver=solver)

        # get solution
        R = centroid.value.round()
        for i in range(len(R)):
            R[i, i] = 1

        self.final_ranks["hornik-meyer rank"] = self.hornikmeyer_rank = self._get_rank_from_matrix(R)

    def _nemenyi_rank_aggregation(self, alpha=0.05, how="plain"):
        """
        First: compare average ranks and build a domination matrix.
        Second: apply a mask for statistically significant difference using Nemenyi
        KNOWN ISSUES: Nemenyi is likely OK when considering rankings with ties, BUT not OK when considering rankings with
            different numbers of ranks involved.
            For instance, we could have for a dataset the rank [(A, B), C] and for another the ranking [A, B, C]:
            Nemenyi will just consider the average rank of A, B, C to determine their significance but not the total
            range of the rankings
        """
        # The domination matrices can be build with a ttest or without
        if how == "plain":
            dommat = self.plain_domination_matrices
        elif how == "ttest":
            dommat = self.ttest_domination_matrices
        else:
            raise ValueError("Invalid value of parameter 'how', ", how)

        # score of an encoder is the average rank
        scores = {
            E: np.mean([r[E] for r in self.ranks]) for E in self.e2i.keys()
        }
        domination_matrix = np.array([
            [s1 <= s2 for s2 in scores.values()]
            for s1 in scores.values()
        ]).astype(int)

        ranks_matrix = np.array([list(l.values())
                                 for l in [self._get_rank_from_matrix(R)
                                           for R in dommat]]).T
        statdiff_matrix = (sp.posthoc_nemenyi(ranks_matrix) < alpha).to_numpy()
        for i in range(len(statdiff_matrix)):
            statdiff_matrix[i, i] = 1

        self.testdom = domination_matrix

        # apply the statistically significant mask: only significant comparisons survive
        domination_matrix = domination_matrix * statdiff_matrix

        self.testdom2 = domination_matrix
        self.teststat = statdiff_matrix

        self.final_ranks["nemenyi rank"] = self.nemenyiRank_rank = self._get_rank_from_matrix(domination_matrix)

    # --- Performance aggregation

    def _mean_performance_aggregation(self):
        scores = {
            E: self.df.loc[self.df.encoder == E]["cv_score"].mean() for E in self.e2i.keys()
        }
        self.final_ranks["mean performance"] = self.meanPerformance_rank = self._get_rank_from_scores(scores, ascending=False)

    def _median_performance_aggregation(self):
        scores = {
            E: self.df.loc[self.df.encoder == E]["cv_score"].median() for E in self.e2i.keys()
        }
        self.final_ranks["median performance"] = self.medianPerformance_rank = self._get_rank_from_scores(scores, ascending=False)

    def _thresholdbest_performance_aggregation(self, th):
        best_performances = {
            dataset: self.df.loc[self.df.dataset == dataset]["cv_score"].max()
            for dataset in self.df.dataset.unique()
        }
        self.df[f"th{th}"] = self.df.dataset.map({k: th * v for k, v in best_performances.items()})

        scores = {
            E: (self.df.loc[self.df.encoder == E]["cv_score"] >= self.df.loc[self.df.encoder == E][f"th{th}"]).sum()
            for E in self.e2i.keys()
        }
        # missing support for changing name
        self.final_ranks[f"{th}bestPerformance rank"] = self._get_rank_from_scores(scores, ascending=False)
        self.__setattr__("thresholdbestPerformance_rank", self.final_ranks[f"{th}bestPerformance rank"])

    def _rescaled_mean_performance_aggregation(self):
        """
        First, take average across fold.
        Second, compute worst performance W.
        Third, compute IQR of performances.
        Fourth, final score[E] = (performance - W) / IQR
        """

        # find average performance for each encoder and dataset
        apf = pd.DataFrame(self.df.groupby(["encoder", "dataset"])["cv_score"].mean())

        # find worst performance per dataset
        wpd = self.df.groupby(["dataset"])["cv_score"].min()

        # find IQR of performances per dataset, if IQR == 0 -> constant -> IQR does not matter
        iqrd = self.df.groupby(["dataset"])["cv_score"].agg(iqr)
        iqrd[iqrd == 0] = 1

        temp = apf.join(wpd, how="left", rsuffix="_worst").join(iqrd, how="left", rsuffix="_iqr")
        temp["score"] = (temp["cv_score"] - temp["cv_score_worst"]) / (temp["cv_score_iqr"])
        scores = {
            E: temp.loc[E]["score"].mean() for E in self.e2i.keys()
        }

        self.final_ranks["rescaled mean performance"] = self.rescaledMeanPerformance_rank = \
            self._get_rank_from_scores(scores, ascending=False)

    # --- Aggregate

    def aggregate(self, strategy, skipped_strategies=tuple(), **kwargs):
        """
        If strategy=='all', all strategies expect those listed in skipped_strategies are considered
        """
        if strategy not in self.supported_strategies:
            raise ValueError(f"\"{strategy}\" is not a supported strategy. "
                             f"Supported strategies: {self.supported_strategies}")

        fkw = self._filter_kwargs(kwargs)
        self._get_e2i()

        if "rank" in strategy:
            try:
                self.ranks
            except AttributeError:
                self._get_domination_matrices(**fkw["_get_domination_matrices"])
                self._get_ranks(**fkw["_get_ranks"])

        # start_time = time.time()

        if strategy == "mean rank":
            self._mean_rank_aggregation()
        elif strategy == "median rank":
            self._median_rank_aggregation()
        elif strategy == "numbest rank":
            self._numbest_rank_aggregation()
        elif strategy == "numworst rank":
            self._numworst_rank_aggregation()
        # elif strategy == "numKbest rank":
        #     self._numkbest_rank_aggregation(**fkw["_numkbest_rank_aggregation"])
        # elif strategy == "hornik-meyer rank":
        #     self._hornikmeyer_rank_aggregation(**fkw["_hornikmeyer_rank_aggregation"])
        elif strategy == "nemenyi rank":
            warnings.warn("Nemenyi rank is not correctly implemented.")
            self._nemenyi_rank_aggregation(**fkw["_nemenyi_rank_aggregation"])
        elif strategy == "mean performance":
            self._mean_performance_aggregation()
        elif strategy == "median performance":
            self._median_performance_aggregation()
        elif strategy == "thrbest performance":
            self._thresholdbest_performance_aggregation(**fkw["_thresholdbest_performance_aggregation"])
        elif strategy == "rescaled mean performance":
            self._rescaled_mean_performance_aggregation()
        elif strategy == "all":
            # recursive call for every strategy
            for strat in set(self.supported_strategies) - set(skipped_strategies):
                if strat == "all":
                    continue
                self.aggregate(strat, **kwargs)
        else:
            raise ValueError(f"For some reason this went unnoticed, but {strategy} is not supported. "
                             f"Supported strategies are {self.supported_strategies}")

        # print(strategy, time.time()-start_time)


# ---  Basics


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
        True: lower score = better rank (for instance, if score is the result of a loss function or a rank itself)
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
        weak_order: totality and transitivity (riflexivity?)
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
    riflexivity = [
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
        return riflexivity + antisymmetry + transitivity
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
        "riflexivity": check([mat[i, i] == 1 for i in range(na)]),
        "antisymmetry": check([mat[i, j] + mat[j, i] == 1 for i, j in product(range(na), repeat=2) if i < j]),
        "transitivity": check([mat[i, j] + mat[j, k] - mat[i, k] <= 1
                               for i, j, k in product(range(na), repeat=3) if i != j != k != i]),
        "acyclicity": check([mat[i, j] - mat[k, j] - mat[i, k] >= -1
                            for i, j, k in product(range(na), repeat=3) if i != j != k != i]),
    }
    return [p for p, satisfied in properties.items() if satisfied]


# --- Aggregation


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

# --- Optima


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
