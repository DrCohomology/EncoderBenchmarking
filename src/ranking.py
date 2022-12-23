# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 10:46:35 2022

@author: federicom

Ranking strategies, from Nießl 2022
"""

import cvxpy as cp
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
import warnings
from collections import defaultdict
from importlib import reload
from itertools import product
from scipy.stats import kendalltau, t, iqr
from tqdm import tqdm

import src.utils as u
import src.results_concatenator as rc

reload(u)
reload(rc)

# TODO: plot with same font as document
"""
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\\usepackage{times}'
mpl.rcParams['text.latex.preamble'] = r'\\usepackage{nicefrac}'
mpl.rc('font', family='serif')
"""


# ---- Import and clean experimental results
experiment_name = "main6 results with tuning and 6-encoders"
df = rc.concatenate_results(experiment_name, clean=False)

# remove prebinned - giving problems
df = df.loc[~ df.encoder.isin(("PBTE0", "DTEM6", ""))]

ne = one = len(df.encoder.unique())
nd = ond = len(df.dataset.unique())
nm = len(df.model.unique())
ns = len(df.scoring.unique())
nsc = len(df.scaler.unique())
nf = len(df.fold.unique())

expected_iters_dataset = ne * nm * ns * nsc * nf
expected_iters_encoder = nd * nm * ns * nsc * nf
printme = True
if printme:
    for encoder in df.encoder.unique():
        print(f"{encoder:15s}", df.loc[df.encoder == encoder].shape[0] // nf, expected_iters_encoder // nf)

    for dataset in df.dataset.unique():
        print(f"{dataset:50s}", df.loc[df.dataset == dataset].shape[0] // nf, expected_iters_dataset // nf)

print("Removing incomplete datasets")
wasted = 0
for d in df.dataset.unique():
    sd = sum(df.dataset == d)
    if sd < expected_iters_dataset:
        df = df.loc[df.dataset != d]
        wasted += sd

ne = len(df.encoder.unique())
nd = len(df.dataset.unique())
nm = len(df.model.unique())
ns = len(df.scoring.unique())
nsc = len(df.scaler.unique())
nf = len(df.fold.unique())

expected_iters_encoder = nd * nm * ns * nsc * nf

print("Removing incomplete encoders")
for E in df.encoder.unique():
    if sum(df.encoder == E) < expected_iters_encoder:
        df = df.loc[df.encoder != E]

# completed datasets
new_datasets = set(df.dataset.unique()).intersection(u.LEFT_DATASETS)

print(f"Surviving datasets: {len(df.dataset.unique())} of {ond}")
print(f"Surviving encoders: {len(df.encoder.unique())} of {one}")
print(f"A total of {wasted} entries are not considered.")
print(f"The new completed datasets are: {new_datasets}")

finished_datasets_id = pd.Series(df.dataset.unique()).map(lambda d: defaultdict(lambda: 0, u.DATASETS)[d])
finished_datasets = pd.DataFrame((df.dataset.unique(), finished_datasets_id)).T.sort_values(1)
printme = False
if printme:
    print("---- These are the completed datasets ----")
    print(finished_datasets)


#%% Aggregator and first aggregation
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
            "hornik-meyer rank",
            # "nemenyi rank", # theoretical issues
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

    def _get_e2i(self):
        self.e2i = {E: i for (i, E) in enumerate(df.encoder.unique())}
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

    def _get_plain_dommat(self):
        n = len(self.e2i)
        for dataset in self.df.dataset.unique():
            df_ = self.df.loc[self.df.dataset == dataset]
            R = - np.ones((n, n))
            for E1, i1 in self.e2i.items():
                for E2, i2 in self.e2i.items():
                    if i1 > i2:
                        continue
                    cv1 = df_[df_.encoder == E1].cv_score.to_numpy()
                    cv2 = df_[df_.encoder == E2].cv_score.to_numpy()

                    if len(cv1) * len(cv2) == 0:
                        raise ValueError("One of the two measurements is null for ", E1, E2)
                    elif len(cv1) != len(cv2):
                        raise ValueError("The two measurements have different size ", E1, E2)

                    # E1, E2 same
                    if cv1.mean() == cv2.mean():
                        R[i1, i2] = R[i2, i1] = 1
                    # E1 better than E2
                    elif cv1.mean() > cv2.mean():
                        R[i1, i2] = 1
                        R[i2, i1] = 0
                    # E2 better than E1
                    else:
                        R[i1, i2] = 0
                        R[i2, i1] = 1

            if (R == -1).sum() > 0:
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
                    comp, p = u.compare_with_ttest(cv1, cv2, alpha=alpha, corrected=corrected)
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

    def _get_rank_from_matrix(self, M):
        """
        Based on the results (in the paper) that two indices have the same dominance if and only if htey belong to the
        same tier.
        """
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
    def _get_rank_from_scores(self, scores, ascending=True):
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

        ranks_matrix = np.array([
            list(l.values())
            for l in [self._get_rank_from_matrix(R)
                      for R in dommat]
        ]).T
        statdiff_matrix = (sp.posthoc_nemenyi(ranks_matrix) < alpha).to_numpy()
        for i in range(len(statdiff_matrix)):
            statdiff_matrix[i, i] = 1

        self.testdom = domination_matrix

        # apply the statistically significant mask: only significant comparisons survive
        domination_matrix = domination_matrix * statdiff_matrix

        self.testdom2 = domination_matrix
        self.teststat = statdiff_matrix


        self.final_ranks["nemenyi rank"] = self.nemenyiRank_rank = self._get_rank_from_matrix(domination_matrix)

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
        elif strategy == "hornik-meyer rank":
            self._hornikmeyer_rank_aggregation(**fkw["_hornikmeyer_rank_aggregation"])
        # elif strategy == "nemenyi rank":
        #     warnings.warn("Nemenyi rank is not correctly implemented.")
            # self._nemenyi_rank_aggregation(**fkw["_nemenyi_rank_aggregation"])
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

run = False
if run:
    scorings = df.scoring.unique()
    models = df.model.unique()
    rank = {}
    for scoring, model in tqdm(list(product(scorings, models))):
        a = Aggregator(df, scoring, model)
        a.aggregate(strategy="all", how="plain", k=5, alpha=0.1, th=0.95, solver=cp.GLPK_MI)
        rank[(scoring, model)] = a

    # Aggregate the final ranks into a single dataframe

    scorings = df.scoring.unique()
    models = df.model.unique()
    dict_ranks = {}
    for model, scoring in product(models, scorings):
        for strategy, final_rank in rank[(scoring, model)].final_ranks.items():
            dict_ranks[model, scoring, strategy] = dict(sorted(final_rank.items(), key=lambda x: x[0]))

    df_ranks = pd.DataFrame(dict_ranks)
    print("Done!")

save = False
if save:
    df_ranks.to_csv(os.path.join(u.RESULT_FOLDER, "df_ranks.csv"))

load = False
if load:
    df_ranks = pd.read_csv(os.path.join(u.RESULT_FOLDER, "df_ranks.csv"),
                           header=[0, 1, 2],
                           index_col=0
                           )


#%% Correlations - plot of gigatable with all of the correlations: choose column order
"""
Interpretation: models dominate: different model, different scene is seen. performance metrics and RAS still matter 
but not as much. I can infer this because the models are CLEARLY visible from the high-correlation squares. 
Put strategy in first and no clear high-correlation squares are visible: no onsistency can be expected by using the same RAS
Same for scoring
"""

"""
IMPORTANT UPDATE TO KENDALLTAU: if the two rankings have the same number of classes, use variant b rather than variant c
"""

"""
Necessary updates: how different are the values? ie, are 0.1 and 0.2 "as close as" 0.9 and 1 are? How do I show this?
Also, how do I quantify the effect of a variable? You can assess that from the heatmap, but how can we make that a number?
I'd say the best way is to fix the variable and let the rest change. 
In other terms, we consider this asd a clustering problem: 
1. cluster by model
2. hopefully, pairs with same model are way more similar than pairse with different models
3. Compute average within-class correlation (average distance within the class) and average out-of-class correlation
4. high WCC -> homogeneous rankings; low OCC -> distinct rankings
5. We can then either compute the average for every class, or boxplots or something  
"""

def reverse_rank(R: pd.Series):
    return R.max() - R

corrsmat = np.zeros((len(df_ranks.columns), len(df_ranks.columns)))
for (i1, c1), (i2, c2) in product(enumerate(df_ranks.columns), repeat=2):
    # if df_ranks[c1].max() == df_ranks[c2].max():
    #     variant = 'b'
    # else:
    #     variant = 'c'
    variant = 'b'

    corrsmat[i1, i2] = kendalltau(df_ranks[c1], df_ranks[c2], variant=variant).correlation

# rescale to [-1, 1] the base is that tau_c(R, -R) = tau_c(R, R)
rescale = False
if rescale:
    corrsmat /= corrsmat.max(1)

corrs = pd.DataFrame(corrsmat, index=df_ranks.columns, columns=df_ranks.columns)
corrs.index.rename(["model", "scoring", "strategy"], inplace=True)
corrs.columns.rename(["model", "scoring", "strategy"], inplace=True)

order = ["model", "scoring", "strategy"]
corrs = corrs.reorder_levels(order, axis=0).reorder_levels(order, axis=1)
corrs = corrs.loc[sorted(corrs.index), sorted(corrs.columns)]

# num5best does not really work as-is: the reason is that my rankings all start from 0 and end up wherever
drop = [col for col in corrs.columns if "num5best rank" in col]
corrs.drop(index=drop, columns=drop, inplace=True)

# plot
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
fig.suptitle(f"Kendall's tau correlation between consensus rankings, {order}")
sns.heatmap(corrs, annot=False, square=True, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xlabel(order)
ax.set_ylabel(order)
plt.tight_layout()
plt.show()

#%% cluster the experiments according to their correlation
"""
1. rescale corrsmat
2. turn corrsmat into distmat
"""
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

def get_acronym_scoring(scoring):
    map = {
        "roc_auc_score": "ROC",
        "accuracy_score": "acc",
        "f1_score": "f1"
    }
    return map[scoring]

def get_acronym_strategy(strategy):
    map = {
        'rescaled mean performance': "p.rm",
        'median rank': "r.med",
        '0.95bestPerformance rank': "p.095b",
        'hornik-meyer rank': "r.hm",
        'mean rank': "r.m",
        'mean performance': "p.m",
        'numworst rank': "r.w",
        'median performance': "p.med",
        'numbest rank': "r.b"
    }
    return map[strategy]


# clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete")
for eps in [0.1, 0.15, 0.2, 0.3, 1]:
    clustering = DBSCAN(metric="precomputed", eps=eps)
    # clustering = KMeans(metric)

    # prettify the index
    newindex = corrs.reset_index()[["model", "scoring", "strategy"]]
    newindex.model = newindex.model.apply(lambda x: u.get_acronym(x, underscore=False))
    newindex.scoring = newindex.scoring.apply(lambda x: get_acronym_scoring(x))
    newindex.strategy = newindex.strategy.apply(lambda x: get_acronym_strategy(x))
    newindex = newindex.apply(lambda x: '_'.join(x), axis=1)

    newcorrs = corrs.copy()
    newcorrs.index = newindex
    newcorrs.columns = newindex

    distmat = 1 - (corrsmat / corrsmat.max(1) + 1) / 2
    clusters = dict(sorted(zip(newindex, clustering.fit_predict(distmat)), key=lambda x: x[1]))

    # reorder according to cluster
    newcorrs = newcorrs.loc[clusters.keys(), clusters.keys()]

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    fig.suptitle(f"Kendall's tau correlation between consensus rankings, {order}, eps = {eps}, {len(set(clusters.values()))} clusters")
    sns.heatmap(newcorrs, annot=False, square=True, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xlabel(order)
    ax.set_ylabel(order)
    plt.tight_layout()
    plt.savefig(os.path.join(u.RESULT_FOLDER, f"heatmap_eps={eps}.png"))


#%% Compute WCC and OCC for every model

strategies = list(a.final_ranks.keys())

pars = ["model", "scoring", "strategy"]
par_lists = [models, scorings, strategies]
# The important thing is that the considered par comes in first position. The rest is averaged so it is not a problem
temp_orders = [
    ["model", "scoring", "strategy"],
    ["scoring", "strategy", "model"],
    ["strategy", "model", "scoring"]
]
average_wcc = {}
average_occ = {}
for par, par_list, temp_order in zip(pars, par_lists, temp_orders):
    corrs3 = corrs.reorder_levels(temp_order, axis=0).reorder_levels(temp_order, axis=1)
    corrs3 = corrs3.loc[sorted(corrs3.index), sorted(corrs3.columns)]
    wccs = {}
    occs = {}
    for par_val in par_list:
        wccs[par_val] = corrs3.loc[par_val, par_val].mean().mean()
        occs[par_val] = corrs3.loc[par_val, [m for m in par_list if m != par_val]].mean().mean()
    average_wcc[par] = np.mean(list(wccs.values()))
    average_occ[par] = np.mean(list(occs.values()))

importance = {
    p: wcc-occ for (p, wcc), (_, occ) in zip(average_wcc.items(), average_occ.items())
}

"""
The obtained importances are 
model: 0.216
strategy: 0.14
scoring: 0.01
This means that overall, the rankings with fixed model are way more similar to other rankings with the same 
model as they are to rankings with different model. 
Instead, rankings with the same scoring are basically as similar as rankings with different scorings. 
model matters because if you fix it you get very different results
scoring matters because no matter what you choose, you will get a different result
What does "matter" mean? We could say that a parameter matters if results change a lot with it (occ), or because it 
is very clear the distinction from value of par to value of par (wcc). 
"""

for p, imp in importance.items():
    print(f"{p:10} {imp:02f}")


#%% What does it mean to have two rankings with 0.9 correlation?

"""
What I have done here is to take a pair of rankings that gave a certain correlation and chechking their differences. 
This requires some more finesse tho, as the two 0.8 ranks look way more similar than the 0.9 ranks
"""
example_indices = {}
example_rankings = {}
for desired_corr in [0, 0.5, 0.9, 0.98]:
    dc = corrs[np.abs(corrs - desired_corr) < 0.01].stack(level=[0, 1, 2])
    if len(dc) == 0:
        continue
    m1, sc1, st1, m2, sc2, st2 = dc.index[1]
    example_indices[desired_corr] = [(m1, sc1, st1), (m2, sc2, st2)]
    example_rankings[desired_corr] = df_ranks[example_indices[desired_corr]]

dc = 0
r = example_rankings[dc]
d1 = defaultdict(lambda: tuple('-'), {k: tuple(v) for k, v in r.groupby(r.columns[0]).groups.items()})
d2 = defaultdict(lambda: tuple('-'), {k: tuple(v) for k, v in r.groupby(r.columns[1]).groups.items()})

for k in set(d1.keys()).union(d2.keys()):
    print(f"{k}")
    print(f"{str(d2[k]).strip('()'):20}|{str(d1[k]).strip('()')}")


#%% Agreement on best class - weighted symmetric difference is 1-intersection_over_union

def agreement(col1: pd.Series, col2: pd.Series, best: bool = True):
    """
    Agreement is measured with intersection over union
    """
    o1 = col1.min() if best else col1.max()
    o2 = col2.min() if best else col2.max()
    b1 = set(col1[col1 == o1].index)
    b2 = set(col2[col2 == o2].index)
    return len(b1.intersection(b2)) / len(b1.union(b2))

best = True

corrsmat = np.zeros((len(df_ranks.columns), len(df_ranks.columns)))
for (i1, c1), (i2, c2) in product(enumerate(df_ranks.columns), repeat=2):
    corrsmat[i1, i2] = agreement(df_ranks[c1], df_ranks[c2], best=best)

corrs = pd.DataFrame(corrsmat, index=df_ranks.columns, columns=df_ranks.columns)
corrs.index.rename(["model", "scoring", "strategy"], inplace=True)
corrs.columns.rename(["model", "scoring", "strategy"], inplace=True)

order = ["model", "scoring", "strategy"]
corrs = corrs.reorder_levels(order, axis=0).reorder_levels(order, axis=1)
corrs = corrs.loc[sorted(corrs.index), sorted(corrs.columns)]

# num5best does not really work as-is: the reason is that my rankings all start from 0 and end up wherever
drop = [col for col in corrs.columns if "num5best rank" in col]
corrs.drop(index=drop, columns =drop, inplace=True)

# fix model and metric
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
fig.suptitle(f"Agreement on {'best' if best else 'worst'} encoder, {order}, average: {corrs.mean().mean():.02f}")
sns.heatmap(corrs, annot=False, square=True, cmap="Reds")
ax.set_xlabel(order)
ax.set_ylabel(order)
plt.tight_layout()
plt.show()

#%% As there seems to be agreement on the worst ones, let's see what they are
import json

worst_encoders = {}
for i, c in enumerate(df_ranks.columns):
    worst_encoders[c] = df_ranks[df_ranks[c] == df_ranks[c].max()].index

occurrences = defaultdict(lambda: 0)
for k, v in worst_encoders.items():
    for enc in v:
        occurrences[enc] += 1

print(json.dumps(occurrences, indent=4))

#%% Compute ranks for sample size correlation
"""
get 2 non overlapping samples of size sample_size from the dataset set
save both ranks as list in ranks2[sample_size]
"""

def sample_non_overlapping(S, n_samples, size, seed=1444):
    if n_samples * size > len(S):
        raise ValueError("Not enough elements in S.")

    np.random.seed(seed)

    S_ = set(S)
    out = []
    for _ in range(n_samples):
        Snew = set(np.random.choice(list(S_), size, replace=False))
        out.append(Snew)
        S_ = S_ - Snew
    np.random.seed(np.random.randint(0, 10))
    return out

scorings = df.scoring.unique()
models = df.model.unique()

sample_sizes = [2, 5, 10, 14]
repetitions = 5
# seed = 1444
seed = 1230123
run = False
if run:
    chronology_sampled_datasets2 = {}
    ranks2 = {}
    for sample_size in tqdm(sample_sizes):
        seed += 1
        ranks2[sample_size] = []
        chronology_sampled_datasets2[sample_size] = []
        # for every repetition, get a pair of non overlapping samples of datasets and save the consensuses obtained from each
        for rep in tqdm(range(repetitions)):
            seed += 1
            sampled_datasets = sample_non_overlapping(df.dataset.unique(), 2, sample_size, seed=seed)
            chronology_sampled_datasets2[sample_size].append(sampled_datasets)
            # for both samples of datasets, get all consensuses
            ranks_both_samples = []
            for sampdat in sampled_datasets:
                df1 = df.loc[df.dataset.isin(sampdat)]
                dict_ranks2 = {}
                for scoring, model in product(scorings, models):
                    a1 = Aggregator(df1, scoring, model)
                    a1.aggregate(strategy="all", skipped_strategies=("numKbest rank",),
                                 how="plain", k=5, alpha=0.1, th=0.95, solver=cp.GLPK_MI)
                    # a1.aggregate(strategy="hornik-meyer rank", how="plain", k=5, alpha=0.1, th=0.95, solver=cp.GLPK_MI)
                    for strategy, final_rank in a1.final_ranks.items():
                        dict_ranks2[model, scoring, strategy] = dict(sorted(final_rank.items(), key=lambda x: x[0]))

                ranks_both_samples.append(pd.DataFrame(dict_ranks2))

            ranks2[sample_size].append(ranks_both_samples)
    print("Done building ranks!")

save = False
if save:
    folder = os.path.join(u.RESULT_FOLDER, "Stability on dataset sample size for non-overlapping datasets iteration 3")
    for sample_size, l in ranks2.items():
        for rep, (r0, r1) in enumerate(l):
            r0.to_csv(os.path.join(folder, f"ranks_ss{sample_size}_rep{rep}_dat0.csv"))
            r1.to_csv(os.path.join(folder, f"ranks_ss{sample_size}_rep{rep}_dat1.csv"))
    print("Saved!")

#%% Load sample size Correlations

import re
load = False
if load:
    folder = os.path.join(u.RESULT_FOLDER, "Stability on dataset sample size for non-overlapping datasets iteration 1")
    repetitions = 10
    ranks2 = defaultdict(lambda: [[] for _ in list(range(repetitions))])
    for filename in glob.glob(os.path.join(folder, "*.csv")):
        ss, rep, dat = [int(x) for x in re.search(r"ranks_ss(\d+)_rep(\d+)_dat(\d+)", filename).groups()]
        temp = pd.read_csv(filename,
                           header=[0, 1, 2],
                           index_col=0
                           )
        temp.columns.name = ("model", "scoring", "strategy")
        # if we have already seen a dataset, just append it. If not, do stuff
        ranks2[ss][rep].append(temp)

    print("Done loading!")

    del filename, folder, ss, rep, temp

#%% Compute and plot sample size Correlations

try:
    strategies
except:
    strategies = ranks2[5][0][0].columns.get_level_values(2).unique()

sample_sizes = [2, 5, 10, 15]
repetitions = 10

"""
For every pair of ranks in ranks2[sample_size][rep], 
    compute their average correlation and save it in sample_corr2
    compute their standard deviation and save it in sample_std2
    compute the correlation for each (dataset, scoring, RAS) and save it in sample_corr_complete
"""

""" 
This whole computation is stored in u.RESULTS_FOLDER\pairwise_correlations_ranks.csv
"""


run = False
if run:
    sample_corr2 = {}
    sample_std2 = {}
    sample_corr_complete = {}
    for sample_size, repeated_ranks in tqdm(ranks2.items()):
        temp_corr = -np.ones(len(repeated_ranks))
        temp_std = -np.ones(len(repeated_ranks))
        temp_complete = pd.DataFrame(index=range(repetitions), columns=ranks2[sample_sizes[0]][0][0].columns)
        for rep, (r1, r2) in enumerate(repeated_ranks):
            assert (r1.columns == r2.columns).all()
            temp = []
            for col in r1.columns:
                k = kendalltau(r1[col], r2[col], variant="b").correlation
                # !!! Removed as it can be misleading and bad practice
                # if np.isnan(k):
                #     if (r1[col] == r2[col]).all():
                #         k = 1
                #     else:
                #         k = 0
                temp.append(k)
            temp_corr[rep] = np.mean(temp)
            temp_std[rep] = np.std(temp)
            temp_complete.loc[rep] = np.array(temp)

        if (temp_corr == -1).any() or (temp_std == -1).any():
            # raise ValueError("Some entries of sample_corr where not initialized")
            print(f"Something went wrong in sample_corr for sample size {sample_size} and column {col}")

        sample_corr2[sample_size] = temp_corr
        sample_std2[sample_size] = temp_std
        sample_corr_complete[sample_size] = temp_complete


run = False
if run:
    list_corr = []
    for ss, corr in sample_corr_complete.items():
        temp = corr.melt(var_name=["model", 'scoring', "strategy"], value_name="corr")
        temp["dataset_sample_size"] = ss
        list_corr.append(temp)
    ddd = pd.concat(list_corr).reset_index(drop=True)
    ddd["control_condition"] = ddd["model"] + "__" + ddd["scoring"] + "__" + ddd["strategy"]

control_conditions = ["model", "scoring", "strategy"]
ncc = len(control_conditions)

fig, axes = plt.subplots(1, ncc, figsize=(6*ncc, 1.5*ncc))
axes = np.array((axes,))
for ax, cc in zip(axes.flatten(), control_conditions):
    if cc != "control_condition":
        sns.pointplot(data=ddd,
                      x="dataset_sample_size",
                      y="corr",
                      hue=cc,
                      # errorbar="sd",
                      capsize=0.2,
                      ax=ax)
        if cc == "strategy":
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    else:
        sns.lineplot(data=ddd,
                     x="dataset_sample_size",
                     y="corr",
                     hue=cc,
                     ci=None,
                     )
        ax.set_xticks(sample_sizes)
        ax.legend([], [], frameon=False)
plt.show()

#%% Final proposed rank with RAS ensemble
scorings = df.scoring.unique()
models = df.model.unique()
ensemble_score = {}
recommendations = {}
for model, scoring in product(models, scorings):
    # number of times each encoder is in best class
    temp = (df_ranks.T.loc[(model, scoring)] == 0).sum().sort_values()
    ensemble_score[(model, scoring)] = temp
    recommendations[f"{model}, {scoring}"] = temp[temp == temp.max()].index.to_list()

ensemble_score = pd.DataFrame(ensemble_score)

import json

with open(os.path.join(u.RESULT_FOLDER, "recommendations.json"), 'w') as fw:
    json.dump(recommendations, fw)


#%% Look for different rankings to use as examples for introduction

s1 = 'hornik-meyer rank'
s2 = 'rescaled mean performance'

for model, scoring in product(models, scorings):
    good1 = df_ranks.loc[:, (model, scoring, s1)]
    good2 = df_ranks.loc[:, (model, scoring, s2)]

#%% Power of rank aggregation strategies

scorings = df.scoring.unique()
models = df.model.unique()
power = defaultdict(lambda: {})
for scoring, model in product(scorings, models):
    df_ranks = pd.DataFrame.from_dict({
        E: tuple(
            final_rank[E]
            for strategy, final_rank in rank[(scoring, model)].final_ranks.items()
        ) for E in rank[(scoring, model)].e2i.keys()
    }).T
    df_ranks.columns = list(rank[(scoring, model)].final_ranks.keys())
    for strategy in df_ranks.columns:
        power[strategy][(scoring, model)] = len(df_ranks[strategy].unique()) / len(df_ranks)

out_power = pd.DataFrame({
    strategy: list(powers.values()) for strategy, powers in power.items()
})

median = out_power.median().sort_values()
out_power = out_power[median.index]

fig, ax = plt.subplots(figsize=(14, 5))
sns.boxplot(data=out_power, orient="h")
fig.suptitle("Power of consensus ranking strategy")
plt.show()
#%% Predictivity of rank aggregation strategies with LOOCV on set of datasets

scoring = "roc_auc_score"
model = "LGBMClassifier"
datasets = df.dataset.unique()

run = False
if run:
    out = {}
    for scoring, model in tqdm(product(scorings, models)):
        rankbig = {}
        rankloo = {}
        for dataset in datasets:
            dfbig = df.loc[df.dataset != dataset]
            dfloo = df.loc[df.dataset == dataset]
            abig = Aggregator(dfbig, scoring, model)
            aloo = Aggregator(dfloo, scoring, model)
            for strategy in abig.supported_strategies:
                abig.aggregate(strategy=strategy, how="plain", k=5, alpha=0.1, th=0.95)
                aloo.aggregate(strategy=strategy, how="plain", k=5, alpha=0.1, th=0.95)
            rankbig[dataset] = abig
            rankloo[dataset] = aloo

        predictiveness = {}
        for dataset, a1 in rankbig.items():
            a2 = rankloo[dataset]

            # use the final rank from rankbig ie from all datasets except 'dataset' to predict the ranking of 'dataset'
            predictiveness[dataset] = {}
            for strategy in a1.final_ranks.keys():
                rpred = {k: v for k, v in sorted(a1.final_ranks[strategy].items(), key=lambda x: x[0])}
                rtrue = {k: v for k, v in sorted(a2.final_ranks[strategy].items(), key=lambda x: x[0])}

                rpred = np.array(tuple(rpred.values()))
                rtrue = np.array(tuple(rtrue.values()))

                predictiveness[dataset][strategy] = kendalltau(rpred, rtrue, variant="b").correlation

        # results in better form strategy: array
        out[(scoring, model)] = pd.DataFrame({
            strategy : np.array(tuple(
                preds[strategy]
                for dataset, preds in predictiveness.items()
            ))
            for strategy in a1.final_ranks.keys()
        })
print("Done!")

fig, axes = plt.subplots(len(scorings), len(models), figsize=(20, 10), sharex=True, sharey=True)
fig.suptitle("Predictivity\nloo-cross-validated Kendall's tau c")
sns.set_style("whitegrid")
for (i1, scoring), (i2, model) in product(enumerate(scorings), enumerate(models)):
    ax = axes[i1, i2]

    sns.boxplot(data=out[(scoring, model)], orient="h", ax=ax)
    ax.axvline(0)
    ax.set_title(f"{model}, {scoring}")

    if i1 == len(scorings)-1:
        ax.set_xlabel("Kendall's tau")

plt.show()


#%% Robustness of rank aggregation: take without noise and compare with noise
stds = [0.001, 0.01, 0.1, 1]
seeds = range(5)
run = False
if run:
    final_robustness = {}
    for std in tqdm(stds):
        for iter, seed in enumerate(seeds):
            np.random.seed(seed)

            # add normal noise
            df_noisy = df.copy()
            df_noisy.cv_score = df_noisy.cv_score + np.random.normal(0, std, len(df_noisy))

            # find noisy rankings
            scorings = df_noisy.scoring.unique()
            models = df_noisy.model.unique()
            rank_noisy = {}
            for scoring, model in product(scorings, models):
                a = Aggregator(df_noisy, scoring, model)
                for strategy in a.supported_strategies:
                    a.aggregate(strategy=strategy, how="plain", k=5, alpha=0.1, th=0.95)
                rank_noisy[(scoring, model)] = a


            # Compare the two
            out_robustness = {}
            for scoring, model in product(scorings, models):
                R = rank[(scoring, model)].final_ranks
                Rnoisy = rank_noisy[(scoring, model)].final_ranks
                robustness = {}
                for strategy in R.keys():
                    r = {k: v for k, v in sorted(R[strategy].items(), key=lambda x: x[0])}
                    rnoisy = {k: v for k, v in sorted(Rnoisy[strategy].items(), key=lambda x: x[0])}

                    r = np.array(tuple(r.values()))
                    rnoisy = np.array(tuple(rnoisy.values()))

                    robustness[strategy] = kendalltau(r, rnoisy, variant="b").correlation
                out_robustness[(scoring, model)] = robustness

            # put in a better form
            actual_robustness = {
                strategy: np.array(tuple(
                    robustness[strategy] for k, robustness in out_robustness.items()
                )) for strategy in R.keys()
            }

            final_robustness[(std, iter)] = actual_robustness
else:
    pass
print("Done!")

# better form std: results
out_rob_temp = defaultdict(lambda: dict())
for (std, iter), rob in final_robustness.items():
    for strategy, robs in rob.items():
        out_rob_temp[std][strategy] = robs
out_rob = dict()
for std, robs in out_rob_temp.items():
    out_rob[std] = pd.DataFrame(robs)

# out_rob = pd.DataFrame(out_rob)

# now plot
sns.set_style("whitegrid")
fig, axes = plt.subplots(len(stds), figsize=(15, 10), sharex=True, sharey=True)
fig.suptitle("Robustness to gaussian noise\nKendall's tau of original ranking and noised ranking")
for i, std in enumerate(stds):
    ax = axes[i]
    sns.boxplot(data=out_rob[std], orient="h", ax=ax)
    ax.set_title(f"Noise: {std}")

    if i == len(stds)-1:
        ax.set_xlabel("Kendall's tau")

plt.show()


#%% Compare runtimes of performances

def rescale_on_baseline(df, pk, col, baseline_encoder="DE", how="-"):
    df_ = df[pk + [col]].copy()
    baseline_col = f"baseline_{col}"
    baseline = df_.loc[df_.encoder == baseline_encoder].rename(columns={col: baseline_col})
    df_ = df_.merge(
        baseline.drop(columns=["encoder"]),
        left_on=["dataset", "model", "scoring", "fold"],
        right_on=["dataset", "model", "scoring", "fold"],
        how="left",
    )

    if how == "-":
        df_[col] -= df_[baseline_col]
    elif how == "/":
        df_[col] /= df_[baseline_col]
    else:
        raise ValueError(f"how = {how} is invalid. Use '-' or '/'. ")

    return df_

pk = ["dataset", "model", "scoring", "encoder", "fold"]
col = "tuning_time"
baseline_encoder = "OHE"
how = "/"

df_ = rescale_on_baseline(df.loc[df.model == "LGBMClassifier"], pk, col, baseline_encoder=baseline_encoder, how=how)

# get order
order = df_.groupby(["encoder"])[col].median().sort_values().index
fig, ax = plt.subplots(1, 1)
ax.set_xscale("log")
g = sns.boxplot(
    data=df_,
    y="encoder",
    x=col,
    orient="h",
    ax=ax,
    order=order,
)

sns.despine(trim=True, left=True)
plt.tight_layout()

plt.show()






#%% Explorative plots
"""
Other random plots 
"""
# find average performance for each encoder and dataset
apf = pd.DataFrame(df.groupby(["model", "scoring", "encoder", "dataset"])["cv_score"].mean())

# find worst performance per dataset
wpd = df.groupby(["model", "scoring", "dataset"])["cv_score"].min()

# find IQR of performances per dataset, if IQR == 0 -> constant -> IQR does not matter
iqrd = df.groupby(["model", "scoring", "dataset"])["cv_score"].agg(iqr)
iqrd[iqrd == 0] = 1

temp = apf.join(wpd, how="left", rsuffix="_worst").join(iqrd, how="left", rsuffix="_iqr")
temp["score"] = (temp["cv_score"] - temp["cv_score_worst"]) / (temp["cv_score_iqr"])

df_rescaled = temp.reset_index()

g = sns.FacetGrid(
    df_rescaled,
    col="scoring",
    row="model",
    sharex=True,
    margin_titles=True
)
g.map(
    sns.boxplot,
    "score",
    "encoder"
)
# g.fig.subplots_adjust(hspace=10, wspace=.15)

plt.tight_layout()
plt.show()


#%% 29dats tuning VS no tuning

df_notuning = rc.concatenate_results("29dats")
df_tuning = rc.concatenate_results("main6 results with tuning and 6-encoders",
                                   clean=True, remove_outdated_experiments=True)
df_tuning.rename(columns={"cv_scores": "cv_score"}, inplace=True)
# df_tuning = df_tuning[df_notuning.columns]
cs = ~ (df_tuning.model is None)
ct = ~ (df_notuning.model is None)
for col in ["dataset", "encoder", "model", "scoring"]:
    cs = cs & (df_tuning[col].isin(df_notuning[col].unique()))
    ct = ct & (df_notuning[col].isin(df_tuning[col].unique()))

df_tuning = df_tuning.loc[cs]
df_notuning = df_notuning.loc[ct]

#%% Test 1: performance gain by tuning
dfn = df_notuning
dft = df_tuning

pk = ["dataset", "encoder", "model", "scoring", "fold"]
dff = pd.merge(dfn, dft, on=pk, suffixes=("_n", "_t"))
pk.remove("fold")  # new pk
dff["cv_score_diff"] = dff.cv_score_t - dff.cv_score_n
df2 = dff.groupby(pk).mean().reset_index()
order = df2.groupby(["encoder"])["cv_score_diff"].median().sort_values().index
plot = False
if plot:
    fig, ax = plt.subplots()
    sns.boxplot(df2, x="cv_score_diff", y="encoder", ax=ax, order=order)
    ax.set_xlim((-0.05, 0.05))
    sns.despine(trim=True, left=True)
    plt.tight_layout()
    plt.show()

#%% Test 2: Aggregated difference of tuning

run = False
if run:
    for tuning, df in zip(("no_tuning", "tuning"), (dfn, dft)):
        scorings = df.scoring.unique()
        models = df.model.unique()
        rank = {}
        for scoring, model in tqdm(list(product(scorings, models))):
            a = Aggregator(df, scoring, model)
            a.aggregate(strategy="all", how="plain", k=5, alpha=0.1, th=0.95, solver=cp.GLPK_MI, 
                        skipped_strategies=("hornik-meyer rank",))
            rank[(scoring, model)] = a

        # Aggregate the final ranks into a single dataframe

        scorings = df.scoring.unique()
        models = df.model.unique()
        dict_ranks = {}
        for model, scoring in product(models, scorings):
            for strategy, final_rank in rank[(scoring, model)].final_ranks.items():
                dict_ranks[model, scoring, strategy] = dict(sorted(final_rank.items(), key=lambda x: x[0]))

        df_ranks = pd.DataFrame(dict_ranks)
        df_ranks.to_csv(os.path.join(u.RESULT_FOLDER, f"df29_ranks_{tuning}.csv"))
    print("Done!")

load = False
if load:
    dfn_ranks = pd.read_csv(os.path.join(u.RESULT_FOLDER, "df29_ranks_no_tuning.csv"),
                            header=[0, 1, 2],
                            index_col=0
                            )
    dft_ranks = pd.read_csv(os.path.join(u.RESULT_FOLDER, "df29_ranks_tuning.csv"),
                            header=[0, 1, 2],
                            index_col=0
                            )

# - Compare the rankings
columns = dfn_ranks.columns
corrs = pd.Series(index=dfn_ranks.columns)
corrbest = pd.Series(index=dfn_ranks.columns)
corrworst = pd.Series(index=dfn_ranks.columns)
for col in columns:
    rn = dfn_ranks[col]
    rt = dft_ranks[col]
    corrs[col] = kendalltau(rn, rt, variant="b").correlation
    corrbest[col] = agreement(rn, rt, best=True)
    corrworst[col] = agreement(rn, rt, best=False)

plot = True
if plot:

    names = ["tau_b", "agreement on best", "agreement on worst"]
    corrr = [
        pd.DataFrame(corr).reset_index().rename(columns={"level_0": "model",
                                                         "level_1": "scoring",
                                                         "level_2": "strategy",
                                                         0: "agreement"})
        for corr in [corrs, corrbest, corrworst]
    ]

    fig, axes = plt.subplots(3, 1, sharex="all", figsize=(10, 7))
    fig.suptitle("Comparison of aggregated ranks with and without tuning, over set of RASs.")

    for ax, (name, corr) in zip(axes, zip(names, corrr)):
        c = corr["agreement"]
        ax.set_title(f"{name}, median : {c.median():.02} (iqr {iqr(c):.02}), mean : {c.mean():.02} ({c.std():.02})")
        g = sns.boxplot(data=corr, ax=ax, orient="h", x="agreement", y="model")
        g.set(xlabel=None, ylabel=None)

    sns.despine(trim=True, left=True)
    plt.tight_layout()
    plt.show()

#%% Test 3: Disaggregated difference
"""
Fix a dataset, compare rankings for model and scoring
"""
run = False
if run:
    ccn = {}
    cct = {}
    for model, scoring in tqdm(tuple(product(models, scorings))):
        an = Aggregator(dfn, model=model, scoring=scoring)
        an._get_domination_matrices("plain")
        at = Aggregator(dft, model=model, scoring=scoring)
        at._get_domination_matrices("plain")
        for dmn, dmt, dataset in zip(an.plain_domination_matrices, at.plain_domination_matrices, at.df.dataset.unique()):
            ccn[(dataset, model, scoring)] = an._get_rank_from_matrix(dmn)
            cct[(dataset, model, scoring)] = an._get_rank_from_matrix(dmt)

    df_ranks_datn = pd.DataFrame(ccn)
    df_ranks_datt = pd.DataFrame(cct)

    df_ranks_datn.to_csv(os.path.join(u.RESULT_FOLDER, f"df29_ranks_perdataset_tuning.csv"))
    df_ranks_datt.to_csv(os.path.join(u.RESULT_FOLDER, f"df29_ranks_perdataset_no_tuning.csv"))

columns = df_ranks_datn.columns
corrs = pd.Series(index=df_ranks_datn.columns)
corrbest = pd.Series(index=df_ranks_datn.columns)
corrworst = pd.Series(index=df_ranks_datn.columns)
for col in columns:
    rn = df_ranks_datn[col]
    rt = df_ranks_datt[col]
    corrs[col] = kendalltau(rn, rt, variant="b").correlation
    corrbest[col] = agreement(rn, rt, best=True)
    corrworst[col] = agreement(rn, rt, best=False)

plot = True
if plot:

    names = ["tau_b", "agreement on best", "agreement on worst"]
    corrr = [
        pd.DataFrame(corr).reset_index().rename(columns={"level_0": "dataset",
                                                         "level_1": "model",
                                                         "level_2": "scoring",
                                                         0: "agreement"})
        for corr in [corrs, corrbest, corrworst]
    ]
    fig, axes = plt.subplots(3, 1, sharex="all", figsize=(10, 20))
    fig.suptitle("Comparison of DISaggregated ranks with and without tuning, over set of datasets.")

    for ax, (name, corr) in zip(axes, zip(names, corrr)):
        c = corr["agreement"]
        ax.set_title(f"{name}, median : {c.median():.02} (iqr {iqr(c):.02}), mean : {c.mean():.02} ({c.std():.02})")
        g = sns.boxplot(data=corr, ax=ax, orient="h", x="agreement", y="dataset", hue="model")
        g.set(xlabel=None, ylabel=None)

    sns.despine(trim=True, left=True)
    plt.tight_layout()
    plt.show()

#%% Test 4 (run after 3): Single dataset analysis

dataset = "Australian"
drn = df_ranks_datn.loc[:, dataset]


# check that I am using the correct ranks
ddn = dfn.loc[dfn.dataset == dataset].groupby(["model", "scoring", "encoder"]).cv_score.mean().reset_index()
for model, scoring in drn.columns:
    n1 = drn[(model, scoring)].sort_values()
    n2 = ddn.loc[ddn.model == model].loc[ddn.scoring == scoring].drop(columns=["model", "scoring"])
    n2.index = n2.encoder
    n2 = n2.drop(columns="encoder").sort_values("cv_score")

    p1 = list(n1.index)
    p2 = list(n2.index)
    p2.reverse()

    for e1, e2, in zip(p1, p2):
        if e1 != e2 and (n2.cv_score[e1] - n2.cv_score[e2] > 0.001):
           raise Exception(f"{e1:15}{e2}")

# try
model1 = u.DecisionTreeClassifier(max_depth=2)
scoring = u.roc_auc_score


"""
Open: 
are there some models for which tuning does not have strong effect? - no
are there some scorings? - no
are there datasets for which the tuning does not work?
    if so, what could be the cause?
"""


#%% Best hpars from tuning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# for every encoder, model, scoring, count the occurrences of every hpar combination
hpars = {}
for (model, encoder, scoring), indices in tqdm(df_tuning.groupby(by=["model", "encoder", "scoring"]).groups.items()):
    tuncol = u.get_pipe_search_space_one_encoder(model=globals()[model](), encoder=None)
    hpars[(model, encoder, scoring)] = df_tuning.loc[indices][tuncol].value_counts()
hpars = {
    (model, encoder): pd.concat([hp for ((mod, enc, _), hp) in hpars.items() if mod == model and enc == encoder]) for (model, encoder, _) in hpars.keys()
}
#%%
fig, axes = plt.subplots(len(df_tuning.model.unique()), len(df_tuning.encoder.unique()), figsize=(30, 30))
for (((model, encoder), hp), ax) in zip(hpars.items(), axes.flatten()):
    sns.histplot(hp, ax=ax)
    # ax.set_xticks(hp.index.unique())
    break

plt.show()

#%% Decision Tree hpars

model = "DecisionTreeClassifier"
hps = list(u.get_pipe_search_space_one_encoder(model=globals()[model](), encoder=None).keys())

dfmod = df_tuning.loc[df_tuning.model == model, ["encoder", "scoring"]+hps]
# order = dfmod.loc[dfmod[hps[0]] == 4].groupby("encoder").count()

fig, ax = plt.subplots(figsize=(10, 10))
fig.suptitle(model)
sns.histplot(data=dfmod, y="encoder", hue=hps[0])
sns.despine(trim=True, left=True)
plt.tight_layout()
plt.show()
#%% SVC hpars

model = "SVC"
hps = list(u.get_pipe_search_space_one_encoder(model=globals()[model](), encoder=None).keys())

dfmod = df_tuning.loc[df_tuning.model == model, ["encoder", "scoring"]+hps]
# order = dfmod.loc[dfmod[hps[0]] == 4].groupby("encoder").count()

fig, ax = plt.subplots(figsize=(10, 10))
fig.suptitle(model)
sns.countplot(data=dfmod, y="encoder", hue=hps[0])
sns.despine(trim=True, left=True)
plt.tight_layout()
plt.show()

#%% KNN hpars

model = "DecisionTreeClassifier"
hps = list(u.get_pipe_search_space_one_encoder(model=globals()[model](), encoder=None).keys())

dfmod = df_tuning.loc[df_tuning.model == model, ["encoder", "scoring"]+hps]
# order = dfmod.loc[dfmod[hps[0]] == 4].groupby("encoder").count()

# dfmod = dfmod.sort_values(dfmod[hps].value_counts().index)

fig, ax = plt.subplots(figsize=(10, 10))
fig.suptitle(model)
sns.histplot(data=dfmod, y="encoder", hue=hps[0], color="red", discrete=True, common_bins=True, multiple="stack")
sns.despine(trim=True, left=True)
plt.tight_layout()
plt.show()
