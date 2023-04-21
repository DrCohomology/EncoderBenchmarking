import numpy as np
import pandas as pd

from scipy.stats import iqr
from tqdm import tqdm
from typing import Union

"""
Usage:

RESULT_FOLDER  # path to folder 

rankings_folder = Path(RESULT_FOLDER).parent / "Rankings"
df = pd.read_csv(Path(RESULT_FOLDER, "final.csv"))
rf = pd.read_csv(rankings_folder / "rank_function_from_average_cv_score.csv",
                 index_col=0, header=[0, 1, 2, 3])

a = Aggregator(df, rf)
a.aggregate(verbose=True)
a.aggrf.to_csv(rankings_folder / "aggregated_ranks_from_average_cv_score.csv")
"""



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
            # "kemeny rank": self._aggr_rank_kemeny,
            # "nemenyi rank": self._aggr_rank_nemenyi,
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

    # def _aggr_rank_kemeny(self, **solver_params):
    #     self.aggrf["rank_kemeny"] = ru.kemeny_aggregation_gurobi_ties(self.rf.set_axis(range(self.rf.shape[1]), axis=1),
    #                                                                   **solver_params)

    # def _aggr_rank_nemenyi(self, alpha=0.05):
    #     """
    #     Issues with no answer (yet):
    #         is the test "good" when the rankings involved have different number of tiers?
    #         does it support missing values?
    #         transitivity is not guaranteed, however, it seems to be always transitive
    #
    #     Compute the outranking (domination) matrix and the matrix of significant differences according to Nemenyi pw tests,
    #     then multiply them together to get the significative differences matrix, and rebuild a rank function from it
    #     """
    #     self.aggrf["rank_nemenyi"] = ru.mat2rf(ru.rf2mat(ru.score2rf(self.rf.mean(axis=1)), kind="domination") *
    #                                            (posthoc_nemenyi_friedman(self.rf.T.reset_index(drop=True)) < alpha)
    #                                            .astype(int).to_numpy(),
    #                                            alternatives=self.rf.index)

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
                                            .join(self.df.groupby("dataset").cv_score.max(), on="dataset", rsuffix="_best")\
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
            self.aggrf[col] = score2rf(self.aggrf[col], ascending=self.ascending[col])


class Aggregator(object):
    """
    Aggregator that subsets the columns of df and rf to select model, tuning, and scoring
        As default behaviour, loops over all combinations
    For each combination, it aggregates scores (self.df) and/or rankings (self.rf) into
        a ranking computed with an aggregation strategy
    Accepted aggregation strategies are the keys of self.supported_strategies
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