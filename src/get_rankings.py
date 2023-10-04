import pandas as pd

from typing import Iterable, AnyStr

import src.relation_utils as rlu
import src.utils as u


def get_rankings(df: pd.DataFrame, factors: Iterable, alternatives: AnyStr, target: AnyStr, increasing=True, impute_missing=True) -> pd.DataFrame:
    """
        Compute a ranking of 'alternatives' for each combination of 'factors', according to 'target'.
        Set increasing = True if 'target is a score, to False if it is a loss or a rank.
        If impute_missing == True, the empty ranks are imputed.
    """

    if not set(factors).issubset(df.columns):
        raise ValueError("factors must be an iterable of columns of df.")
    if alternatives not in df.columns:
        raise ValueError("alternatives must be a column of df.")
    if target not in df.columns:
        raise ValueError("target must be a column of df.")

    rankings = {}
    for group, indices in df.groupby(factors).groups.items():
        score = df.iloc[indices].set_index(alternatives)[target]
        rankings[group] = rlu.score2rf(score, increasing=increasing, impute_missing=impute_missing)

    return pd.DataFrame(rankings)


if __name__ == "__main__":
    df = u.load_df().groupby(["dataset", "model", "scoring", "tuning", "encoder"])["cv_score"].mean().reset_index()
    rf = get_rankings(df, factors=["dataset", "model", "tuning", "scoring"], alternatives="encoder", target="cv_score",
                      increasing=True, impute_missing=False)
    rf.to_parquet(u.RESULTS_DIR / "rankings.parquet")

