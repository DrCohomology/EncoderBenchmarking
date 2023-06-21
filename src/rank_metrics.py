"""
Metrics and similarity coefficients for rankings.
"""

import numpy as np
import pandas as pd
import warnings

from scipy.stats import kendalltau, spearmanr, ConstantInputWarning
from typing import Iterable, Union


def jaccard(col1: pd.Series, col2: pd.Series, best: bool = True) -> float:
    """
    Jaccard similarity of the best or worst tier.

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


def jaccard_best(col1: pd.Series, col2: pd.Series) -> float:
    return jaccard(col1, col2, best=True)


def jaccard_worst(col1: pd.Series, col2: pd.Series) -> float:
    return jaccard(col1, col2, best=False)


def spearman_rho(x: Iterable, y: Iterable, nan_policy="omit"):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConstantInputWarning)
            return spearmanr(x, y, nan_policy=nan_policy)[0]
    except ValueError:
        return np.nan

