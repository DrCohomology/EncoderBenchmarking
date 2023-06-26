import functools
import numpy as np
import pandas as pd
import seaborn as sns
import rpy2.robjects as ro

from category_encoders import (
    BackwardDifferenceEncoder,
    BinaryEncoder,
    CatBoostEncoder,
    CountEncoder,
    GLMMEncoder,
    HashingEncoder,
    HelmertEncoder,
    JamesSteinEncoder,
    LeaveOneOutEncoder,
    MEstimateEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialEncoder,
    SumEncoder,
    WOEEncoder,
)
from collections import defaultdict
from dirty_cat import MinHashEncoder
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import src.utils as u

# ----------------------------------------------------------------------------------------------------------------------
# General Purpose
# ----------------------------------------------------------------------------------------------------------------------


class DFImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, imputer):
        self.imputer = imputer
        
    def fit(self, X, y, **kwargs):
        self.imputer.fit(X, y)
        return self
        
    def transform(self, X, **kwargs):
        return pd.DataFrame(self.imputer.transform(X), index=X.index, columns=X.columns)
    
    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X, y, **kwargs).transform(X)

    def __str__(self):
        return self.imputer.__class__.__name__


class PreBinner(BaseEstimator, TransformerMixin):
    """
    Bins the levels of every attribute of X before encoding.
    The binning algorithm tries to maximize the number of bins such that the sum of relative frequencies of the
        levels in each bin exceeds 'thr'.
    Convergence of the algorithm is not proven.
    """

    def __init__(self,
                 thr: float = 0,
                 **kwargs):
        self.thr = thr
        self.binnings = defaultdict(lambda: {})
        self.cols = None

    def _bin(self, X:pd.DataFrame):
        """
        Bins every attribute and stores the binnings in self.binnings
        """
        for col in self.cols:
            tmp = X[col].value_counts(ascending=True) / len(X)
            for box in u.find_bins(tmp.index.to_numpy(), tmp.to_numpy(), self.thr):
                for lvl in box:
                    self.binnings[col][lvl] = '; '.join(box)

    def fit(self, X: pd.DataFrame, y=None, **kwargs):
        self.cols = X.columns
        self._bin(X)
        return self

    def _update_binnings(self, X):
        for col in self.cols:
            self.binnings[col].update({x: x for x in set(X[col].unique())-set(self.binnings[col].keys())})

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        # Update self.binnings with the unseen levels of X[col], left as-is to self.base_encoder to deal with
        X = X.copy()
        self._update_binnings(X)
        for col in self.cols:
            X[col] = X[col].map(self.binnings[col])
        return X

    def fit_transform(self, X: pd.DataFrame, y=None, **kwargs):
        return self.fit(X, y, **kwargs).transform(X, y, **kwargs)

# ----------------------------------------------------------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------------------------------------------------------


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, default=-1, **kwargs):
        self.default = default
        self.encoding = defaultdict(lambda: defaultdict(lambda: self.default))
        self.inverse_encoding = defaultdict(
            lambda: defaultdict(lambda: self.default))
        self.cols = None

    def fit(self, X: pd.DataFrame, y, **kwargs):
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        X = X.copy()
        return X

    def fit_transform(self, X: pd.DataFrame, y=None, **kwargs):
        return self.fit(X, y, **kwargs).transform(X, y, **kwargs)


class DropEncoder(Encoder):
    """
    Transform into a vector of 1's.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        self.cols = X.columns
        return pd.DataFrame(np.ones(len(X)), index=X.index, columns=['cat'])


class TargetEncoder(Encoder):
    """
    Transform a level with the average target conditional on attribute == level.
    """

    def __init__(self, default=-1, **kwargs):
        super().__init__(default=default, **kwargs)

    def fit(self, X: pd.DataFrame, y, **kwargs):
        self.cols = X.columns
        self.feature_names = self.cols.to_list()
        target_name = y.name
        X = X.join(y.squeeze())
        for col in self.cols:
            temp = X.groupby(col)[target_name].mean().to_dict()
            self.encoding[col].update(temp)
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].apply(lambda cat: self.encoding[col][cat])
        return X.applymap(np.float)


class MeanEstimateEncoder(Encoder):
    """
    TargetEncoder with smoothing parameter, adapted from [1].

    [1] Micci-Barreca, Daniele. "A preprocessing scheme for high-cardinality categorical attributes in classification
        and prediction problems." ACM SIGKDD Explorations Newsletter 3.1 (2001): 27-32.
    """

    def __init__(self, default=-1, m=10, **kwargs):
        super().__init__(default=default, **kwargs)
        self.m = m

    def fit(self, X: pd.DataFrame, y, **kwargs):
        self.cols = X.columns
        X = X.join(y.squeeze())
        global_mean = y.mean()
        for col in self.cols:
            temp = X.groupby(col).agg(['sum', 'count'])
            temp['STE'] = (temp.target['sum'] + self.m * global_mean) / (temp.target['count'] + self.m)
            self.encoding[col].update(temp['STE'].to_dict())
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].apply(lambda cat: self.encoding[col][cat])
        return X

    def __str__(self):
        return f"MEstimate{self.m}Encoder"


class RGLMMEncoder(Encoder):
    """
    Alternative faster implementation of category_encoders.GLMMEncoder.
    Relies on rpy2.
    Adapted from [2].

    [2] Pargent, F., Pfisterer, F., Thomas, J., & Bischl, B. (2022). Regularized target encoding outperforms
        traditional methods in supervised machine learning with high cardinality features.
        Computational Statistics, 37(5), 2671-2692.

    """

    def __init__(self, default=-1, **kwargs):
        super().__init__(default=default, **kwargs)
        self.rlibs = kwargs["rlibs"] if "rlibs" in kwargs else None

    def _is_constant(self, X, col):
        """
        If the attribute has only one level, return a constant value
        """
        return len(X[col].unique()) == 1

    def _import_rlibs(self):
        if self.rlibs is None:
            importr("lme4")
            importr("base")
            importr("utils")
        elif "lme4" not in self.rlibs:
            importr("lme4")
        elif "base" not in self.rlibs:
            importr("base")
        elif "utils" not in self.rlibs:
            importr("utils")
        pass

    def fit(self, X: pd.DataFrame, y, **kwargs):
        # self._import_rlibs()
        self.cols = X.columns
        with localconverter(ro.default_converter + pandas2ri.converter):
            env = ro.globalenv

            rX = ro.conversion.py2rpy(X)
            ry = ro.conversion.py2rpy(y)

            for col in self.cols:
                if self._is_constant(X, col):
                    self.encoding[col] = defaultdict(lambda: 0)
                    continue

                env["rdf_"] = ro.DataFrame({
                    "x": rX[rX.colnames.index(col)],
                    "y": ry
                })

                ro.r("""
                    fitted_model <- do.call(
                        lme4::lmer,
                        args = list(
                            formula = y ~ 1 + (1 | x),
                            data = rdf_,
                            control = lme4::lmerControl(
                                check.conv.singular = lme4::.makeCC(
                                    action="ignore", 
                                    tol=formals(isSingular)$tol
                                )
                            )
                        )
                    )
                    coefs <- data.frame(coef(fitted_model)$x)
                """)
                tmp = ro.conversion.rpy2py(ro.globalenv["coefs"])
                self.encoding[col].update(tmp[tmp.columns[0]].to_dict())
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].apply(lambda cat: self.encoding[col][cat])
        return X

# ----------------------------------------------------------------------------------------------------------------------
# Wrappers
# ----------------------------------------------------------------------------------------------------------------------


class Regularization(BaseEstimator, TransformerMixin):

    def __init__(self):
        super().__init__()


class Discretized(Regularization):
    """
    Wrapper around an Encoder object that takes as input a pd.DataFrame.
    Discretized discretizes the encoding according to the 'how' parameter.
    """
    
    def __init__(self,
                 base_encoder: Encoder,
                 how: str = "minmaxbins",
                 n_bins: int = 10,
                 **kwargs):
        super().__init__()
        self.base_encoder = base_encoder
        self.how = how
        self.n_bins = n_bins
        
        self.accepted_how = [
            "boxes",
            "minmaxbins", 
        ]
        if self.how not in self.accepted_how:
            raise ValueError(f"{how} is not a valid value for parameter how.")
        
        self.discretization = defaultdict(lambda: {})
        self.cols = None
    
    def fit(self, X: pd.DataFrame, y, **kwargs):
        XE = self.base_encoder.fit_transform(X, y).applymap(np.float)
        self.cols = XE.columns
        
        for col in self.cols:
            xs = XE[col].unique()

            if self.how == "boxes":
                dxs = np.floor(self.n_bins * xs) / self.n_bins
            elif self.how == "minmaxbins":
                m, M = xs.min(), xs.max()
                if m == M:
                    dxs = np.ones_like(xs)
                else:
                    # [m, M] -> [0, 1]
                    dxs = (xs-m)/(M-m)
                    # partition [0, 1] in bins
                    dxs = np.floor(self.n_bins * dxs) / self.n_bins
                    # [0, 1] -> [m, M]
                    dxs = (M-m)*dxs + m
            else:
                raise NotImplementedError(f"Strategy {self.how} not yet implemented")

            self.discretization[col] = dict(zip(xs, dxs))
            self.discretization[col][self.base_encoder.default] = self.base_encoder.default
        return self
          
    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        XE = self.base_encoder.transform(X, y)
        for col in self.cols:
            XE[col] = XE[col].map(self.discretization[col])
        return XE

    def fit_transform(self, X: pd.DataFrame, y=None, **kwargs):
        return self.fit(X, y, **kwargs).transform(X, y, **kwargs)

    def __str__(self):
        return f"Discretized{self.base_encoder.__class__.__name__}{self.how.capitalize()}{self.n_bins}"


class PreBinned(Regularization):
    """
    Bins the levels of an attribute X before encoding.
    The binning algorithm tries to maximize the number of bins such that the sum of relative frequencies of the
        levels in each bin exceeds 'thr'.
    Convergence of the algorithm is not proven.
    """

    def __init__(self,
                 base_encoder: Encoder,
                 thr: float = 0,
                 **kwargs):
        super().__init__()
        self.thr = thr
        self.base_encoder = base_encoder
        self.base_binner = PreBinner(thr=self.thr, **kwargs)

    def fit(self, X: pd.DataFrame, y, **kwargs):
        self.base_encoder.fit(self.base_binner.fit_transform(X, y, **kwargs), y, **kwargs)
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        return self.base_encoder.transform(self.base_binner.transform(X, y, **kwargs), y, **kwargs)

    def fit_transform(self, X: pd.DataFrame, y=None, **kwargs):
        return self.fit(X, y, **kwargs).transform(X, y, **kwargs)

    def __str__(self):
        return f"PreBinned{self.base_encoder.__class__.__name__}{self.thr}"


class CVRegularized(Regularization):
    """
    Split X into 'n_splits' folds, encode each fold with an instance of 'base_encoder' trained on the other folds.
    """

    def __init__(self,
                 base_encoder: Encoder,
                 n_splits: int = 5,
                 random_state: int = 1444,
                 default=-1,
                 **kwargs):
        super().__init__()
        self.base_encoder = base_encoder
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = StratifiedKFold(
            n_splits=self.n_splits,
            random_state=self.random_state,
            shuffle=True
        )
        self.cols = None
        self.splits = None
        self.fold_encoders = [clone(self.base_encoder) for _ in range(n_splits)]
        self.default = default

    def fit(self, X: pd.DataFrame, y, **kwargs):
        self.cols = X.columns

        # Fit a different targetEncoder on each training fold
        self.splits = []
        for E, (tr, te) in zip(self.fold_encoders, self.cv.split(X, y)):
            self.splits.append((tr, te))
            E.fit(X.iloc[tr], y.iloc[tr])

        # default encoding
        self.base_encoder.fit(X, y)

        return self

    def fit_transform(self, X: pd.DataFrame, y=None, **kwargs):
        """
        Training step: each fold is encoded differently
        """
        X = X.copy().astype('object')
        self.fit(X, y, **kwargs)

        # default values
        Xdefault = self.base_encoder.transform(X)

        for E, (tr, te) in zip(self.fold_encoders, self.splits):
            X.iloc[te] = E.transform(X.iloc[te])

        # default values handling
        default = X.isna()
        X[default] = Xdefault[default]

        # still missing values?
        X.fillna(self.default, inplace=True)

        return X

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        """
        Test step: the whole dataset is encoded with the base_encoder
        """

        X = X.copy().astype('object')
        X = self.base_encoder.transform(X, y)

        # missing values?
        X.fillna(self.default, inplace=True)

        return X

    def __str__(self):
        return f'CV{self.n_splits}{self.base_encoder.__class__.__name__}'


class CVBlowUp(Regularization):
    """
    Split 'X' into 'n_splits' folds, encode 'X" with an instance of 'base_encoder' trained on a fold, for every fold.
    """

    def __init__(self,
                 base_encoder: Encoder,
                 n_splits: int = 5,
                 random_state: int = 1444,
                 **kwargs):
        super().__init__()
        self.base_encoder = base_encoder
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = StratifiedKFold(
            n_splits=self.n_splits,
            random_state=self.random_state,
            shuffle=True
        )
        self.fold_encoders = [clone(self.base_encoder) for _ in range(n_splits)]

    def fit(self, X: pd.DataFrame, y, **kwargs):
        self.cols = X.columns

        # Fit a different targetEncoder on each training fold
        for E, (tr, te) in zip(self.fold_encoders, self.cv.split(X, y)):
            Xtr, ytr = X.iloc[tr], y.iloc[tr]
            E.fit(Xtr, ytr)

        # default encoding
        self.base_encoder.fit(X, y)

        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        X = X.copy()

        # Transform with each encoder the whole dataset
        XEs = []
        for fold, E in enumerate(self.fold_encoders):
            XEs.append(E.transform(X).add_prefix(f'f{fold}_'))
        XE = functools.reduce(lambda x, y: x.join(y), XEs)

        return XE

    def __str__(self):
        return f'BlowUpCV{self.n_splits}{self.base_encoder.__class__.__name__}'