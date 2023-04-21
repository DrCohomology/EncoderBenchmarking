# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:59:19 2022

@author: federicom
"""

import functools
import glob
import math
import numpy as np
import os
import pandas as pd
import random
import string

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
from collections import defaultdict, Counter
from collections.abc import Iterable
from dirty_cat import (
    MinHashEncoder,
    SimilarityEncoder
)
from inspect import signature
from scipy.stats import ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

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
    For every attribute, bins the levels so that the ratio of occurrences is above a certain threshold for every bin.
    When transforming leaves unseen values unchanged bz default.
    """

    def __init__(self, thr=0, **kwargs):
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
    Evey categorical value is mapped to 1
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        self.cols = X.columns
        return pd.DataFrame(np.ones(len(X)), index=X.index, columns=['cat'])


class TargetEncoder(Encoder):
    """
    Maps categorical values into the average target associated to them
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


class MEstimate(Encoder):
    """
    TargetEncoder with smoothing parameter.
    From https://github.com/rapidsai/deeplearning/blob/main/RecSys2020Tutorial/03_3_TargetEncoding.ipynb
    use micci_barreca
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
    As GLMMEncoder is super slow, this wraps an R GLMMEncoder
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
    Discretizes the encoded values 
    The __str__ method ignores any parameter of the base_encoder
    For Encoder objects. Does not wor if base_encoder does not encode to pd.DataFrame
    """
    
    def __init__(self, base_encoder, how="minmaxbins", n_bins=10, **kwargs):
        super().__init__()
        self.base_encoder = base_encoder
        self.how = how
        self.n_bins = n_bins # only used if how != adaptive
        
        self.accepted_how = [
            "boxes",
            "minmaxbins", 
            # "adaptive"
        ]
        if self.how not in self.accepted_how:
            raise ValueError(f"{how} is not a valid value for parameter how")
        
        self.discretization = defaultdict(lambda: {})
    
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
    For every attribute, bins the levels so that the ratio of occurrences is above a certain hreshold for every bin.
    Regularization for TE
    """

    def __init__(self, base_encoder, thr=0, **kwargs):
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
    Encodes every test fold with an Encoder trained on the training fold.
    Default value of base_encoder is supposed to be np.nan to allow the usage of
    default encoder trained on the whole dataset.

    The __str__ method ignores any parameter of the base_encoder
    """

    def __init__(self, base_encoder: Encoder, n_splits=5, random_state=1444, default=-1, **kwargs):
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
    Has a column for every fold

    The __str__ method ignores any parameter of the base encoder
    """

    def __init__(self, base_encoder, n_splits=5, random_state=1444, **kwargs):
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


# ----------------------------------------------------------------------------------------------------------------------
# Unused encoders
# ----------------------------------------------------------------------------------------------------------------------

class DistanceEncoder(Encoder):
    """
    Binary Classification Only
    for each categorical value d, take the target vector y in the entries with d, get y_d,
    and find the closest constant vector to it.
    the formula is very simple, every entry P of the vector is #1s^2 / (#1s^2+#0s^2)
    where #1s is the number of ones in y_d
    encode d with P
    """

    def __init__(self, default=-1, **kwargs):
        super().__init__(default=default, **kwargs)

    def fit(self, X: pd.DataFrame, y, **kwargs):
        self.cols = X.columns
        for col in self.cols:
            for cat in X[col].unique():
                yd = y.loc[X[col] == cat]
                ones = sum(yd)
                zeros = len(yd) - ones
                self.encoding[col][cat] = ones**2 / (ones**2 + zeros**2)
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].apply(lambda cat: self.encoding[col][cat])
        return X


class LocalOptimizerEncoder(Encoder):
    """
    For each categorical value d, take the constant value P(d) such that a
    constant algorithm mapping to P(d) will optimize performance to y_d. 
    Then encode with the metric value
    In case of Binary Classification, take the majority class
    """

    def __init__(self, default=-1, score=u.accuracy_score, majority_class=False, **kwargs):
        super().__init__(default=default, **kwargs)
        self.score = score
        self.majority_class = majority_class

    def fit(self, X: pd.DataFrame, y, **kwargs):
        self.cols = X.columns
        for col in self.cols:
            for cat in X[col].unique():
                ycat = y.loc[X[col] == cat]
                if self.majority_class:
                    ymaj = np.ones_like(ycat) \
                        if ycat.sum()/len(ycat) > 0.5 \
                        else np.zeros_like(ycat)
                else:
                    ymaj = np.ones_like(ycat)
                try:
                    self.encoding[col][cat] = self.score(ycat, ymaj)
                except:
                    self.encoding[col][cat] = -1
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].apply(lambda cat: self.encoding[col][cat])
        return X


class JGLMMEncoder(Encoder):
    """
    Use julia to speed up GLMMEncoder

    """
    def __init__(self, default=-1, **kwargs):
        super().__init__(default=default, **kwargs)
        self.basedir = os.path.dirname(__file__)
        self.tmpdir = None
        self.tmpdat = None
        self.encoding = None

        raise Exception("This encoder is broken and should not be used")


    class JuliaError(Exception):
        def __init__(self, t):
            super().__init__(t)

    def _add_path_to_julia(self):
        # TODO: add automatic check for julia path (call julia and see if it works)
        path_to_julia = "C:\\Users\\federicom\\AppData\\Local\\Programs\\Julia-1.8.2\\bin"
        if path_to_julia not in os.getenv("PATH").split(';'):
            os.environ["PATH"] += f";{path_to_julia}"

    def _make_tmpdir(self):
        # self.tmpdir = ''.join([random.choice(string.ascii_letters) for _ in range(5)])

        self.tmpdir = 'testitesttestest'
        try:
            os.mkdir(os.path.join(self.basedir, self.tmpdir))
        except:
            pass
        self.tmpdat = os.path.join(self.basedir, self.tmpdir, "dataset.csv")

    def _dump_dataset(self, X, y):
        tmp = X.copy()
        tmp["y"] = y
        tmp.to_csv(self.tmpdat, index=False)

    def _call_julia(self):
        # TODO: assumes that juia file is in the same directory as encoders.py
        # os.system(f"julia {self.basedir}\\fit_glmm.jl {self.basedir} {self.tmpdir}")
        # os.system(f"{self.basedir}\\FitGLMMcompiled\\bin\\FitGLMM.exe {self.basedir} {self.tmpdir}")
        # print(os.path.join(self.basedir, self.tmpdir, "FitGLMMnoargscompiled", "bin", "FitGLMMnoargs.exe"))
        # os.system(os.path.join(self.basedir, self.tmpdir, "FitGLMMnoargscompiled", "bin", "FitGLMMnoargs.exe"))
        #
        # import glob
        # for xx in glob.glob(os.path.join(self.basedir, self.tmpdir, "FitGLMMnoargscompiled", "bin", "*.exe")):
        #     print(xx)

        import rpy2.robjects as robjects
        robjects.r.source(os.path.join(self.basedir, self.tmpdir, "fit_glmm.R"), encoding="utf-8")


    def _gather_results(self, check_cols):
        self.encoding = {}
        for complete_fname in glob.glob(os.path.join(self.basedir, self.tmpdir, "*.csv")):
            fname = os.path.split(os.path.splitext(complete_fname)[0])[-1]
            if fname == "dataset":
                continue
            elif fname not in check_cols:
                raise self.JuliaError(f"{fname} is not a column of the original DataFrame.")

            check_cols.pop(check_cols.index(fname))
            self.encoding[fname] = pd.read_csv(complete_fname, index_col=0).to_dict()

        # check that all columns are OK
        if len(check_cols) > 0:
            raise self.JuliaError(f"{check_cols} are still left.")

    def _delete_tmpdir(self):
        pass

    def fit(self, X: pd.DataFrame, y, **kwargs):
        self._make_tmpdir()
        self._dump_dataset(X, y)
        self._add_path_to_julia()
        self._call_julia()
        self._gather_results(check_cols=X.columns.to_list())
        self._delete_tmpdir()

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].apply(lambda cat: self.encoding[col][cat])
        return X


class OOFTEWrapper(Encoder):
    
    def __init__(self, encoder_class):
        self.encoder_class = encoder_class
        # Can the encoder handle a single column?
        self.singlecol = False
        if "col" in signature(self.encoder_class).parameters:
            self.singlecol = True
        self.encoders = {}
        
    def fit(self, X: pd.DataFrame, y, **kwargs):
        self.cols = X.columns
        X = X.copy().reset_index(drop=True)
        
        if self.singlecol:
            for col in self.cols:
                self.encoders[col] = self.encoder_class(col, **kwargs).fit(X, y)
        else:
            X = self.encoder_class(**kwargs).fit_transform(X, y)
        return self
        
    def fit_transform(self, X: pd.DataFrame, y, **kwargs):
        self.cols = X.columns
        X = X.copy().reset_index(drop=True)
        
        if self.singlecol:
            for col in self.cols:
                self.encoders[col] = self.encoder_class(col, **kwargs).fit(X, y)
                X = self.encoders[col].transform(X).drop(columns=col)
        else:
            X = self.encoder_class(**kwargs).fit_transform(X, y)
        return X
    
    def transform(self, X: pd.DataFrame):
        X = X.copy()
        if self.singlecol: 
            for col in self.cols:
                X = self.encoders[col].transform(X).drop(columns=col)
        else:
            raise Exception('invalid')
        return X

    def __str__(self):
        return f"{self.encoder_class.__name__}()"


class OOFTE(BaseEstimator, TransformerMixin):
    def __init__(self, col, default=-1, random_state=1444):
        self.col = col
        self.colname = f"{self.col}_TE"
        self._d = {}
        self.random_state = random_state
        self.kfold = StratifiedKFold(n_splits=10, random_state=self.random_state, shuffle=True)
        self.default = -1
    
    def fit(self, X, y):
        X = X.reset_index(drop=True)
        new_x = X[[self.col]].copy().reset_index(drop=True)
        X.loc[:, self.colname] = 0
        for n_fold, (trn_idx, val_idx) in enumerate(self.kfold.split(new_x, y)):
            trn_x = new_x.iloc[trn_idx].copy()
            trn_x.loc[:, 'target'] = y.iloc[trn_idx]
            val_x = new_x.iloc[val_idx].copy()
            val_x.loc[:, 'target'] = y.iloc[val_idx]
            val = trn_x.groupby(self.col)['target'].mean().to_dict()
            # with default and other error handling
            val = defaultdict(lambda: -1, {
                k : v if not math.isnan(v) else -1 for k, v in val.items()  
            })
            self._d[n_fold] = val
        return self

    def fit_transform(self, X, y):
        X = X.reset_index(drop=True)
        new_x = X[[self.col]].copy().reset_index(drop=True)
        X.loc[:, self.colname] = 0
        for n_fold, (trn_idx, val_idx) in enumerate(self.kfold.split(new_x, y)):
            trn_x = new_x.iloc[trn_idx].copy()
            trn_x.loc[:, 'target'] = y.iloc[trn_idx]
            val_x = new_x.iloc[val_idx].copy()
            val_x.loc[:, 'target'] = y.iloc[val_idx]
            val = trn_x.groupby(self.col)['target'].mean().to_dict()
            # with default and other error handling
            val = defaultdict(lambda: -1, {
                k : v if not math.isnan(v) else -1 for k, v in val.items()  
            })
            self._d[n_fold] = val
            X.loc[val_idx, self.colname] = X.loc[val_idx, self.col].map(val)
        return X

    def transform(self, X):
        X.loc[:, self.colname] = 0
        for key, val in self._d.items():
            X.loc[:, self.colname] += X[self.col].map(val)

        X.loc[:, self.colname] /= key + 1
        return X


class LOOTE:
    def __init__(self, col):
        self.col = col
        pass

    def fit_transform(self, X, y):
        new_x = X[[self.col]].copy()
        new_x.loc[:, 'target'] = y
        a = (new_x.groupby(self.col)['target'].transform(np.sum) - y)\
            / new_x.groupby(self.col)['target'].transform(len)
        X.loc[:, 'TE'] = a
        self._d = X.groupby(self.col)['TE'].mean()
        return X

    def transform(self, X):
        X.loc[:, 'TE'] = X[self.col].map(self._d)
        return X
