# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:59:19 2022

@author: federicom
"""

import functools
import math
import numpy as np
import pandas as pd

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
    PolynomialEncoder,
    SumEncoder,
    WOEEncoder,
)

from collections import defaultdict, Counter
from collections.abc import Iterable
from inspect import signature
from scipy.stats import ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

import src.utils as u


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, default=-1, **kwargs):
        self.default = default
        self.encoding = defaultdict(lambda: defaultdict(lambda: self.default))
        self.inverse_encoding = defaultdict(
            lambda: defaultdict(lambda: self.default))
        self.cols = None

    def fit(self, X: pd.DataFrame, y=None, **kwargs):
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        X = X.copy()
        return X

    def fit_transform(self, X: pd.DataFrame, y, **kwargs):
        return self.fit(X, y, **kwargs).transform(X, y, **kwargs)


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


class Discretized():
    """
    Discretizes the encoded values 
    The __str__ method ignores any parameter of the base_encoder
    For Encoder objects. Does not wor if base_encoder does not encode to pd.DataFrame
    """
    
    def __init__(self, base_encoder, how="minmaxbins", n_bins=10, **kwargs):
        self.base_encoder = base_encoder
        self.how = how
        self.n_bins = n_bins # only used if how != adaptive
        
        self.accepted_how = [
            "bins", 
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
            
            if self.how == "bins":
                dxs = np.floor(self.n_bins * xs) / self.n_bins
            elif self.how == "minmaxbins":
                m, M = xs.min(), xs.max()
                if m == M:
                    xs = np.ones_like(xs)
                else:
                    dxs = (xs-m)/(M-m) 
                    dxs = np.floor(self.n_bins * dxs) / self.n_bins
                    dxs = (M-m)*dxs + m
            else:
                raise NotImplementedError(f"Strategy {self.how} not yet implemented")
                
                
            self.discretization[col] = dict(zip(xs, dxs))
          
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
        
            


class DiscretizedTargetEncoder(Encoder):
    """
    Maps categorical values into the average target associated to them, 
    merging values with close avg target together. 
    "close" = 
    With another column to ensure invertibility of encoding.
    """

    def __init__(self, default=-1, how="bins", ttest_th=0.9, n_bins=10, **kwargs):
        super().__init__(default=default, **kwargs)
        self.accepted_how = [
            "bins", 
            "minmaxbins", 
            "adaptive"
        ]
        if how not in self.accepted_how:
            raise ValueError(f"{how} is not a valid value for parameter how")
        self.how = how
        # only used if how=adaptive
        self.ttest_th = ttest_th
        self.n_bins = n_bins

    def fit(self, X: pd.DataFrame, y, **kwargs):
        self.cols = X.columns
        # self.feature_names = self.cols.to_list()
        target_name = y.name
        X = X.join(y.squeeze())
        for col in self.cols:
            # -- Base: TargetEncoding
            temp = X.groupby(col)[target_name].mean().to_dict()

            # -- Discretize to bins of 0.1, from 0.0 to 1.0 both included
            if self.how == "bins":
                temp = {
                    val: np.floor(tgt*self.n_bins)/self.n_bins for val, tgt in temp.items()
                }
                
            # -- First rescale to minmax, then discretize
            elif self.how == "minmaxbins":
                m, M = min(temp.values()), max(temp.values())
                m -= 0.01
                M += 0.01
                if m < M:
                    # rescale to 0,1
                    temp = {
                        val: (tgt-m)/(M-m) for val, tgt in temp.items()
                    }
                    # discretize
                    temp = {
                        val: np.floor(tgt*self.n_bins)/self.n_bins for val, tgt in temp.items()
                    }
                    # upscale again
                    temp = {
                        val: (M-m)*tgt + m for val, tgt in temp.items()
                    }
                else:
                    pass
                
            #!!! -- test for consecutive averages to be stat. equal
            elif self.how == "adaptive":
                # group together close instances, do not touch the rest
                # 1. test if two consecutive values have stat. different target (t-test 90%)
                # requires: average target, stddev and sample size
                # the test has H0: equal avg, so we want to NOT reject H0
                # this means requiring a pvalue > 90% (sketchy but it'll do)
                # 2. group the 'equal' values
                
                # order by target
                avg = Counter(temp).most_common() # [(k1, v1), ...]
                # list of observations
                x = X.groupby(col)[target_name].agg(list).to_dict()     
                
                # - Iterate over avg and see if two samples are stat diff
                sim = []
                for i in range(1, len(avg)):
                    # retrieve corresponding samples
                    x1 = x[avg[i-1][0]]
                    x2 = x[avg[i][0]]
                    if ttest_ind(x1, x2, equal_var=False, nan_policy='omit').pvalue > self.ttest_th:
                        sim.append({avg[i-1][0], avg[i][0]})
                    
                # - Iteratively merge sets of similar values with non empty 
                # intersection
                # EQUIVALENT: check if the last element of simI-1 == first element of simI
                # due to the way sim is built from ordered lists
                if len(sim) > 1:
                    i = 1
                    while True:
                        if sim[i-1].intersection(sim[i]) != set():
                            sim[i-1] = sim[i-1].union(sim[i])
                            del sim[i]
                        else:
                            i += 1
                        if i == len(sim):
                            break
                    
                # - Encode each of those with the average target of the group
                tempsim = {
                    tuple(grp) : np.mean([temp[val] 
                                          for val in grp]) 
                    for grp in sim    
                }

                # assign each val to its group
                v2g = {}
                for val in temp.keys():
                    for grp in tempsim.keys():
                        if val in grp:
                            v2g[val] = grp
                            break

                # assign each val to the group avg
                temp.update({
                    val : tempsim[grp] for val, grp in v2g.items()
                })
                
            else:
                # Check on value of how is enforced before
                pass

            # -- Save
            self.encoding[col].update(temp)

        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].apply(lambda cat: self.encoding[col][cat])
        return X

    def __str__(self):
        return f"DiscretizedTargetEncoder{self.how.capitalize()}{self.n_bins}"


class CollapseEncoder(Encoder):
    """
    Evey categorical value is mapped to 1
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        self.cols = X.columns
        return pd.DataFrame(np.ones(len(X)), index=X.index, columns=['cat'])


class CVRegularized(Encoder):
    """
    Encodes every test fold with an Encoder trained on the training fold.
    Default value of base_encoder is supposed to be np.nan to allow the usage of
    default encoder trained on the whole dataset. 
    
    The __str__ method ignores any parameter of the base_encoder
    """
    
    def __init__(self, base_encoder, n_splits=5, random_state=1444, default=-1, **kwargs):
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
    
    
class CVBlowUp(Encoder):
    """
    Has a column for every fold
    
    The __str__ method ignores any parameter of the base encoder
    """

    def __init__(self, base_encoder, n_splits=5, random_state=1444, **kwargs):
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


class SmoothedTE(Encoder):
    """
    TargetEncoder with smoothing parameter. 
    From https://github.com/rapidsai/deeplearning/blob/main/RecSys2020Tutorial/03_3_TargetEncoding.ipynb

    """

    def __init__(self, default=-1, w=10, **kwargs):
        super().__init__(default=default, **kwargs)
        self.w = w

    def fit(self, X: pd.DataFrame, y, **kwargs):
        self.cols = X.columns
        target_name = y.name
        X = X.join(y.squeeze())
        global_mean = y.mean()
        for col in self.cols:
            temp = X.groupby(col).agg(['sum', 'count'])
            temp['STE'] = (temp.target['sum'] + self.w *
                           global_mean) / (temp.target['count'] + self.w)
            # temp['TE'] = temp.target['sum'] / temp.target['count']
            self.encoding[col].update(temp['STE'].to_dict())
        return self

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
