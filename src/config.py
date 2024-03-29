import src.encoders as e
import src.utils as u

from functools import reduce
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# --- General parameters
RANDOM_STATE = 1
MAIN_PARAMETERS = {
    "n_splits": 5,  # splits of cross-validation
    "timeout": 6000,  # maximum allowed run time in seconds
}
NUM_PROCESSES = 1  # pass to joblib.Parallel for parallel execution


# --- Encoders
rlibs = None
std = [e.BinaryEncoder(), e.CatBoostEncoder(), e.CountEncoder(), e.DropEncoder(), e.MinHashEncoder(), e.OneHotEncoder(),
       e.OrdinalEncoder(), e.RGLMMEncoder(rlibs=rlibs), e.SumEncoder(), e.TargetEncoder(), e.WOEEncoder()]
cvglmm = [e.CVRegularized(e.RGLMMEncoder(rlibs=rlibs), n_splits=ns) for ns in [2, 5, 10]]
cvte = [e.CVRegularized(e.TargetEncoder(), n_splits=ns) for ns in [2, 5, 10]]
buglmm = [e.CVBlowUp(e.RGLMMEncoder(rlibs=rlibs), n_splits=ns) for ns in [2, 5, 10]]
bute = [e.CVBlowUp(e.TargetEncoder(), n_splits=ns) for ns in [2, 5, 10]]
dte = [e.Discretized(e.TargetEncoder(), how="minmaxbins", n_bins=nb) for nb in [2, 5, 10]]
binte = [e.PreBinned(e.TargetEncoder(), thr=thr) for thr in [1e-3, 1e-2, 1e-1]]
me = [e.MeanEstimateEncoder(m=m) for m in [1e-1, 1, 10]]
ENCODERS = reduce(lambda x, y: x+y, [std, cvglmm, cvte, buglmm, bute, dte, binte, me])

# --- Datasets
DATASET_NAMES = {
    "no tuning": list(u.DATASETS.keys()),
    "model tuning": list(u.DATASETS.keys()),
    "full tuning": list(u.DATASETS_SMALL.keys())
}
DATASET_IDS = {
    "no tuning": list(u.DATASETS.values()),
    "model tuning": list(u.DATASETS.values()),
    "full tuning": list(u.DATASETS_SMALL.values())
}

# --- Models
MODELS = {
    "no tuning": [
        DecisionTreeClassifier(random_state=RANDOM_STATE + 2, max_depth=5),
        SVC(random_state=RANDOM_STATE + 4, C=1.0, kernel="rbf", gamma="scale", probability=True),
        KNeighborsClassifier(n_neighbors=5),
        LogisticRegression(max_iter=100, random_state=RANDOM_STATE + 6, solver="lbfgs")],
    "model tuning": [
        DecisionTreeClassifier(random_state=RANDOM_STATE+2),
        KNeighborsClassifier(),
        LogisticRegression(max_iter=100, random_state=RANDOM_STATE+6, solver="lbfgs")
    ],
    "full tuning": [
        DecisionTreeClassifier(random_state=RANDOM_STATE+2),
        SVC(random_state=RANDOM_STATE+4, probability=True),
        KNeighborsClassifier(),
        LogisticRegression(max_iter=100, random_state=RANDOM_STATE+6, solver="lbfgs"),
        LGBMClassifier(random_state=RANDOM_STATE+3, n_estimators=3000, metric="None"),  # LGBM needs early_stopping
    ]
}

# --- Quality metrics
SCORINGS = [accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score]

# --- Other pre-processing classes
SCALERS = [RobustScaler()]
IMPUTERS_CAT = [e.DFImputer(SimpleImputer(strategy="most_frequent"))]
IMPUTERS_NUM = [e.DFImputer(SimpleImputer(strategy="median"))]
