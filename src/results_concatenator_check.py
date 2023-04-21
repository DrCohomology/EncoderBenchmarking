"""
Goal: concatenate results, see missing runs, aggregate different experiments together
"""
import itertools
import os
import pandas as pd

from functools import reduce
from importlib import reload
from tqdm import tqdm

import src.encoders as e
import src.utils as u
import src.results_concatenator as rc

reload(e), reload(u), reload(rc)

rlibs = None
std = [e.BinaryEncoder(), e.CatBoostEncoder(), e.CountEncoder(), e.DropEncoder(), e.MinHashEncoder(), e.OneHotEncoder(),
       e.OrdinalEncoder(), e.RGLMMEncoder(rlibs=rlibs), e.SumEncoder(), e.TargetEncoder(), e.WOEEncoder()]
cvglmm = [e.CVRegularized(e.RGLMMEncoder(rlibs=rlibs), n_splits=ns) for ns in [2, 5, 10]]
cvte = [e.CVRegularized(e.TargetEncoder(), n_splits=ns) for ns in [2, 5, 10]]
buglmm = [e.CVBlowUp(e.RGLMMEncoder(rlibs=rlibs), n_splits=ns) for ns in [2, 5, 10]]
bute = [e.CVBlowUp(e.TargetEncoder(), n_splits=ns) for ns in [2, 5, 10]]
dte = [e.Discretized(e.TargetEncoder(), how="minmaxbins", n_bins=nb) for nb in [2, 5, 10]]
binte = [e.PreBinned(e.TargetEncoder(), thr=thr) for thr in [1e-3, 1e-2, 1e-1]]
ste = [e.MEstimate(m=m) for m in [1e-1, 1, 10]]
encoders = reduce(lambda x, y: x+y, [std, cvglmm, cvte, buglmm, bute, dte, binte, ste])
encoders = set(u.get_acronym(str(x), underscore=False) for x in encoders)

scorings = ["accuracy_score", "roc_auc_score", "f1_score"]

#%% main6 results

main6 = [
    "main6 results with tuning and 6-encoders",
    "main6_0110_adult",
    "main6_0116_SumEncoder",
    "main6_WoEE_OE_LR_STE",
    "main6_0125_MEstimate",
    "main6_0322_LGBM"
]

datasets = set(u.DATASETS_SMALL.keys())
models = {
    "DecisionTreeClassifier",
    "SVC",
    "KNeighborsClassifier",
    "LogisticRegression",
    "LGBMClassifier"
}

for expdir in main6:
    rc.concatenate_results(expdir, force=False, clean=True, remove_outdated_experiments=False, ignore_concatenated=True)

dfs = []
for expdir in main6:
    temp = rc.concatenate_results(expdir, force=False, clean=True, remove_outdated_experiments=False)
    temp.rename(columns={"tuning_scores": "tuning_score", "cv_scores": "cv_score"}, inplace=True)

    dfs.append(temp.loc[temp.encoder.isin(encoders)])

df = pd.concat(dfs, ignore_index=True)

# check encoders, models
# print("Missing encoders: ", encoders - set(df.encoder.unique()))
# print("Missing models:   ", models - set(df.model.unique()))
# print("Missing datasets: ", datasets - set(df.dataset.unique()))
#
# print("Additional encoders: ", set(df.encoder.unique()) - encoders)
# print("Additional models:   ", set(df.model.unique()) - models)
# print("Additional datasets: ", set(df.dataset.unique()) - datasets)

experiments = set(itertools.product(datasets, encoders, models, scorings))
run_experiments = set(df.groupby("dataset encoder model scoring".split()).groups)

missing = experiments - run_experiments
print("Missing runs: ", len(missing))

# check duplicates
if len(df) != len(df.drop_duplicates()):
    print("Duplicates: ", len(df), len(df.drop_duplicates()))

# check models
for model in df.model.unique():
    hpar = "model__"
    if model == "SVC":
        hpar += "C"
    elif model == "DecisionTreeClassifier":
        hpar += "max_depth"
    elif model == "KNeighborsClassifier":
        hpar += "n_neighbors"
    elif model == "LogisticRegression":
        hpar += "C"
    elif model == "LGBMClassifier":
        hpar += "n_estimators"

    if df.loc[df.model == model][hpar].isna().sum() > 0:
        print("Missing hpars: ", model, df.loc[df.model == model][hpar].isna().sum())

df["tuning"] = "full"
df.to_csv(os.path.join(u.RESULT_FOLDER, "main6_final.csv"), index=False)

#%% main8 results

main8 = [
    # "main8_29dats",
    "main8_50_no_logreg",
    'main8_1222_LR',
    'main8_STEnew',
    "main8_0116_SumEncoder",
    "main8_woee",
    "main8_0125_MEstimate",
    "main8_1215_29dats",
    "main8_50_no_logreg_no_svc",
    "main8_0131_DT_final",
    "main8_0131_kNN_final",
    "main8_0131_LR_final",
    "main8_0131_SVC_final"
]

datasets = set(u.DATASETS)
models = {
    "DecisionTreeClassifier",
    "SVC",
    "KNeighborsClassifier",
    "LogisticRegression",
}

for expdir in main8:
    # print(expdir)
    rc.concatenate_results(expdir, force=False, clean=True, remove_outdated_experiments=False, ignore_concatenated=True)

dfs = []
for expdir in main8:
    temp = rc.concatenate_results(expdir, force=False, clean=True, remove_outdated_experiments=False)
    temp.rename(columns={"tuning_scores": "tuning_score", "cv_scores": "cv_score"}, inplace=True)

    # print(expdir)
    # print("Check unnecessary encoders: ", set(temp.encoder.unique()) - encoders)
    # print("Check duplicates: ", len(temp), len(temp.drop_duplicates()))
    # print()

    dfs.append(temp.loc[temp.encoder.isin(encoders)])

df = pd.concat(dfs, ignore_index=True)

# check encoders, models
# print("Missing encoders: ", encoders - set(df.encoder.unique()))
# print("Missing models:   ", models - set(df.model.unique()))
# print("Missing datasets: ", datasets - set(df.dataset.unique()))

# print("Additional encoders: ", set(df.encoder.unique()) - encoders)
# print("Additional models:   ", set(df.model.unique()) - models)
# print("Additional datasets: ", set(df.dataset.unique()) - datasets)

experiments = set(itertools.product(datasets, encoders, models))
run_experiments = set(df.groupby("dataset encoder model".split()).groups)

missing = pd.DataFrame(list(experiments - run_experiments), columns="dataset encoder model".split())\
            .sort_values("dataset encoder model".split()).reset_index(drop=True)



print(f"Missing runs: {missing.shape[0]} / {len(experiments)}")

for model in models:
    print(f"{len(missing.query('model == @model'))} Missing for {model}")



# check duplicates
if len(df) != len(df.drop_duplicates()):
    print("Duplicates: ", len(df), len(df.drop_duplicates()))

df = df.drop_duplicates()
df["tuning"] = "no"
df.to_csv(os.path.join(u.RESULT_FOLDER, "main8_final.csv"), index=False)

#%% main9 results

main9 = [
    "main9_EB1222_logreg_tuning",
    'main9_model_tuning_no_SVC',
    'main9_0116_SumEncoder',
    "main9_0125_MEstimate",
    "main9_0306_LR_final",
    "main9_0306_DT_final",
    "main9_0306_kNN_final"
]
datasets = set(u.DATASETS.keys())
models = {
    "DecisionTreeClassifier",
    "KNeighborsClassifier",
    "LogisticRegression",
}

for expdir in main9:
    print(expdir)
    rc.concatenate_results(expdir, force=False, clean=True, remove_outdated_experiments=False, ignore_concatenated=False)

dfs = []
for expdir in main9:
    temp = rc.concatenate_results(expdir, force=False, clean=True, remove_outdated_experiments=False)
    temp.rename(columns={"tuning_scores": "tuning_score", "cv_scores": "cv_score"}, inplace=True)

    # print(expdir)
    # print("Check unnecessary encoders: ", set(temp.encoder.unique()) - encoders)
    # print("Check duplicates: ", len(temp), len(temp.drop_duplicates()))
    # print()

    # dfs.append(temp.loc[temp.encoder.isin(encoders)])
    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)

# check encoders, models
# print("Missing encoders: ", encoders - set(df.encoder.unique()))
# print("Missing models:   ", models - set(df.model.unique()))
# print("Missing datasets: ", datasets - set(df.dataset.unique()))
#
# print("Additional encoders: ", set(df.encoder.unique()) - encoders)
# print("Additional models:   ", set(df.model.unique()) - models)
# print("Additional datasets: ", set(df.dataset.unique()) - datasets)
experiments = set(itertools.product(datasets, encoders))
run_experiments = set(df.groupby("dataset encoder".split()).groups)
missing = pd.DataFrame(list(experiments - run_experiments), columns="dataset encoder".split())\
            .sort_values("dataset encoder".split()).reset_index(drop=True)
print(f"Missing runs: {len(missing)} / {len(experiments)}")


# check duplicates
if len(df) != len(df.drop_duplicates()):
    print("Duplicates: ", len(df), len(df.drop_duplicates()))

# check models
for model in df.model.unique():
    hpar = ""
    if model == "DecisionTreeClassifier":
        hpar += "max_depth"
        df.loc[(df.model == model), hpar] = df.loc[(df.model == model), hpar].fillna("None")
    elif model == "KNeighborsClassifier":
        hpar += "n_neighbors"
    elif model == "LogisticRegression":
        hpar += "C"

    if df.loc[df.model == model][hpar].isna().sum() > 0:
        print("Missing hpars: ", model, df.loc[df.model == model][hpar].isna().sum(), " out of ", df.loc[df.model == model].shape[0])

df = df.drop_duplicates()
df["tuning"] = "model"
df.to_csv(os.path.join(u.RESULT_FOLDER, "main9_final.csv"), index=False)










