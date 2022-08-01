# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 15:50:06 2022

@author: federicom
"""
from tqdm import tqdm

import pandas as pd

from utils import DATASET_FOLDER


def load_adult():
    df = pd.read_csv(DATASET_FOLDER + "/adult/adult.csv")

    target = (df["income"] == ">50K").rename("target")
    df.drop("income", axis=1, inplace=True)

    return df, target


def load_credit():
    dftr = pd.read_csv(DATASET_FOLDER + "/credit/application_train.csv")
    dfte = pd.read_csv(DATASET_FOLDER + "/credit/application_test.csv")

    df = pd.concat([dftr, dfte], ignore_index=True)

    target = (df["TARGET"] == 1).rename("target")
    todrop = ["SK_ID_CURR", "TARGET"]
    df.drop(todrop, axis=1, inplace=True)

    return df, target


def load_kick():
    dftr = pd.read_csv(DATASET_FOLDER + "/kick/training.csv")
    dfte = pd.read_csv(DATASET_FOLDER + "/kick/test.csv")

    df = pd.concat([dftr, dfte], ignore_index=True)

    target = (df["IsBadBuy"] == 1).rename("target")
    todrop = ["RefId", "IsBadBuy"]
    df.drop(todrop, axis=1, inplace=True)

    return df, target


def load_promotion():
    dftr = pd.read_csv(DATASET_FOLDER + "/promotion/train_LZdllcl.csv")
    dfte = pd.read_csv(DATASET_FOLDER + "/promotion/test_2umaH9m.csv")

    df = pd.concat([dftr, dfte], ignore_index=True)

    target = (df["is_promoted"] == 1).rename("target")
    todrop = ["is_promoted", "employee_id"]
    df.drop(todrop, axis=1, inplace=True)

    return df, target


def load_telecom():
    df = pd.read_csv(DATASET_FOLDER + "/telecom/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    target = (df["Churn"] == "Yes").rename("target")
    todrop = ["Churn", "customerID", "TotalCharges"]
    df.drop(todrop, axis=1, inplace=True)

    return df, target


def load_kaggle_cat_dat_2():
    dftr = pd.read_csv(DATASET_FOLDER + "/kaggle_cat_dat_2/train.csv")
    dfte = pd.read_csv(DATASET_FOLDER + "/kaggle_cat_dat_2/test.csv")

    df = pd.concat([dftr, dfte], ignore_index=True)

    target = (df["target"] == 1).rename("target")
    todrop = ["target", "id"]
    df.drop(todrop, axis=1, inplace=True)

    return df, target


def load_kaggle_cat_dat_1():
    dftr = pd.read_csv(DATASET_FOLDER + "/kaggle_cat_dat_1/train.csv")
    dfte = pd.read_csv(DATASET_FOLDER + "/kaggle_cat_dat_1/test.csv")

    df = pd.concat([dftr, dfte], ignore_index=True)

    target = (df["target"] == 1).rename("target")
    todrop = ["target", "id"]
    df.drop(todrop, axis=1, inplace=True)

    return df, target


def load_kaggle_used_car():
    # https://www.kaggle.com/austinreese/craigslist-carstrucks-data
    df = pd.read_csv(DATASET_FOLDER + "/kaggle_used_car/vehicles.csv")

    target = (df["price"]).rename("target")
    todrop = ["price", "id", "url", "description"]
    df.drop(todrop, axis=1, inplace=True)

    return df, target


def load_openml_ames_housing():
    df, target = fetch_openml(data_id=42165, as_frame=True, return_X_y=True)

    target.rename("target", inplace=True)

    return df, target


def load_datasets():
    for folder in tqdm(
        [
            f
            for f in os.listdir(DATASET_FOLDER)
            if os.path.isdir(os.path.join(DATASET_FOLDER, f))
        ]
    ):
        if f"{folder}.csv" in os.listdir(DATASET_FOLDER):
            continue
        if "adult" in folder:
            df, target = load_adult()
        elif "credit" in folder:
            df, target = load_credit()
        elif "kick" in folder:
            df, target = load_kick()
        elif "promotion" in folder:
            df, target = load_promotion()
        elif "telecom" in folder:
            df, target = load_telecom()
        elif "kaggle_cat_dat_2" in folder:
            df, target = load_kaggle_cat_dat_2()
        elif "kaggle_cat_dat_1" in folder:
            df, target = load_kaggle_cat_dat_1()
        elif "kaggle_used_car" in folder:
            df, target = load_kaggle_used_car()
        elif "openml_ames_housing" in folder:
            df, target = load_openml_ames_housing()
        else:
            continue
        dfcat = df.select_dtypes(["object"])
        dfnum = df.select_dtypes(["number"])

        dfcat.columns = [f"cat_{i}" for i, _ in enumerate(dfcat)]
        dfnum.columns = [f"num_{i}" for i, _ in enumerate(dfnum)]

        df = dfcat.join(dfnum).join(target)
        df.to_csv(f"{DATASET_FOLDER}/{folder}.csv", index=False)