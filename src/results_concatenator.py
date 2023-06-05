# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 10:50:29 2022

@author: federicom
"""

import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import src.utils as u
import warnings
from tqdm import tqdm

def clean_concatenated_dataset(df, remove_outdated_experiments=True):
    # clean the encoder names
    df.encoder = df.encoder.apply(lambda x: x.split('(')[0])
    df.encoder = df.encoder.replace("CollapseEncoder", "DropEncoder")
    df.encoder = df.encoder.apply(u.get_acronym, underscore=False)

    if remove_outdated_experiments:
        # remove old encoders
        remove = [enc for enc in df.encoder.unique() if ('6' in enc or ("RGLMM" not in enc and "GLMM" in enc))]
        df = df.loc[~ df.encoder.isin(remove)]

    # remove duplicates
    df = df.drop_duplicates()

    # group duplicates
    df = df.groupby("dataset encoder model scoring fold".split()).mean().reset_index()

    # add the fold column and check that it is ok -> every group in groups has the same fold numbers
    if "fold" not in df.columns:
        df["fold"] = list(range(5)) * int(len(df)/5)
    groups = ["dataset", "encoder", "model", "scoring"]
    test = df.groupby(by=groups).fold.sum()
    if (test != 10).any():
        raise Exception(f"The concatenated dataset has invalid fold column. {(test != 10).sum()}")

    return df

def concatenate_results(experiment_name, force=False, clean=True, remove_outdated_experiments=True, ignore_concatenated=False):
    """
    Concatenates ALL of the .csv files in the directory into  _concatenated file.
    Use as utility function to merge the datasets from a single experimental run (a set of experiments run together)
    Further concatenation of results between experimental runs is in src.result_concatenator and src.result_analysis
    clean: has effect only if force=True
    ignore_concatenated: has effect only if _concatenated.csv is in experiment_dir and force=True
    """

    warnings.filterwarnings("ignore")

    results_folder = os.path.join(u.RESULTS_DIR, experiment_name)
    results_concatenated = os.path.join(results_folder, "_concatenated.csv")
    results_files = glob.glob(os.path.join(results_folder, '*.csv'))

    if results_concatenated in results_files:
        if not force:
            return pd.read_csv(results_concatenated)
        if ignore_concatenated:
            results_files.pop(results_files.index(results_concatenated))

    print("Concatenating datasets")
    temps = []
    for filename in tqdm(results_files):
        temp = pd.read_csv(filename)
        if len(temp) == 0:
            raise ValueError(f"{filename} is empty.")
        if "fold" not in temp.columns:
            temp["fold"] = list(range(5)) * int(len(temp) / 5)
        temps.append(temp)

    df = pd.concat(temps, ignore_index=True).drop_duplicates()
    df = df.drop(columns="Unnamed: 0", errors="ignore")
    if clean:
        df = clean_concatenated_dataset(df, remove_outdated_experiments=remove_outdated_experiments)

    df.to_csv(results_concatenated, index=False)

    return df

if __name__ == "__main__":
    experiment_name = "main8_0131_LR_final"
    df = concatenate_results(experiment_name, force=True, clean=True, remove_outdated_experiments=False, ignore_concatenated=False)


