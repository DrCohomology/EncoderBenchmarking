import glob
import pandas as pd

from tqdm import tqdm

import src.utils as u


def concatenate_results(experiment_name, force=False, ignore_concatenated=False):
    """
    Concatenates ALL of the .csv files in the directory into  _concatenated file.
    Use as utility function to merge the results from single experimental runs into one dataframe.
    Further concatenation of results between experimental runs is in src.result_concatenator and src.result_analysis.

    if force == False, return just the concatenated dataset (if any).
    if ignore_concatenated == True, the concatenated dataset is ignored.
    """

    if experiment_name is None:
        raise ValueError("experiment_name must be the name of an existing subdirectory of src.utils.RESULTS_DIR.")

    experiment_dir = u.RESULTS_DIR / experiment_name
    results_concatenated = str(experiment_dir / "_concatenated.csv")
    results_files = glob.glob(str(experiment_dir / "*.csv"))

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
        temps.append(temp)

    df = pd.concat(temps, ignore_index=True).drop_duplicates().drop(columns="Unnamed: 0", errors="ignore")
    return df.to_csv(results_concatenated, index=False) or df


if __name__ == "__main__":
    df = concatenate_results("example_experiment_fulltuning", force=True, ignore_concatenated=True)
