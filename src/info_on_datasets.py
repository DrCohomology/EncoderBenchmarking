import os
import pandas as pd

from openml.datasets import get_dataset
from tqdm import tqdm

import src.utils as u

# name, ID, #obs, #vars, #bin, #cat, #num, %nan, most_classes


def count_binary(X):
    return len([col for col in X.columns if len(X[col].unique()) == 2])


def count_categorical(X):
    return len([col for col in X.select_dtypes(include=["category", "object"]).columns if len(X[col].unique()) > 2])


def count_numerical(X):
    return len([col for col in X.select_dtypes(exclude=["category", "object"]).columns])


def fraction_nans(X):
    return X.isna().sum().sum() / (X.shape[0] * X.shape[1])


def count_classes(X):
    return [len(X[col].unique()) for col in X.select_dtypes(include=["category", "object"]).columns]


if __name__ == "__main__":
    datasets = []

    for did in tqdm(u.DATASETS.values()):
        dataset = get_dataset(did)

        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )

        tmp = {
            "name": dataset.name,
            "ID": dataset.id,
            "num_obs": len(X),
            "num_feats": len(X.columns),
            "num_binary": count_binary(X),
            "num_categorical": count_categorical(X),
            "num_numeric": count_numerical(X),
            "%nans": fraction_nans(X),
            "most_classes": max(count_classes(X))


        }
        datasets.append(tmp)

    datasets = pd.DataFrame(datasets).sort_values("name")
    datasets.to_csv(os.path.join(u.RESULT_FOLDER, "datasets_info.csv"), index=False)



