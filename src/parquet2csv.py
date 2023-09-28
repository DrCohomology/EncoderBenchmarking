import pandas as pd

from pathlib import Path
from typing import Union

import src.utils as u


def parquet2csv(file_path: Union[str, Path]) -> None:
    """
    Convert a parquet file into a csv file.
    """
    file_path = Path(file_path)
    pd.read_parquet(file_path).to_csv(file_path.with_suffix(".csv"))


if __name__ == "__main__":
    results_file = u.RESULTS_DIR / "results.parquet"
    parquet2csv(results_file)


