# A benchmark of categorical encoders for binary classification

Repository for the paper _A benchmark of categorical encoders for binary classification_.

<img alt="Replicability of experimental results" src="analysis/plots/sample_model.png" title="Replicability"/>

# Replicating the experimental results

## Installation

### Requirements
1. Install [Python 3.8.10](https://www.python.org/downloads/release/python-3810/);
2. create and activate a [virtual environment](https://python.land/virtual-environments/virtualenv), we call it `venv`;
3. install dependencies with `pip install -r requirements`.

### Optional requirements
Our implementations of GLMM-based encoders require the [rpy2 module](https://pypi.org/project/rpy2/) and R to be installed.
The R version we used is `4.2.2`, with the `lme4` package version `1.1-31`.\
To aggregate results with Kemeny aggregation, install and configure [Gurobi](https://www.gurobi.com/) and its [Python API](https://pypi.org/project/gurobipy/).

## Execute the experiments
1. Open a terminal and navigate to `EncoderBenchmarking`;
2. activate `venv`;
3. configure the experimental parameters by editing `src\config.py`; 
4. run `src\main_full_tuning.py`, `src\main_model_tuning.py`, and `src\main_no_tuning.py`; the data is automatically fetched from [OpenML](https://www.openml.org/);
5. after execution, results in the form of a `.csv` file per factor combination, are stored in `analysis\experimental_results`, in the subfolders `full tuning`, `model tuning`, and `no tuning` --- not provided;
6. the abovementioned files are concatenated and saved into `analysis\experimental_results\results.parquet`, from which the rankings `rankings.parquet` are computed; 

## Analysis and figures
All of the code necessary to reproduce the analysis and the plots is available in the `analysis` folder.
`results_analysis.ipynb` depicts how to process `results.parquet` and `rankings.parquet` into the analysis' results stored in `analysis_results`.

# Additional information 

## Add custom Encoder, ML model, quality metric
The objects must implement the scikit-learn API:
- an `Encoder` implements the `fit`, `transform`, and `fit_transform` methods;
- a `Model` implements the `fit`, `predict`, and `fit_predict` methods;
- a `quality metric` is a function with signature `(y_true, y_pred) -> float` or `(y_true, y_score) -> float` if the metric requires probabilities (such as ROC AUC).

## Add a custom dataset
The data is automatically fetched from [OpenML](https://www.openml.org/) via the dataset `id`'s stored in `src\utils.DATASETS` and `src\utils.DATASETS_SMALL`. 
To add a new OpenML dataset, add a corresponding `name: id` entry to the abovementioned dictionaries.  
The benchmark is not configured to easily add non-OpenML datasets, requiring changes to the `src\main_[]_tuning.py` files. 
 
Edit the corresponding parameters in `src\config.py` to add the objects to the benchmark. 

[//]: # (## Aggregation strategy)

[//]: # (Modify `src.rank_utils.BaseAggregator` with a custom aggregation strategy: )

[//]: # (1. add the method, which operates on the `df` and `rf` dataframes, described in `src.rank_utils.BaseAggregator`;)

[//]: # (2. the method updates `self.aggrf` with a new column of scores for the ranking &#40;they do not have to be actual rankings&#41;)

[//]: # (3. add your method name and method to `self.supported_strategies` and `self.increasing`. The key of `self.increasing` must be the same as in `self.aggrf`)

  
