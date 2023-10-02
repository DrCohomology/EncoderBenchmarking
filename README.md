# A benchmark of categorical encoders for binary classification

Repository for the paper [A benchmark of categorical encoders for binary classification](https://arxiv.org/abs/2307.09191), 
accepted at NeurIPS 2023, Datasets and Benchmarks track.

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

## Configure, add to, and run the experiments
In `src/README.md`.

## Analysis and figures
All of the code necessary to reproduce the analysis and the plots is available in the `analysis` folder.
`results_analysis.ipynb` depicts how to process `results.parquet` and `rankings.parquet` into the analysis' results stored in `analysis_results`.

[//]: # (## Aggregation strategy)

[//]: # (Modify `src.rank_utils.BaseAggregator` with a custom aggregation strategy: )

[//]: # (1. add the method, which operates on the `df` and `rf` dataframes, described in `src.rank_utils.BaseAggregator`;)

[//]: # (2. the method updates `self.aggrf` with a new column of scores for the ranking &#40;they do not have to be actual rankings&#41;)

[//]: # (3. add your method name and method to `self.supported_strategies` and `self.increasing`. The key of `self.increasing` must be the same as in `self.aggrf`)

  
