# A benchmark of categorical encoders for binary classification

# Replicating results

## Installation
The necessary installation steps: 
1. install [Python 3.8](https://www.python.org/downloads/release/python-3810/);
2. create and activate a [virtual environment](https://python.land/virtual-environments/virtualenv);
3. install dependencies with `pip install -r requirements` 

## Execute the evaluation
1. Open a terminal with base directory `EncoderBenchmarking`;
1. Run `main_full_tuning.py`, `main_model_tuning.py`, and `main_no_tuning.py`; 

## Analysis and figures
1. `results_analysis.ipynb` provides the code we used to aanalyze our results. 
2. You will need Gurobi: https://pypi.org/project/gurobipy/

# Add stuff

## Add your own encoder, model, or quality metric 
You'll need to edit `config.py` with an instantiated encoder/model class with sklearn API: 
1. For an encoder, it immplements the `fit`, `transform`, and `fit_transform` methods;
2. For a model, it implments the `fit`, `predict`, and `fit_predict` methods. 

## Add your own aggregation strategy
This requires adding your method to `src.rank_utils.BaseAggregator`. 
Steps required: 
1. add the method, which operates on `df` and `rf` dataframes, described in `src.rank_utils.BaseAggregator`;
2. the method updates `self.aggrf` with a new column of scores for the ranking (they do not have to be actual rankings)
3. add your method name and method to `self.supported_strategies` and `self.increasing`. The key of `self.increasing` must be the same as in `self.aggrf`

  