# Execute the experiments
1. Open a terminal and navigate to `EncoderBenchmarking`;
2. activate `venv`;
3. configure the experiment as detailed below;
4. run `src\main_full_tuning.py`, `src\main_model_tuning.py`, and `src\main_no_tuning.py`; the data is automatically fetched from [OpenML](https://www.openml.org/);
5. results are stored in the experiment directory specified in `utils.RESULTS_DIR / [experiment_name]`;
    - the directory is created automatically;
    - before running the experiment, the existing results are checked to avoid repeating existing runs;
      - this behavior can be controlled within `main_X.py`;
      - _if you run an experiment multiple times_, concatenate the results first (step 6.) and
        move `_concatenated.csv` in the experiment directory
6. the results of an experimental run have with names in the format:
     - _full tuning_: `[dataset]_[encoder]_[model]_[scoring].csv`,
     - _model_tuning_: `[dataset]_[encoder]_[model].csv`,
     - _no_tuning_: `[dataset]_[encoder].csv`.
7. concatenate the files into a single dataframe `_concatenated.csv` with `results_concatenator.py`;
8. compute the dataframe of rankings `rankings.csv` from `_concatenated.csv`; 
9. rename `_concatenated.csv` to `results.csv`, convert `results.csv` and `rankings.csv` to parquet.

# Configure an experiment
You can configure the experiment from the following files:
- `config.py`: 
  - ML models;
  - scorings;
  - encoders;
  - max run time;
  - number of parallel processes.
- `utils.py`:
  - datasets;
  - root directory for experimental results.
- `main_X.py`:
  - name of the experiment; 

## Add custom Encoder, ML model, quality metric
The objects must implement the scikit-learn API:
- an `Encoder` implements the `fit`, `transform`, and `fit_transform` methods;
- a `Model` implements the `fit`, `predict`, and `fit_predict` methods;
- a `quality metric` is a function with signature  (1) `(y_true, y_pred) -> float` or (2) `(y_true, y_score) -> float` if the metric requires probabilities (such as ROC AUC).
  - by default, metrics are treated as of type (1). To add a metric of type (2), you need to change the following LoC in `main_X.py`:
  `if scoring.__name__ == "roc_auc_score"` to `if scoring.__name__ in ["roc_auc_score", YOUR_METRIC]`

Edit the corresponding parameters in `src\config.py` to run the benchmark with the new objects. 

## Add a custom dataset
The data is automatically fetched from [OpenML](https://www.openml.org/) via the dataset `id`'s stored in `src\utils.DATASETS` and `src\utils.DATASETS_SMALL`. 
To add a new OpenML dataset, add a corresponding `name: id` entry to the abovementioned dictionaries.  
_The scripts do not support quick integration of datasets not from OpenML._ 





