## To run an experiment: 
1. configure the experiment; 
2. run one of the `main_X_tuning.py`'s;
3. results are stored in the experiment directory `utils.RESULTS_DIR / [experiment_name]`;
    - the directory is created automatically;
    - before running the experiment, the existing results are checked to avoid repeating existing runs;
      - this behavior can be controlled within `main_X_tuning.py`;
      - _if you run an experiment multiple times_, concatenate the results first (as in step 5.) and
        put `_concatenated.csv` in the experiment directory
4. the results are in multiple files, each corresponding to an experimental run, with names in the format:
     - _full tuning_: `[dataset]_[encoder]_[model]_[scoring].csv`,
     - _model_tuning_: `[dataset]_[encoder]_[model].csv`,
     - _no_tuning_: `[dataset]_[encoder].csv`.
5. concatenate the files into a single dataframe `_concatenated.csv` with `results_concatenator.py`;
6. compute the dataframe of rankings `rankings.csv` from `_concatenated.csv`; 
7. rename `_concatenated.csv` to `results.csv`, convert `results.csv` and `rankings.csv` to parquet.

## To configure an experiment
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
- `main_X_tuning.py`:
  - name of the experiment; 





