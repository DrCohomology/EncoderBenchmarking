This directory contains the experimental results from our benchmark, obtained as described in `src.README.md`.

### `results.parquet` 
Long-format dataframe storing the results of the experiments. Load it with `src.utils.load_df`.
The table has columns:
- `dataset`: OpenML id of the dataset;
- `model`: ML model used to evaluate the encoder;
- `tuning`: how the model-encoder pipeline was tuned;
- `scoring`: evaluation metric;
- `encoder`: the encoder;
- `cv_score`: average of a 5-fold cross-validated evaluation.

### `rankings.parquet`
Wide-format dataframe storing the results as rankings, obtained with `src.utils`. Load it with `src.utils.load_rf`.
The table's index is the list of encoders, the table's columns is a pandas MultiIndex with all the combinations of 
experimental factors (`dataset`, `model`, `tuning`, `scoring`).

