This directory contains the post-processed experimental results of our benchmark. 
The post-processing is described in detail in our paper as well as `result_analysis.ipynb`.

### `consensuses.parquet`
Wide-format dataframe storing the aggregated rankings, obtained with `src.rank_utils.Aggregator`.
Load it with `src.utils.load_aggrf`.
The table's index is the list of encoders, the table's columns is a pandas MultiIndex with all the combinations of 
experimental factors and aggregations (`model`, `tuning`, `scoring`, `aggregation`).

### `pw_X_.parquet`
Upper-triangle-format dataframe. 
Each cell contains the pairwise similarity between two columns of `rankings.parquet`. 
Load all such dataframes with `utils.load_similarity_dataframes`.
The dataframe's index and columns are the columns of `rankings.parquet` (`dataset`, `model`, `tuning`, `scoring`). 

### `pw_AGG_X.parquet` 
Upper-triangle-format dataframe. 
Each cell contains the pairwise similarity between two columns of `consensuses.parquet`. 
Load all such dataframes with `utils.load_aggregated_similarity_dataframes`.
The dataframe's index and columns are the columns of `consensuses.parquet` (`model`, `tuning`, `scoring`, `aggregation`). 


