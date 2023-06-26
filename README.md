# A benchmark of categorical encoders for binary classification
 
 _The final version will be uploaded by 25.06.2023._ 

Reproduce results: 
1. Install Python 3.8 
1. Install the requirements in an activated Python 3.8 environment  
1. Run `main_full_tuning.py`, `main_model_tuning.py`, and `main_no_tuning.py`, results and logs are saved in `?`
2. Analyze the results with `result_analysis.ipynb`


In order to reproduce th Kemeny rank aggregation, we used a Gurobi solver with academic license, with the Python API 
provided by `gurobipy`.
If you don't have an academic license, we provide the code to run aggregation with free cvxpy solvers. 















