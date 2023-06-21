import json
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from collections import defaultdict
from functools import reduce
from importlib import reload
from itertools import product
from pathlib import Path
from scipy.stats import kendalltau, iqr
from scikit_posthocs import posthoc_nemenyi_friedman
from tqdm import tqdm
from typing import List, Union

import src.encoders as e
import src.utils as u
import src.rank_utils as ru
import src.rank_metrics as rm

df, rf = u.load_df_rf()

# sensitivity to change in model (everything else fixed)
jaccard_model, rho_model = u.pairwise_similarity_wide_format(rf,
                                                             simfuncs=[rm.jaccard_best,
                                                                       rm.spearman_rho],
                                                             shared_levels=[0, 2, 3])

# # sensitivity to change in scoring (everything else fixed)
# jaccard_scoring, rho_scoring = u.pairwise_similarity_wide_format(rf,
#                                                                  simfuncs=[rm.jaccard_best,
#                                                                            rm.spearman_rho],
#                                                                  shared_levels=[0, 2, 3])
#
# # sensitivity to change in tuning (everything else fixed)
# jaccard_tuning, rho_tuning = u.pairwise_similarity_wide_format(rf,
#                                                                simfuncs=[rm.jaccard_best,
#                                                                          rm.spearman_rho],
#                                                                shared_levels=[0, 2, 3])
#
# jaccard = reduce(lambda x, y: x.fillna(y), [jaccard_model, jaccard_scoring, jaccard_tuning])
# rho = reduce(lambda x, y: x.fillna(y), [rho_model, rho_tuning, rho_scoring])
