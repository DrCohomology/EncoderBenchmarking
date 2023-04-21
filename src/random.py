import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry
import seaborn as sns

from collections import defaultdict
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_log_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from Levenshtein import distance as levd

import src.encoders as e


x = pd.DataFrame(list(zip(list("aaabbcde"), [1.5, 0.3, 1, 1.9, 0, 0.5, 0.1, 1.4])), columns=["A", "target"])
encoder = e.Discretized(e.TargetEncoder(), n_bins=6)
encoder = e.PreBinned(e.TargetEncoder(), thr=0.5)

xe = encoder.fit_transform(pd.DataFrame(x.A), x.target)
print(xe)
print(encoder.base_binner.binnings)
#%%



x = pd.DataFrame(list(zip(["a", "a", "a", "b", "b", "c"], [1, 0, 0, 1, 1, 1])), columns=["A", "y"])

encoder = e.MEstimateEncoder(m=10)
xe = encoder.fit_transform(x.A, x.y)

print(xe)




#%%
x = [
    "France",
    "Germany",
    "Germany_south",
    "Frence",
    "Italy"
]

edit = {
    (s, t): levd(s, t) for s, t in itertools.product(x, repeat=2)
}



#%% what the hell does SumEncoder do?

x = pd.DataFrame(["a", "a", "b", "b", "c", "d", "c"], columns=["A"])

encoder = e.SumEncoder()
xe = encoder.fit_transform(x["A"])

x = x.join(xe)

#%%
"""
Data from 
https://developers.google.com/public-data/docs/canonical/countries_csv
https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
https://data.worldbank.org/indicator/NY.GDP.PCAP.CD
"""

DATA_DIR = "C:\\Data\\_datasets"

df = pd.read_csv(DATA_DIR + "\\countries_gdp.csv")
df["coordinates"] = pd.Series(zip(df.longitude, df.latitude))

X = df.drop(columns="GDP_PC")
y = df["GDP_PC"]

matrix_edit = np.array([
    [levd(x, y) for y in df.name] for x in df.name]
)

matrix_geo = np.array([
    [(x1-y1) ** 2 + (x2-y2) ** 2 for (y1, y2) in df.coordinates] for (x1, x2) in df.coordinates]
)

k = 10
scoring = r2_score
weights = "distance"
scores_edit = defaultdict(lambda: [])
scores_geo = defaultdict(lambda: [])
train_shares = np.linspace(0.1, 0.6, 10).round(2)
for tts in train_shares:
    for _ in range(50):
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=tts)

        matrix_edit_train = matrix_edit[Xtrain.index, :][:, Xtrain.index]
        model_edit = KNeighborsRegressor(metric="precomputed", n_neighbors=k, weights=weights).fit(matrix_edit_train, ytrain)
        pred_edit = model_edit.predict(matrix_edit[Xtest.index, :][:, Xtrain.index])
        scores_edit[tts].append(scoring(ytest, pred_edit))

        matrix_geo_train = matrix_geo[Xtrain.index, :][:, Xtrain.index]
        model_geo = KNeighborsRegressor(metric="precomputed", n_neighbors=k, weights=weights).fit(matrix_geo_train, ytrain)
        pred_geo = model_geo.predict(matrix_geo[Xtest.index, :][:, Xtrain.index])
        scores_geo[tts].append(scoring(ytest, pred_geo))


scores_geo = pd.DataFrame(scores_geo)
scores_edit = pd.DataFrame(scores_edit)

fig, ax = plt.subplots(1, 1, sharex="all", sharey="all", figsize=(5, 5))

scores_geo["Encoding"] = "Coordinates"
scores_edit["Encoding"] = "Name"

scores = pd.concat([scores_geo, scores_edit])
score = pd.melt(scores, id_vars=["Encoding"], value_vars=train_shares, var_name="Share", value_name="R2")

sns.boxplot(score, x="Share", y="R2", hue="Encoding", ax=ax)
ax.set_ylabel("R2")
ax.set_xlabel("Share of training data")
ax.set_ylim(-1, 1)

plt.legend()
plt.tight_layout()
plt.show()


#%%

gdp = pd.read_csv(DATA_DIR + "\\countries_gdp_procapita.csv", encoding="utf8", sep=";")

#%%
def get_alpha3(alpha_2):
    try:
        return pycountry.countries.get(alpha_2=alpha_2).alpha_3
    except AttributeError:
        return "---"

countries = pd.read_csv(DATA_DIR + "\\country_coordinates.csv", index_col=False).iloc[:, 1:].dropna(axis=0, how="any")
gdp = pd.read_csv(DATA_DIR + "\\countries_gdp_procapita.csv", encoding="utf8", sep=";")
gdp.columns = ["name", "alpha_3", "GDP_PC"]

countries["alpha_3"] = countries.country.apply(get_alpha3)


gdp = countries.merge(gdp, on="alpha_3", how="inner")[["alpha_3", "name_y", "latitude", "longitude", "GDP_PC"]]
gdp.columns = ["alpha_3", "name", "latitude", "longitude", "GDP_PC"]

gdp = gdp.dropna(axis=0, how="any")

gdp.to_csv(DATA_DIR + "\\countries_gdp.csv", index=False)