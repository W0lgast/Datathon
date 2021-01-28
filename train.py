"""
Main script for training and testing Datathon insurance claim prediction model
"""

# ---------------------------------------------------------------------------

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, ElasticNet, LassoLars
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.tree import export_graphviz
import pandas as pd
from graphviz import Source

# ---------------------------------------------------------------------------

def visualize_tree(tree, feature_names):
    """
    Visualises the tree, download graphviz and add the location of dot.exe to system path,
    that worked for me.
    """
    viz_source = export_graphviz(tree, out_file=None, feature_names=feature_names, filled=True, rounded=True)
    s = Source(viz_source, filename="tree.gv", format="png")
    s.view()

# ---------------------------------------------------------------------------

# these are the features to use, only keep the ones you want, but it shouldn't be a problem
# if bad features are in this list, they just probs wont be used.
# "Customer ID", "Effective To Date", "Country", "State" are not included in this list.
features = ["State Code", "Response",
            "Coverage", "Education", "EmploymentStatus",
            "Gender", "Income", "Location Code", "Marital Status",
            "Monthly Premium Auto", "Months Since Last Claim",
            "Months Since Policy Inception", "Number of Open Complaints",
            "Number of Policies", "Policy Type", "Policy", "Claim Reason",
            "Sales Channel", "Vehicle Class",
            "Vehicle Size"]

targets = ["Total Claim Amount"]

# load dataframe
df = pd.read_csv("new_train.csv")
target_df = df[targets]
# use one hot encoding to deal with catagorical vars
df = pd.get_dummies(df[features]).join(df[targets])
features = [c for c in list(df.columns) if c not in targets]

# split train and test - 90% is train here
train_test_index = np.random.rand(len(df)) < 0.90
X_tr = df[train_test_index][features]
y_tr = df[train_test_index][targets]
X_te = df[train_test_index == 0][features]
y_te = df[train_test_index == 0][targets]

print("Number of training samples: ", len(X_tr))
print("Number of testing samples: ", len(X_te))

# define regressor and fit to training data
#regressor = DecisionTreeRegressor(max_depth=7)
#regressor = RandomForestRegressor(n_estimators=, criterion="mae", max_depth=7)
#regressor = LinearRegression()
#regressor = KNeighborsRegressor(n_neighbors=1)
#regressor = LassoLars()
regressor = SVR(kernel='linear', C=100, gamma='auto')
#regressor = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)

regressor.fit(X_tr, [v[0] for v in y_tr.values])

# predict for test data and return MAE
y_te_hat = regressor.predict(X_te)
err = mean_absolute_error(y_te, y_te_hat)
print("Mean Absolute Error: ", err)

# This is for vizualisation - it might not work on your pc - you need to get graphviz installed and added
# to path - it's annoying.
##visualize_tree(regressor, features)

print("Done!")

# ---------------------------------------------------------------------------
