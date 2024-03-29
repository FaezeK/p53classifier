############################################################################
# This script contains the code to perform a grid search to find the most
# suitable hyperparameter values for a given dataset
############################################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# set parameters
clf = RandomForestClassifier()
n_jobs = 40
#cv = 5
n_splits = 5
param_space = {
    'n_estimators': [25, 50, 100, 200],
    'max_samples': [0.2, 0.4, 0.6, 0.8, 0.99],
    'max_features': ['sqrt', 0.01, 0.03, 0.05],
    'max_depth': [10, 50, 100, 200],
    'min_samples_split': [2, 10, 25, 50],
    'min_samples_leaf': [1, 2, 10, 50]
}


def findHyperparam(X, y):
    grid = GridSearchCV(estimator=clf, param_grid=param_space, n_jobs=n_jobs, cv=StratifiedKFold(n_splits=n_splits, shuffle=True))

    grid_result = grid.fit(X, y)
    return grid_result
