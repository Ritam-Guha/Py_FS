"""
Programmer: Rohit Kundu
Date of Development: 05/07/2021
"""

import numpy as np
from Py_FS.filter._utilities import normalize, Result
from sklearn.feature_selection import mutual_info_classif
from sklearn import datasets

def IG(data,target):
    #Information Gain
    importances = mutual_info_classif(data,target)
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = importances
    result.ranks = np.argsort(np.argsort(importances))
    result.ranked_features = feature_values[:, result.ranks]
    return result

if __name__ == '__main__':
    data = datasets.load_iris()
    sol = IG(data.data, data.target)
