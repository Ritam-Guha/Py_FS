"""
Programmer: Rohit Kundu
Date of Development: 05/07/2021
"""

import numpy as np
from Py_FS.filter._utilities import normalize, Result
from sklearn.feature_selection import chi2
from sklearn import datasets

def ChiSquare(data, target):
    chi,_ = chi2(data, target)
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = chi
    result.ranks = np.argsort(np.argsort(chi))
    result.ranked_features = feature_values[:, result.ranks]
    return result

if __name__ == '__main__':
    data = datasets.load_iris()
    sol = ChiSquare(data.data, data.target)
