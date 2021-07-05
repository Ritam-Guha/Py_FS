"""
Programmer: Rohit Kundu
Date of Development: 05/07/2021
"""

import numpy as np
from Py_FS.filter._utilities import normalize, Result
from sklearn import datasets

def MAD(data,target):
    #mean absolute deviation
    mean_abs_diff = np.sum(np.abs(data-np.mean(data,axis=0)),axis=0)/data.shape[0]
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = mean_abs_diff
    result.ranks = np.argsort(np.argsort(mean_abs_diff))
    result.ranked_features = feature_values[:, result.ranks]
    return result

if __name__ == '__main__':
    data = datasets.load_iris()
    sol = MAD(data.data, data.target)
