"""
Programmer: Rohit Kundu
Date of Development: 05/07/2021
"""

import numpy as np
from Py_FS.filter._utilities import normalize, Result
from sklearn import datasets

def DispersionRatio(data,target):
    data[np.where(data==0)[0]] = 1
    var = np.var(data,axis=0)
    mean = np.mean(data,axis=0)
    disp_ratio = var/mean
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = disp_ratio
    result.ranks = np.argsort(np.argsort(disp_ratio))
    result.ranked_features = feature_values[:, result.ranks]
    return result

if __name__ == '__main__':
    data = datasets.load_iris()
    sol = DispersionRatio(data.data, data.target)
