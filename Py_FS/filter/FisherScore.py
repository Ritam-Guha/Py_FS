"""
Programmer: Rohit Kundu
Date of Development: 05/07/2021
"""

import numpy as np
from Py_FS.filter._utilities import normalize, Result
from sklearn import datasets

def FisherScore(data,target):
    mean = np.mean(data)
    sigma = np.var(data)
    unique = np.unique(target)
    mu = np.zeros(shape=(data.shape[1],unique.shape[0]))
    n = np.zeros(shape=(unique.shape[0],))
    var = np.zeros(shape=(data.shape[1],unique.shape[0]))
    for j in range(data.shape[1]):
        for i,u in enumerate(unique):
            d = data[np.where(target==u)[0]]
            n[i] = d.shape[0]
            mu[j,i] = np.mean(d[j])
            var[j,i] = np.var(d[j])
    fisher = np.zeros(data.shape[1])
    for j in range(data.shape[1]):
        sum1=0
        sum2=0
        for i,u in enumerate(unique):
            sum1+=n[i]*((mu[j,i]-mean)**2)
            sum2+=n[i]*var[j,i]
        fisher[j] = sum1/sum2
    feature_values = np.array(data)
    result = Result()
    result.features = feature_values
    result.scores = fisher
    result.ranks = np.argsort(np.argsort(fisher))
    result.ranked_features = feature_values[:, result.ranks]
    return result

if __name__ == '__main__':
    data = datasets.load_iris()
    sol = FisherScore(data.data, data.target)
