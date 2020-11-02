"""

Programmer: Ritam Guha
Date of Development: 28/10/2020

"""

import numpy as np
from Py_FS.filter._utilities import normalize, Result
# _utilities import normalize, Result
from sklearn import datasets

def SCC(data, target):
    # function that assigns scores to features according to Spearman's Correlation Coefficient (SCC)
    # the rankings should be done in increasing order of the SCC scores 

    # initialize the variables and result structure
    feature_values = np.array(data)
    num_features = feature_values.shape[1]
    SCC_mat = np.zeros((num_features, num_features))
    SCC_values_feat = np.zeros(num_features)
    SCC_values_class = np.zeros(num_features)
    result = Result()
    result.features = feature_values
    weight_feat = 0.3   # weightage provided to feature-feature correlation
    weight_class = 0.7  # weightage provided to feature-class correlation

    # generate the correlation matrix
    for ind_1 in range(num_features):
        for ind_2 in range(num_features):
            SCC_mat[ind_1, ind_2] = SCC_mat[ind_2, ind_1] = compute_SCC(feature_values[:, ind_1], feature_values[:, ind_2])

    for ind in range(num_features):
        SCC_values_feat[ind] = -np.sum(abs(SCC_mat[ind,:]))
        SCC_values_class[ind] = compute_SCC(feature_values[:, ind], target)

    # produce scores and ranks from the information matrix
    SCC_values_feat = normalize(SCC_values_feat)
    SCC_values_class = normalize(SCC_values_class)
    SCC_scores = (weight_class * SCC_values_class) + (weight_feat * SCC_values_feat)
    SCC_ranks = np.argsort(np.argsort(-SCC_scores))

    # assign the results to the appropriate fields
    result.scores = SCC_scores
    result.ranks = SCC_ranks
    result.ranked_features = feature_values[:, np.argsort(-SCC_scores)]

    return result

def compute_SCC(x, y):
    # function to compute the SCC value for two variables
    x_order = np.argsort(np.argsort(x))
    y_order = np.argsort(np.argsort(y))
    mean_x = np.mean(x_order)
    mean_y = np.mean(y_order)
    numerator = np.sum((x_order - mean_x) * (y_order - mean_y))
    denominator = np.sqrt(np.sum(np.square(x_order - mean_x)) * np.sum(np.square(y_order - mean_y)))
    SCC_val = numerator/denominator

    return SCC_val

if __name__ == '__main__':
    data = datasets.load_iris()
    SCC(data.data, data.target)
