"""

Programmer: Ritam Guha
Date of Development: 28/10/2020

"""

import numpy as np
from Py_FS.filter._utilities import normalize, Result
# from sklearn import datasets

def PCC(data, target):
    # function that assigns scores to features according to Pearson's Correlation Coefficient (PCC)
    # the rankings should be done in increasing order of the PCC scores 
    
    # initialize the variables and result structure
    feature_values = np.array(data)
    num_features = feature_values.shape[1]
    PCC_mat = np.zeros((num_features, num_features))
    PCC_values_feat = np.zeros(num_features)
    PCC_values_class = np.zeros(num_features)
    PCC_scores = np.zeros(num_features)
    result = Result()
    result.features = feature_values
    weight_feat = 0.3   # weightage provided to feature-feature correlation
    weight_class = 0.7  # weightage provided to feature-class correlation

    # generate the correlation matrix
    mean_values = np.mean(feature_values, axis=0)
    for ind_1 in range(num_features):
        for ind_2 in range(num_features):
            PCC_mat[ind_1, ind_2] = PCC_mat[ind_2, ind_1] = compute_PCC(feature_values[:, ind_1], feature_values[:, ind_2])

    for ind in range(num_features):
        PCC_values_feat[ind] = -np.sum(abs(PCC_mat[ind,:]))
        PCC_values_class[ind] = abs(compute_PCC(feature_values[:, ind], target))

    # produce scores and ranks from the information matrix
    PCC_values_feat = normalize(PCC_values_feat)
    PCC_values_class = normalize(PCC_values_class)
    PCC_scores = (weight_class * PCC_values_class) + (weight_feat * PCC_values_feat)
    PCC_ranks = np.argsort(np.argsort(-PCC_scores))

    # assign the results to the appropriate fields
    result.scores = PCC_scores
    result.ranks = PCC_ranks
    result.ranked_features = feature_values[:, np.argsort(-PCC_scores)]

    return result


def compute_PCC(x, y):
    # function to compute the PCC value for two variables
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum(np.square(x - mean_x)) * np.sum(np.square(y - mean_y)))
    PCC_val = numerator/denominator

    return PCC_val


if __name__ == '__main__':
    data = datasets.load_iris()
    PCC(data.data, data.target)