import numpy as np
from _utilities import normalize, Result
from sklearn import datasets

def SCC(data):
    # function that assigns scores to features according to Spearman's Correlation Coefficient (SCC)
    # the rankings should be done in increasing order of the SCC scores 

    # initialize the variables and result structure
    feature_values = np.array(data)
    feature_ranks = np.argsort(-feature_values, axis=0)
    num_features = feature_values.shape[1]
    SCC_mat = np.zeros((num_features, num_features))
    SCC_values = np.zeros(num_features)
    result = Result()
    result.features = feature_values

    # generate the correlation matrix
    mean_values = np.mean(feature_values, axis=0)
    for ind_1 in range(num_features):
        for ind_2 in range(num_features):
            numerator = np.sum((feature_values[:, ind_1] - mean_values[ind_1]) * (feature_values[:, ind_2] - mean_values[ind_2]))
            denominator = np.sqrt(np.sum(np.square(feature_values[:, ind_1] - mean_values[ind_1])) * np.sum(np.square(feature_values[:, ind_2] - mean_values[ind_2])))
            SCC_mat[ind_1, ind_2] = SCC_mat[ind_2, ind_1] = numerator/denominator

    for ind in range(num_features):
        SCC_values[ind] = -np.sum(abs(SCC_mat[ind,:]))

    # produce scores and ranks from the information matrix
    SCC_scores = normalize(SCC_values)
    SCC_ranks = np.argsort(-SCC_scores)

    # assign the results to the appropriate fields
    result.scores = SCC_scores
    result.ranks = SCC_ranks
    result.ranked_features = feature_values[:, SCC_ranks]

    return result

if __name__ == '__main__':
    SCC(datasets.load_iris().data)
