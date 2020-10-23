from ReliefF import ReliefF
import numpy as np
from _utilities import normalize, Result
from sklearn import datasets

def Relief(data, target):
    # function that assigns scores to features according to Spearman's Correlation Coefficient (SCC)
    # the rankings should be done in increasing order of the SCC scores 

    # initialize the variables and result structure
    feature_values = np.array(data)
    num_features = feature_values.shape[1]
    result = Result()
    result.features = feature_values

    # generate the ReliefF scores
    relief = ReliefF(n_neighbors=1, n_features_to_keep=num_features)
    relief.fit_transform(data, target)
    print(relief.top_features)

    # produce scores and ranks from the information matrix
    SCC_scores = normalize(SCC_values)
    SCC_ranks = np.argsort(-SCC_scores)

    # assign the results to the appropriate fields
    result.scores = SCC_scores
    result.ranks = SCC_ranks
    result.ranked_features = feature_values[:, SCC_ranks]

    return result

if __name__ == '__main__':
    data = datasets.load_wine()
    Relief(data.data, data.target)