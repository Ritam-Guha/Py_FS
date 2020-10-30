"""

Programmer: 
Date of Development: 

"""

import numpy as np
from sklearn import datasets

# from Py_FS.filter._utilities import normalize, Result
from _utilities import normalize, Result

def Name_of_the_filter(data, target):
    # function that assigns scores to features according to 'write the name of your filter method'
    # the rankings should be done in increasing order of the 'write the name of your filter method' scores 
    
    # initialize the variables and result structure
    feature_values = np.array(data)
    num_features = feature_values.shape[1]
    Name_of_the_filter_mat = np.zeros((num_features, num_features))
    Name_of_the_filter_values_feat = np.zeros(num_features)
    Name_of_the_filter_values_class = np.zeros(num_features)
    Name_of_the_filter_scores = np.zeros(num_features)
    result = Result()
    result.features = feature_values
    weight_feat = 0.3   # weightage provided to feature-feature correlation
    weight_class = 0.7  # weightage provided to feature-class correlation

    ################ write your main feature ranking code here ################



    ###########################################################################
    

    # produce scores and ranks from the information matrix
    Name_of_the_filter_values_feat = normalize(Name_of_the_filter_values_feat)
    Name_of_the_filter_values_class = normalize(Name_of_the_filter_values_class)
    Name_of_the_filter_scores = (weight_class * Name_of_the_filter_values_class) + (weight_feat * Name_of_the_filter_values_feat)
    Name_of_the_filter_ranks = np.argsort(np.argsort(-Name_of_the_filter_scores))

    # assign the results to the appropriate fields
    result.scores = Name_of_the_filter_scores
    result.ranks = Name_of_the_filter_ranks
    result.ranked_features = feature_values[:, np.argsort(-Name_of_the_filter_scores)]

    return result


if __name__ == '__main__':
    data = datasets.load_iris()
    Name_of_the_filter(data.data, data.target)