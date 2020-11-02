"""

Programmer: Ritam Guha
Date of Development: 28/10/2020

"""

import numpy as np
from Py_FS.filter._utilities import normalize, Result
# _utilities import normalize, Result
from sklearn import datasets

def MI(data, target):
    # function that assigns scores to features according to Mutual Information (MI)
    # the rankings should be done in increasing order of the MI scores 
    
    # initialize the variables and result structure
    feature_values = np.array(data)
    num_features = feature_values.shape[1]
    MI_mat = np.zeros((num_features, num_features))
    MI_values_feat = np.zeros(num_features)
    MI_values_class = np.zeros(num_features)
    result = Result()
    result.features = feature_values
    weight_feat = 0.3   # weightage provided to feature-feature correlation
    weight_class = 0.7  # weightage provided to feature-class correlation
    
    # generate the information matrix
    for ind_1 in range(num_features):
        for ind_2 in range(num_features):
            MI_mat[ind_1, ind_2] = MI_mat[ind_2, ind_1] = compute_MI(feature_values[:, ind_1], feature_values[:, ind_2])

    for ind in range(num_features):
        MI_values_feat[ind] = -np.sum(abs(MI_mat[ind,:]))
        MI_values_class[ind] = compute_MI(feature_values[:, ind], target)

    # produce scores and ranks from the information matrix
    MI_values_feat = normalize(MI_values_feat)
    MI_values_class = normalize(MI_values_class)
    MI_scores = (weight_class * MI_values_class) + (weight_feat * MI_values_feat)
    MI_ranks = np.argsort(np.argsort(-MI_scores))

    # assign the results to the appropriate fields
    result.scores = MI_scores
    result.ranks = MI_ranks
    result.ranked_features = feature_values[:, np.argsort(-MI_scores)]

    return result  

def compute_MI(x, y):
    # function to compute mutual information between two variables 
    sum_mi = 0.0
    x_value_list = np.unique(x)
    y_value_list = np.unique(y)
    Px = np.array([ len(x[x==xval])/float(len(x)) for xval in x_value_list ]) #P(x)
    Py = np.array([ len(y[y==yval])/float(len(y)) for yval in y_value_list ]) #P(y)
    for i in range(len(x_value_list)):
        if Px[i] ==0.:
            continue
        sy = y[x == x_value_list[i]]
        if len(sy)== 0:
            continue
        pxy = np.array([len(sy[sy==yval])/float(len(y))  for yval in y_value_list]) #p(x,y)
        t = pxy[Py>0.]/Py[Py>0.] /Px[i] # log(P(x,y)/( P(x)*P(y))
        sum_mi += sum(pxy[t>0]*np.log2( t[t>0]) ) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )

    return sum_mi

if __name__ == '__main__':
    data = datasets.load_iris()
    MI(data.data, data.target)