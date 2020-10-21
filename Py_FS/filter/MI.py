import numpy as np
from _utilities import normalize
from sklearn import datasets

def MI(data):
    # function that assigns scores to features according to Mutual Information (MI)
    # the rankings should be done in increasing order of the MI scores 
    feature_values = np.array(data)
    num_features = feature_values.shape[1]
    MI_mat = np.zeros((num_features, num_features))
    MI_values = np.zeros(num_features)
    
    for ind_1 in range(num_features):
        for ind_2 in range(num_features):
            MI_mat[ind_1, ind_2] = MI_mat[ind_2, ind_1] = compute_MI(feature_values[:, ind_1], feature_values[:, ind_2])

    for ind in range(num_features):
        MI_values[ind] = -np.sum(abs(MI_mat[ind,:]))

    MI_scores = normalize(MI_values)
    print(MI_scores)
    return MI_scores  

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
    MI(data.data)