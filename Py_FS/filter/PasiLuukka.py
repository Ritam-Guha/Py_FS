"""
Programmer: Rohit Kundu
Date of Development: 05/07/2021
"""

import numpy as np
from Py_FS.filter._utilities import normalize, Result
from sklearn import datasets

def PasiLuukka(in_data, target, measure = 'luca', p = 1):
    d = pd.DataFrame(in_data)
    t = pd.DataFrame(target)
    data = pd.concat([d,t],axis=1)
    
    # Feature selection method using similarity measure and fuzzy entroropy 
    # measures based on the article:
    # P. Luukka, (2011) Feature Selection Using Fuzzy Entropy Measures with
    # Similarity Classifier, Expert Systems with Applications, 38, pp. 4600-4607

    l = int(max(data.iloc[:,-1]))
    m = data.shape[0]
    t = data.shape[1]-1
    
    dataold = data.copy()
    
    idealvec_s = np.zeros((l,t)) 
    for k in range(l):
        idx = data.iloc[:,-1] == k+1
        idealvec_s[k,:] = data[idx].iloc[:,:-1].mean(axis = 0)
    
    # scaling data between [0,1]
    data_v = data.iloc[:,:-1]
    data_c = data.iloc[:,-1] # labels
    mins_v = data_v.min(axis = 0)
    Ones   = np.ones((data_v.shape))
    data_v = data_v + np.dot(Ones,np.diag(abs(mins_v)))
    
    tmp =[]
    for k in range(l):
        tmp.append(abs(mins_v))
    
    idealvec_s = idealvec_s+tmp
    maxs_v     = data_v.max(axis = 0)
    data_v     = np.dot(data_v,np.diag(maxs_v**(-1)))
    tmp2 =[];
    for k in range(l):
        tmp2.append(abs(maxs_v))
        
    idealvec_s = idealvec_s/tmp2
    
    data_vv = pd.DataFrame(data_v) # Convert the array of feature to a dataframe
    data    = pd.concat([data_vv, data_c], axis=1, ignore_index=False)

    # sample data
    datalearn_s = data.iloc[:,:-1]
    
    # similarities
    sim = np.zeros((t,m,l))
    
    for j in range(m):
        for i in range(t):
            for k in range(l):
                sim[i,j,k] = (1-abs(idealvec_s[k,i]**p - datalearn_s.iloc[j,i])**p)**(1/p)
            
    sim = sim.reshape(t,m*l)
    
    # possibility for two different entropy measures
    if measure =='luca':
        # moodifying zero and one values of the similarity values to work with 
        # De Luca's entropy measure
        delta = 1e-10
        sim[sim == 1] = delta
        sim[sim == 0] = 1-delta
        H = (-sim*np.log(sim)-(1-sim)*np.log(1-sim)).sum(axis = 1)
    elif measure == 'park':
        H = (np.sin(np.pi/2*sim)+np.sin(np.pi/2*(1-sim))-1).sum(axis = 1) 
        
    feature_values = np.array(in_data)
    result = Result()
    result.features = feature_values
    result.scores = H
    result.ranks = np.argsort(np.argsort(-H))
    result.ranked_features = feature_values[:, result.ranks]
    return result

if __name__ == '__main__':
    data = datasets.load_iris()
    sol = PasiLuukka(data.data, data.target)
