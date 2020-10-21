#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[18]:


def PCC(num_features,data):
    dict={}
    #calculating pearson correlation coefficient
    from scipy.stats.stats import pearsonr
    for i in range(num_features):
        for j in range(num_features):
            t=(i,j)
            val=pearsonr(np.array(data[:,i]).tolist(),np.array(data[:,j]).tolist()  )
            dict[t]=val
    #calculating pcc values for each feature
    pcc=[0 for i in range(num_features)]
    for i in range(0,num_features):
        sum=0
        for j in range(0,i):
            sum=sum+dict[(j,i)][0]
        for j in range(i+1,num_features):
            sum=sum+dict[(i,j)][0]
        sum=sum/(num_features-1)
        pcc[i]=sum
        
    return pcc


# In[ ]:


from sklearn import datasets
iris = datasets.load_iris()
print(PCC(iris.data.shape[1],iris.data))


# In[ ]:




