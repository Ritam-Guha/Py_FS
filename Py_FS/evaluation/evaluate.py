from sklearn.model_selection import train_test_split
from sklearn import datasets

from Py_FS.evaluation._utilities import Metric
# from _utilities import Metric

import numpy as np

def evaluate(train_X, test_X, train_Y, test_Y, agent=None, classifier='knn', save_conf_mat=False):
    # driver function
    metric = Metric(train_X, test_X, train_Y, test_Y, agent, classifier, save_conf_mat)
    return metric


if __name__ == "__main__":
    iris = datasets.load_iris()
    train_X, test_X, train_Y, test_Y = train_test_split(iris.data, iris.target, stratify=iris.target, test_size=0.2)
    num_features = iris.data.shape[1]
    agent = np.ones(num_features)
    agent[0] = agent[2] = 0
    result = evaluate(train_X, test_X, train_Y, test_Y, agent, save_conf_mat=True)
    print(result.confusion_matrix)