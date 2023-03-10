# set the directory path
import os,sys
import os.path as path
abs_path_pkg =  path.abspath(path.join(__file__ ,"../../../"))
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, abs_path_pkg)

from sklearn.model_selection import train_test_split

from Py_FS.evaluation._utilities import Metric
from Py_FS.datasets import get_dataset

import numpy as np

def evaluate(train_X, test_X, train_Y, test_Y, agent=None, classifier='knn', save_conf_mat=False, averaging="weighted"):
    # driver function
    metric = Metric(train_X, test_X, train_Y, test_Y, agent, classifier, save_conf_mat, averaging)
    return metric


if __name__ == "__main__":
    data = get_dataset("Arrhythmia")
    train_X, test_X, train_Y, test_Y = train_test_split(data.data, data.target, stratify=data.target, test_size=0.2, random_state=2)
    print(test_Y)
    num_features = data.data.shape[1]
    agent = np.ones(num_features)
    result = evaluate(train_X, test_X, train_Y, test_Y, agent, save_conf_mat=True)
    print(result.precision)
    print(result.recall)
    print(result.f1_score)