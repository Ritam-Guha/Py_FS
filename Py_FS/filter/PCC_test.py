"""

Programmer: Ritam Guha
Date of Development: 28/10/2020

"""
# set the directory path
import os,sys
import os.path as path
abs_path_pkg =  path.abspath(path.join(__file__ ,"../../../"))
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, abs_path_pkg)

# import other libraries
import numpy as np
from Py_FS.filter._utilities import normalize, Result
from Py_FS.filter.algorithm import Algorithm
from sklearn import datasets

class PCC(Algorithm):
    def __init__(self, 
                data, 
                target,
                default_mode=False,
                verbose=True):
        
        super().__init__(
            data=data,
            target=target,
            default_mode=default_mode,
            verbose=verbose
        )
    
    def user_input(self):
        # accept the parameters as user inputs (if default_mode not set)
        if self.default_mode:
            self.set_default()
        else:
            self.algo_params["weight_feat"] = float(input(f"Weight for feature-feature correlation: {self.default_vals['weight_feat']}") or self.default_vals['weight_feat'])
            self.algo_params["weight_class"] = float(input(f"Weight for feature-class correlation: {self.default_vals['weight_class']}") or self.default_vals['weight_class'])

    def initialize(self):
        super().initialize()
        self.correlation_matrix = np.zeros((self.num_features, self.num_features))
        self.feature_feature_relation = np.zeros(self.num_features)
        self.feature_class_relation = np.zeros(self.num_features)

    def compute_PCC(self, x, y):
        # function to compute the PCC value for two variables
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sqrt(np.sum(np.square(x - mean_x)) * np.sum(np.square(y - mean_y)))
        if denominator == 0:
            return 0
        PCC_val = numerator/denominator

        return PCC_val

    def execute(self):
        # generate the correlation matrix
        self.feature_mean = np.mean(self.data, axis=0)
        for ind_1 in range(self.num_features):
            for ind_2 in range(self.num_features):
                self.correlation_matrix[ind_1, ind_2] = self.correlation_matrix[ind_2, ind_1] = self.compute_PCC(self.data[:, ind_1], self.data[:, ind_2])

        for ind in range(self.num_features):
            self.feature_feature_relation[ind] = -np.sum(abs(self.correlation_matrix[ind,:])) # -ve because we want to remove the corralation
            self.feature_class_relation[ind] = abs(self.compute_PCC(self.data[:, ind], self.target))

        # produce scores and ranks from the information matrix
        self.feature_feature_relation = normalize(self.feature_feature_relation)
        self.feature_class_relation = normalize(self.feature_class_relation)
        self.scores = (self.algo_params["weight_class"] * self.feature_class_relation) + (self.algo_params["weight_feature"] * self.feature_feature_relation)

############# for testing purpose ################
if __name__ == '__main__':
    from scipy.stats.stats import pearsonr
    data = datasets.load_wine()
    algo = PCC(data.data, data.target)
    res = algo.run()
    print(res.correlation_matrix)
############# for testing purpose ################