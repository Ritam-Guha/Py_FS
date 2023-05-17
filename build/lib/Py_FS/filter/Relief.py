"""

Programmer: Ritam Guha
Date of Development: 28/10/2020

"""
# set the directory path
import os, sys
import os.path as path

abs_path_pkg = path.abspath(path.join(__file__, "../../../"))
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, abs_path_pkg)

# import other libraries
import numpy as np
from Py_FS.filter._utilities import normalize, Result
from Py_FS.filter.algorithm import Algorithm
from sklearn import datasets
from ReliefF import ReliefF


class Relief(Algorithm):
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
        self.feature_scores = None

    def user_input(self):
        # accept the parameters as user inputs (if default_mode not set)
        self.default_vals["n_neighbors"] = 5

        if self.default_mode:
            self.set_default()
        else:
            self.algo_params["n_neighbors"] = float(
                input(f"Number of neighbors considered for relief scores (default = {self.default_vals['n_neighbors']}): ") or
                self.default_vals['n_neighbors'])

    def initialize(self):
        super().initialize()

    def execute(self):
        # generate the ReliefF scores
        relief = ReliefF(n_neighbors=int(self.algo_params["n_neighbors"]), n_features_to_keep=self.num_features)
        relief.fit_transform(self.data, self.target)

        # produce scores and ranks from the information matrix
        self.feature_scores = normalize(relief.feature_scores)


############# for testing purpose ################
if __name__ == '__main__':
    data = datasets.load_wine()
    algo = Relief(data.data, data.target)
    res = algo.run()
    print(res.feature_scores)


############# for testing purpose ################


def Relief(data, target):
    # function that assigns scores to features according to Relief algorithm
    # the rankings should be done in increasing order of the Relief scores 

    # initialize the variables and result structure
    feature_values = np.array(data)
    num_features = feature_values.shape[1]
    result = Result()
    result.features = feature_values

    # generate the ReliefF scores
    relief = ReliefF(n_neighbors=5, n_features_to_keep=num_features)
    relief.fit_transform(data, target)
    result.scores = normalize(relief.feature_scores)
    result.ranks = np.argsort(np.argsort(-relief.feature_scores))

    # produce scores and ranks from the information matrix
    Relief_scores = normalize(relief.feature_scores)
    Relief_ranks = np.argsort(np.argsort(-relief.feature_scores))

    # assign the results to the appropriate fields
    result.scores = Relief_scores
    result.ranks = Relief_ranks
    result.ranked_features = feature_values[:, Relief_ranks]

    return result


if __name__ == '__main__':
    data = datasets.load_iris()
    Relief(data.data, data.target)
