# set the directory path
import os,sys
import os.path as path
abs_path_pkg =  path.abspath(path.join(__file__ ,"../../../"))
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, abs_path_pkg)

import numpy as np
import copy
from abc import abstractmethod
from Py_FS.filter._utilities import normalize, Result

class Algorithm():
    def __init__(self,
                data,
                target,
                default_mode=False,
                verbose=False):
        
        self.data = np.array(data)
        self.target = target
        self.num_features = self.data.shape[1]
        self.feature_scores = None
        self.feature_ranks = None
        self.ranked_features = None
        # self.feature_matrix = np.zeros((self.num_features, self.num_features))
        # self.feature_feaure_relation = np.zeros(self.num_features)
        # self.feature_class_relation = np.zeros(self.num_features)
        self.verbose = verbose
        self.print = self.verboseprint()
        self.algo_params = None
        self.default_mode = default_mode
        self.algo_params = {}
        self.default_vals = {}
    
    @abstractmethod
    def user_input(self):
        pass
    
    def set_default(self):
        # function to set the algo params to default values
        list_keys = list(self.default_vals.keys())
        for key in list_keys:
            self.algo_params[key] = self.default_vals[key]

    def initialize(self):
        self.default_vals["weight_feat"] = 0.3
        self.default_vals["weight_class"] = 0.7
        self.features = copy.deepcopy(self.data)

    def run(self):
        self.initialize()  # initialize the algorithm
        self.user_input()   # take the user inputs
        self.execute()
        self.feature_ranks = np.argsort(np.argsort(-self.feature_scores))
        self.ranked_features = self.data[:, self.feature_ranks]

        return self

    def verboseprint(self):
        if self.verbose:
            def mod_print(*args, end="\n"):
                # Print each argument separately so caller doesn't need to
                # stuff everything to be printed into a single string
                for arg in args:
                    print(arg, end=end),
                print
        else:
            def mod_print(*args, end="\n"):
                pass

        return mod_print

        