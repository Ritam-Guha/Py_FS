import numpy as np

def normalize(vector, lb=0, ub=1):
    # function to normalize a numpy vector in [lb, ub]
    norm_vector = np.zeros(vector.shape[0])
    maximum = max(vector)
    minimum = min(vector)
    norm_vector = lb + ((vector - minimum)/(maximum - minimum)) * (ub - lb)

    return norm_vector

class Result():
    # structure of the result
    def __init__(self):
        self.ranks = None
        self.scores = None
        self.features = None
        self.ranked_features = None 