import numpy as np

def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1/(1 + np.exp(gamma))
    else:
        return 1/(1 + np.exp(-gamma))


def get_trans_function(shape):
    if (shape=='s' or shape=='S'):
        return sigmoid