import numpy as np

def sigmoid(val):
    if val < 0:
        return 1 - 1/(1 + np.exp(gamma))
    else:
        return 1/(1 + np.exp(-gamma))

def v_func(val):
    return val/(np.sqrt(1 + val*val))

def u_func(val):
    alpha, beta = 2, 1.5
    return alpha * np.power(abs(val), beta)


def get_trans_function(shape):
    if (shape=='s' or shape=='S'):
        return sigmoid