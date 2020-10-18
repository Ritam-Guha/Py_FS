import numpy as np

def sigmoid(val):
    if val < 0:
        return 1 - 1/(1 + np.exp(val))
    else:
        return 1/(1 + np.exp(-val))

def v_func(val):
    return val/(np.sqrt(1 + val*val))

def u_func(val):
    alpha, beta = 2, 1.5
    return alpha * np.power(abs(val), beta)


def get_trans_function(shape):
    if (shape.lower() == 's'):
        return sigmoid

    elif (shape.lower() == 'v'):
        return v_func

    elif(shape.lower() == 'u'):
        return u_func

    else:
        print('\n[Error!] We don\'t currently support {}-shaped transfer functions...\n'.format(shape))
        exit(1)