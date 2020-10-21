import numpy as np

def normalize(vector, lb=0, ub=1):
    # function to normalize a numpy vector in [lb, ub]
    norm_vector = np.zeros(vector.shape[0])
    maximum = max(vector)
    minimum = min(vector)
    norm_vector = lb + ((vector - minimum)/(maximum - minimum)) * (ub - lb)

    return norm_vector


if __name__=='__main__':
    a = np.array([1, 5, 6, 2, 7, 8])
    vect = normalize(a)
    print(vect)