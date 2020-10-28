from sklearn import datasets
from .. import filter

if __name__ == '__main__':
    dataset = datasets.load_iris()
    data = dataset.data
    target = dataset.target