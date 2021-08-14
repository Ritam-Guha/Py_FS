import os, sys
from Py_FS.datasets.get_dataset import get_dataset

#  set the directory path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

__all__ = [
    "get_dataset"
]