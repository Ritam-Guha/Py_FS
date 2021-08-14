from Py_FS.filter.MI import MI
from Py_FS.filter.PCC import PCC
from Py_FS.filter.Relief import Relief
from Py_FS.filter.SCC import SCC
import os, sys

#  set the directory path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

__all__ =[
    "MI",
    "PCC",
    "Relief",
    "SCC"
]