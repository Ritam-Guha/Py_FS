import Py_FS.wrapper.nature_inspired
import os, sys

#  set the directory path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

__all__ = [
    "nature_inspired"
]