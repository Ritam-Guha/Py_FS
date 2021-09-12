# set the directory path
import os,sys
import os.path as path
abs_path_pkg =  path.abspath(path.join(__file__ ,"../../../../"))
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, abs_path_pkg)


from Py_FS.wrapper.population_based.CS import CS
from Py_FS.wrapper.population_based.GA import GA
from Py_FS.wrapper.population_based.MA import MA
from Py_FS.wrapper.population_based.WOA import WOA

algo_mapper = {
    "CS": CS,
    "GA": GA,
    "MA": MA,
    "WOA": WOA
}

def get_algorithm(algo_name):
    # function to get instance of an algorithm
    list_algo = list(algo_mapper.keys())
    if algo_name not in list_algo:
        print(f"[Error!] {algo_name} not present in Py_FS....")
    else:
        return algo_mapper[algo_name]