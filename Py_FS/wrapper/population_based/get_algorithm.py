# set the directory path
import os,sys
import os.path as path
abs_path_pkg =  path.abspath(path.join(__file__ ,"../../../../"))
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, abs_path_pkg)

from Py_FS.wrapper.population_based.BBA import BBA
from Py_FS.wrapper.population_based.CS import CS
from Py_FS.wrapper.population_based.EO import EO
from Py_FS.wrapper.population_based.GA import GA
from Py_FS.wrapper.population_based.GSA import GSA
from Py_FS.wrapper.population_based.GWO import GWO
from Py_FS.wrapper.population_based.HS import HS
from Py_FS.wrapper.population_based.MA import MA
from Py_FS.wrapper.population_based.PSO import PSO
from Py_FS.wrapper.population_based.RDA import RDA
from Py_FS.wrapper.population_based.SCA import SCA
from Py_FS.wrapper.population_based.WOA import WOA

algo_mapper = {
    "BBA":BBA, 
    "CS":CS, 
    "EO":EO, 
    "GA":GA, 
    "GSA":GSA, 
    "GWO":GWO, 
    "HS":HS, 
    "MA":MA, 
    "PSO":PSO, 
    "RDA":RDA, 
    "SCA":SCA, 
    "WOA":WOA
}


def get_algorithm(algo_name):
    # function to get instance of an algorithm
    list_algo = list(algo_mapper.keys())
    if algo_name not in list_algo:
        print(f"[Error!] {algo_name} not present in Py_FS....")
    else:
        return algo_mapper[algo_name]