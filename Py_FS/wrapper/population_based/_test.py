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
from Py_FS.datasets import get_dataset
from Py_FS.wrapper.population_based.get_algorithm import get_algorithm

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

def plot(algo_res, ax, color="r"):
    avg_fitness = []
    for cur in algo_res.history:
        avg_fitness.append(np.mean(cur['fitness']))

    ax.plot(np.arange(len(avg_fitness)), avg_fitness, c=color, label=algo_res.algo_name)

def main():
    dataset_name = "BreastCancer"
    data = get_dataset(dataset_name)
    list_algo = ["GA", "CS", "MA", "WOA"]
    algo_res = {}
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))   
    colors = iter(cm.rainbow(np.linspace(0, 1, len(list_algo))))

    for algo_name, c in zip(list_algo, colors):
        algo = get_algorithm(algo_name)
        algo_def = algo(default_mode=True, verbose=False, num_agents=40, max_iter=100, train_data=data.data, train_label=data.target, save_conv_graph=False)
        algo_res[algo_name] = algo_def.run()
        plot(algo_res[algo_name], ax, c)
    
    ax.set_title(dataset_name)
    ax.legend(loc="best")
    plt.show()
    fig.savefig("test_" + dataset_name + ".jpg")

if __name__ == "__main__":
    main()
