"""

Programmer: Bitanu Chatterjee
Date of Development: 14/10/2020
This code has been developed according to the procedures mentioned in the following research article:
"Fathollahi-Fard, Amir Mohammad, Mostafa Hajiaghaei-Keshteli, and Reza Tavakkoli-Moghaddam.
'Red deer algorithm (RDA): a new nature-inspired meta-heuristic.''" Soft Computing (2020): 1-29."

"""

# set the directory path
import os,sys
import os.path as path
abs_path_pkg =  path.abspath(path.join(__file__ ,"../../../../"))
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, abs_path_pkg)

# import other libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import random, math

from Py_FS.wrapper.population_based.algorithm import Algorithm
from Py_FS.wrapper.population_based._utilities import Data, compute_fitness, initialize, sort_agents, compute_accuracy, call_counter
from Py_FS.wrapper.population_based._transfer_functions import get_trans_function

class RDA(Algorithm):
    # Red Deer Algorithm (RDA)
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of agents                                              #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #
    #   test_data (optional): test samples of data                                #
    #   test_label (optional): class labels for the test samples                  #
    #   save_conv_graph (optional): True to save conv graph, else False           #
    #   seed (optional): seed for our random number generator                     #
    #   default_mode (optional): True to use default values for every             #
    #                            user input                                       #
    #   verbose (optional): True to print simulation, else False                  #
    ###############################################################################

    def __init__(self,
                num_agents, 
                max_iter, 
                train_data, 
                train_label, 
                test_data=None,
                test_label=None,
                save_conv_graph=False, 
                seed=0,
                default_mode=False,
                verbose=True):

        super().__init__(num_agents=num_agents,
                        max_iter=max_iter,
                        train_data=train_data,
                        train_label=train_label,
                        test_data=test_data,
                        test_label=test_label,
                        save_conv_graph=save_conv_graph,
                        seed=seed,
                        default_mode=default_mode,
                        verbose=verbose)

        self.algo_name = 'RDA'
        self.agent_name = 'RedDeer'
        self.algo_params["trans_function"] = None
        self.algo_params["num_males"] = None
        self.algo_params["num_hinds"] = None
        self.algo_params["num_coms"] = None
        self.algo_params["num_stags"] = None
        self.algo_params["num_harems"] = None
        self.algo_params["males"] = None
        self.algo_params["hinds"] = None
        self.algo_params["coms"] = None
        self.algo_params["stags"] = None
        self.algo_params["harem"] = None
        self.algo_params["population_pool"] = None

    def initialize(self):
        super(RDA, self).initialize()

    def user_input(self):
        # first set the default values for the attributes
        self.default_vals["trans_function"] = 's'
        self.default_vals["UB"] = 5
        self.default_vals["LB"] = -5
        self.default_vals["gamma"] = 0.5
        self.default_vals["alpha"] = 0.2
        self.default_vals["beta"] = 0.1

        # accept the parameters as user inputs (if default_mode not set)
        if self.default_mode:
            self.set_default()
        else:
            self.algo_params['trans_function'] = input(f"Shape of Transfer Function [s/v/u] (default={self.default_vals['trans_function']}): ") or self.default_vals["trans_function"]
            self.algo_params['UB'] = float(input(f"UB (default={self.default_vals['UB']}): ") or self.default_vals["UB"])  # Upper bound
            self.algo_params['LB'] = float(input(f"LB (default={self.default_vals['LB']}): ") or self.default_vals["LB"])  # Lower bound
            self.algo_params['gamma'] = float(input(f"gamma (default={self.default_vals['gamma']}): ") or self.default_vals["gamma"])  # Fraction of total number of males who are chosen as commanders
            self.algo_params['alpha'] = float(input(f"alpha (default={self.default_vals['alpha']}): ") or self.default_vals["alpha"])  # Fraction of total number of hinds in a harem who mate with the commander of their harem
            self.algo_params['beta'] = float(input(f"beta (default={self.default_vals['beta']}): ") or self.default_vals["beta"])  # Fraction of total number of hinds in a harem who mate with the commander of a different harem
        
        self.algo_params["trans_function"] = get_trans_function(self.algo_params['trans_function'])

    def roar(self):
        # roaring of male deer
        for i in range(self.algo_params["num_males"]):
            r1 = np.random.random()  # r1 is a random number in [0, 1]
            r2 = np.random.random()  # r2 is a random number in [0, 1]
            r3 = np.random.random()  # r3 is a random number in [0, 1]
            new_male = self.algo_params["males"][i].copy()
            if r3 >= 0.5:  # Eq. (3)
                new_male += r1 * (((self.algo_params['UB'] - self.algo_params['LB']) * r2) + self.algo_params['LB'])
            else:
                new_male -= r1 * (((self.algo_params['UB'] - self.algo_params['LB']) * r2) + self.algo_params['LB'])

            # apply transformation function on the new male
            for j in range(self.num_features):
                trans_value = self.algo_params["trans_function"](new_male[j])
                if np.random.random() < trans_value:
                    new_male[j] = 1
                else:
                    new_male[j] = 0

            if self.obj_function(new_male, self.training_data) < self.obj_function(self.algo_params["males"][i], self.training_data):
                self.algo_params["males"][i] = new_male

    def fight(self):
        # fight between male commanders and stags
        for i in range(self.algo_params["num_coms"]):
            chosen_com = self.algo_params["coms"][i].copy()
            chosen_stag = random.choice(self.algo_params["stags"])
            r1 = np.random.random()
            r2 = np.random.random()
            new_male_1 = (chosen_com + chosen_stag) / 2 + r1 * (
                        ((self.algo_params['UB'] - self.algo_params['LB']) * r2) + self.algo_params['LB'])  # Eq. (6)
            new_male_2 = (chosen_com + chosen_stag) / 2 - r1 * (
                        ((self.algo_params['UB'] - self.algo_params['LB']) * r2) + self.algo_params['LB'])  # Eq. (7)

            # apply transformation function on new_male_1
            for j in range(self.num_features):
                trans_value = self.algo_params["trans_function"](new_male_1[j])
                if (np.random.random() < trans_value):
                    new_male_1[j] = 1
                else:
                    new_male_1[j] = 0

            # apply transformation function on new_male_2
            for j in range(self.num_features):
                trans_value = self.algo_params["trans_function"](new_male_2[j])
                if (np.random.random() < trans_value):
                    new_male_2[j] = 1
                else:
                    new_male_2[j] = 0

            fitness = np.zeros(4)
            fitness[0] = self.obj_function(chosen_com, self.training_data)
            fitness[1] = self.obj_function(chosen_stag, self.training_data)
            fitness[2] = self.obj_function(new_male_1, self.training_data)
            fitness[3] = self.obj_function(new_male_2, self.training_data)

            bestfit = np.max(fitness)
            if fitness[0] < fitness[1] and fitness[1] == bestfit:
                self.algo_params["coms"][i] = chosen_stag.copy()
            elif fitness[0] < fitness[2] and fitness[2] == bestfit:
                self.algo_params["coms"][i] = new_male_1.copy()
            elif fitness[0] < fitness[3] and fitness[3] == bestfit:
                self.algo_params["coms"][i] = new_male_2.copy()

    def form_harems(self):
        # formation of harems
        com_fitness = self.obj_function(self.algo_params["coms"], self.training_data)
        self.algo_params["coms"], com_fitness = sort_agents(agents=self.algo_params["coms"], fitness=com_fitness)
        norm = np.linalg.norm(com_fitness)
        normal_fit = com_fitness / norm
        total = np.sum(normal_fit)
        power = normal_fit / total  # Eq. (9)
        self.algo_params["num_harems"] = [int(x * self.algo_params["num_hinds"]) for x in power]  # Eq.(10)
        max_harem_size = np.max(self.algo_params["num_harems"])
        self.algo_params["harem"] = np.empty(shape=(self.algo_params["num_coms"], max_harem_size, self.num_features))
        random.shuffle(self.algo_params["hinds"])
        itr = 0
        for i in range(self.algo_params["num_coms"]):
            harem_size = self.algo_params["num_harems"][i]
            for j in range(harem_size):
                self.algo_params["harem"][i][j] = self.algo_params["hinds"][itr]
                itr += 1

    def mate(self):
        # mating of commander with hinds in his harem
        num_harem_mate = [int(x * self.algo_params['alpha']) for x in self.algo_params["num_harems"]]  # Eq. (11)
        self.algo_params["population_pool"] = list(self.population)
        for i in range(self.algo_params["num_coms"]):
            np.random.shuffle(self.algo_params["harem"][i])
            for j in range(num_harem_mate[i]):
                r = np.random.random()  # r is a random number in [0, 1]
                offspring = (self.algo_params["coms"][i] + self.algo_params["harem"][i][j]) / 2 + (self.algo_params['UB'] - self.algo_params['LB']) * r  # Eq. (12)

                # apply transformation function on offspring
                for k in range(self.num_features):
                    trans_value = self.algo_params["trans_function"](offspring[k])
                    if (np.random.random() < trans_value):
                        offspring[k] = 1
                    else:
                        offspring[k] = 0
                self.algo_params["population_pool"].append(list(offspring))

                # if number of commanders is greater than 1, inter-harem mating takes place
                if self.algo_params["num_coms"] > 1:
                    # mating of commander with hinds in another harem
                    k = i
                    while k == i:
                        k = random.choice(range(self.algo_params["num_coms"]))

                    num_mate = int(self.algo_params["num_harems"][k] * self.algo_params['beta'])  # Eq. (13)

                    random.shuffle(self.algo_params["harem"][k])
                    for k in range(num_mate):
                        r = np.random.random()  # r is a random number in [0, 1]
                        offspring = (self.algo_params["coms"][i] + self.algo_params["harem"][k][j]) / 2 + (self.algo_params['UB'] - self.algo_params['LB']) * r
                        # apply transformation function on offspring
                        for j in range(self.num_features):
                            trans_value = self.algo_params["trans_function"](offspring[j])
                            if (np.random.random() < trans_value):
                                offspring[j] = 1
                            else:
                                offspring[j] = 0
                        self.algo_params["population_pool"].append(list(offspring))

        # mating of stag with nearest hind
        for stag in self.algo_params["stags"]:
            dist = np.zeros(self.algo_params["num_hinds"])
            for i in range(self.algo_params["num_hinds"]):
                dist[i] = np.sqrt(np.sum((stag - self.algo_params["hinds"][i]) * (stag - self.algo_params["hinds"][i])))
            min_dist = np.min(dist)
            for i in range(self.algo_params["num_hinds"]):
                distance = math.sqrt(np.sum((stag - self.algo_params["hinds"][i]) * (stag - self.algo_params["hinds"][i])))  # Eq. (14)
                if (distance == min_dist):
                    r = np.random.random()  # r is a random number in [0, 1]
                    offspring = (stag + self.algo_params["hinds"][i]) / 2 + (self.algo_params['UB'] - self.algo_params['LB']) * r

                    # apply transformation function on offspring
                    for j in range(self.num_features):
                        trans_value = self.algo_params["trans_function"](offspring[j])
                        if (np.random.random() < trans_value):
                            offspring[j] = 1
                        else:
                            offspring[j] = 0
                    self.algo_params["population_pool"].append(list(offspring))

                    break

    def select_next_generation(self):
        # selection of the next generation
        self.algo_params["population_pool"] = np.array(self.algo_params["population_pool"])
        fitness = self.obj_function(self.algo_params["population_pool"], self.training_data)
        self.algo_params["population_pool"], fitness = sort_agents(agents=self.algo_params["population_pool"], fitness=fitness)
        maximum = sum([f for f in fitness])
        selection_probs = [f / maximum for f in fitness]
        indices = np.random.choice(len(self.algo_params["population_pool"]), size=self.num_agents, replace=True, p=selection_probs)
        deer = self.algo_params["population_pool"][indices]

    # main loop
    def next(self):
        self.print('\n================================================================================')
        self.print('                          Iteration - {}'.format(self.cur_iter + 1))
        self.print('================================================================================\n')

        # selection of males and hinds
        self.algo_params["num_males"] = int(0.25 * self.num_agents)
        self.algo_params["num_hinds"] = self.num_agents - self.algo_params["num_males"]
        self.algo_params["males"] = self.population[:self.algo_params["num_males"], :]
        self.algo_params["hinds"] = self.population[self.algo_params["num_males"]:, :]

        self.roar()

        # selection of male commanders and stags
        self.algo_params["num_coms"] = int(self.algo_params["num_males"] * self.algo_params['gamma'])  # Eq. (4)
        self.algo_params["num_stags"] = self.algo_params["num_males"] - self.algo_params["num_coms"]  # Eq. (5)
        self.algo_params["coms"] = self.algo_params["males"][:self.algo_params["num_coms"], :]
        self.algo_params["stags"] = self.algo_params["males"][self.algo_params["num_coms"]:, :]

        self.fight()
        self.form_harems()
        self.mate()
        self.select_next_generation()

        self.cur_iter += 1


############# for testing purpose ################
if __name__ == '__main__':
    data = datasets.load_digits()
    algo = RDA(num_agents=20, max_iter=20, train_data=data.data, train_label=data.target, default_mode=True)
    solution = algo.run()
############# for testing purpose ################
