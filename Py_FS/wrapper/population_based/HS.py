"""
    Programmer: Khalid Hassan
    Date of Development: 15/10/2020

    Paper 1: A New Heuristic Optimization Algorithm: Harmony Search
    Authors: Zong Woo Geem and Joong Hoon Kim, G. V. Loganathan

    Paper 2: An improved harmony search algorithm for solving optimization problems
    Authors: M. Mahdavi, M. Fesanghary, E. Damangir 
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

class HS(Algorithm):
    # Harmony Search (HS)
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

        self.algo_name = 'HS'
        self.agent_name = 'Harmony'

    def user_input(self):
        # first set the default values for the attributes
        self.default_vals["HMCR"] = 0.9
        self.default_vals["PAR"] = 0.3

        # accept the parameters as user inputs (if default_mode not set)
        if self.default_mode:
            self.set_default()
        else:    
            self.algo_params['HMCR'] = float(input(f"HMCR [0-1] (default={self.default_vals['HMCR']}): ") or self.default_vals['HMCR'])
            self.algo_params['PAR'] = float(input(f"PAR [0-1] (default={self.default_vals['PAR']}): ") or self.default_vals['PAR'])

    def improvise(self):
        HMCR_randValue = np.random.rand()
        num_features = self.population[0, :].shape[0]

        newHarmony = np.zeros([1, num_features])

        # Harmony Memory consideration rate
        if HMCR_randValue <= self.algo_params['HMCR']:
            for featureNum in range(num_features):
                selectedAgent = random.randint(0, self.num_agents - 1)
                newHarmony[0, featureNum] = self.population[selectedAgent, featureNum]

        else:
            for featureNum in range(num_features):
                newHarmony[0, featureNum] = random.randint(0, 1)

        for featureNum in range(num_features):
            # Pitch adjacement
            PAR_randValue = np.random.rand()
            if PAR_randValue > self.algo_params['PAR']:
                newHarmony[0, featureNum] = 1 - newHarmony[0, featureNum]

        fitnessHarmony = self.obj_function(
            newHarmony, self.training_data)

        if self.fitness[self.num_agents-1] < fitnessHarmony:
            self.population[self.num_agents-1, :] = newHarmony
            self.fitness[self.num_agents-1] = fitnessHarmony

        # sort harmony memory
        self.population, self.fitness = sort_agents(
            self.population, self.fitness)

        if self.fitness[0] > self.Leader_fitness:
            self.Leader_agent = self.population[0].copy()
            self.Leader_fitness = self.fitness[0].copy()

    def next(self):
        self.print('\n================================================================================')
        self.print('                          Iteration - {}'.format(self.cur_iter+1))
        self.print('================================================================================\n')

        # perform improvisation, replacement
        self.improvise()

        self.cur_iter += 1

############# for testing purpose ################
if __name__ == '__main__':
    data = datasets.load_digits()
    algo = HS(num_agents=20, max_iter=20, train_data=data.data, train_label=data.target, default_mode=True)
    solution = algo.run()
############# for testing purpose ################
