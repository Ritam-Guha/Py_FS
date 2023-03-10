"""

Programmer: Ritam Guha
Date of Development: 6/10/2020

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

from Py_FS.wrapper.population_based.algorithm import Algorithm
from Py_FS.wrapper.population_based._utilities import Data, compute_fitness, initialize, sort_agents, compute_accuracy, call_counter
from Py_FS.wrapper.population_based._transfer_functions import get_trans_function

class NAME_OF_THE_ALGORITHM(Algorithm):
    # Genetic Algorithm (GA)
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

        self.algo_name = ''
        self.agent_name = ''

    def user_input(self):
        # first set the default values for the attributes
        self.default_vals["USER_INPUT"] = 0.3

        # accept the parameters as user inputs (if default_mode not set)
        if self.default_mode:
            self.set_default()
        else:
            self.algo_params['prob_cross'] = float(input(f'USER INPUT [RANGE OF THE INPUT] (default={self.default_vals["USER_INPUT"]}): ') or self.default_vals["USER_INPUT"])

    def ALGORITHM_SPECIFIC_FUNCTIONS(self):
        ##### define your algorithm specific functions #####
            ### Like crossover, mutation for GA ###
        ##### define your algorithm specific functions #####
        pass

    def next(self):
        self.print('\n================================================================================')
        self.print('                          Iteration - {}'.format(self.cur_iter+1))
        self.print('================================================================================\n')

        # CALL YOUR ALGO SPECIFIC OPERATIONS FOR EACH ITERATION
        self.ALGORITHM_SPECIFIC_FUNCTIONS()

        self.cur_iter += 1

############# for testing purpose ################
if __name__ == '__main__':
    data = datasets.load_digits()
    algo = NAME_OF_THE_ALGORITHM(num_agents=20, max_iter=20, train_data=data.data, train_label=data.target, default_mode=True)
    solution = algo.run()
############# for testing purpose ################
