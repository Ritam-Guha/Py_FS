"""

Programmer: Shameem Ahmed
Date of Development: 9/10/2020
This code has been developed according to the procedures mentioned in the following research article:
"Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer.
Advances in engineering software, 69, 46-61."

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

class GWO(Algorithm):
    # Grey Wolf Optimizer (GWO)
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

        self.algo_name = 'GWO'
        self.agent_name = 'Greywolf'
        self.trans_function = None
        self.alpha, self.beta, self.delta = None, None, None
        self.alpha_fit, self.beta_fit, self.delta_fit = None, None, None

    def initialize(self):
        super(GWO, self).initialize()
        self.alpha, self.beta, self.delta = np.zeros((1, self.num_features)), np.zeros(
            (1, self.num_features)), np.zeros((1, self.num_features))
        self.alpha_fit, self.beta_fit, self.delta_fit = float("-inf"), float("-inf"), float("-inf")

    def user_input(self):
        # first set the default values for the attributes
        self.default_vals["trans_function"] = 's'

        # accept the parameters as user inputs (if default_mode not set)
        if self.default_mode:
            self.set_default()
        else:    
            self.algo_params['trans_function'] = input(f"Shape of Transfer Function [s/v/u] (default={self.default_vals['trans_function']}):") or self.default_vals["trans_function"]
        self.trans_function = get_trans_function(self.algo_params['trans_function'])

    def update_wolves(self):
        # update the alpha, beta and delta grey wolves
        for i in range(self.num_agents):
            # update alpha, beta, delta
            if self.fitness[i] > self.alpha_fit:
                self.delta_fit = self.beta_fit
                self.delta = self.beta.copy()
                self.beta_fit = self.alpha_fit
                self.beta = self.alpha.copy()
                self.alpha_fit = self.fitness[i]
                self.alpha = self.population[i, :].copy()

            # update beta, delta
            elif self.fitness[i] > self.beta_fit:
                self.delta_fit = self.beta_fit
                self.delta = self.beta.copy()
                self.beta_fit = self.fitness[i]
                self.beta = self.population[i, :].copy()

            # update delta
            elif self.fitness[i] > self.delta_fit:
                self.delta_fit = self.fitness[i]
                self.delta = self.population[i, :].copy()

    def update_positions(self):
        for i in range(self.num_agents):
            for j in range(self.num_features):
                # calculate distance between alpha and current agent
                r1 = np.random.random()  # r1 is a random number in [0,1]
                r2 = np.random.random()  # r2 is a random number in [0,1]
                A1 = (2 * self.algo_params['a'] * r1) - self.algo_params['a']  # calculate A1
                C1 = 2 * r2  # calculate C1
                D_alpha = abs(C1 * self.alpha[j] - self.population[i, j])  # find distance from alpha
                X1 = self.alpha[j] - (A1 * D_alpha)  # Eq. (3.6)

                # calculate distance between beta and current agent
                r1 = np.random.random()  # r1 is a random number in [0,1]
                r2 = np.random.random()  # r2 is a random number in [0,1]
                A2 = (2 * self.algo_params['a'] * r1) - self.algo_params['a']  # calculate A2
                C2 = 2 * r2  # calculate C2
                D_beta = abs(C2 * self.beta[j] - self.population[i, j])  # find distance from beta
                X2 = self.beta[j] - (A2 * D_beta)  # Eq. (3.6)

                # calculate distance between delta and current agent
                r1 = np.random.random()  # r1 is a random number in [0,1]
                r2 = np.random.random()  # r2 is a random number in [0,1]
                A3 = (2 * self.algo_params['a'] * r1) - self.algo_params['a']  # calculate A3
                C3 = 2 * r2  # calculate C3
                D_delta = abs(C3 * self.delta[j] - self.population[i, j])  # find distance from delta
                X3 = self.delta[j] - A3 * D_delta  # Eq. (3.6)

                # update the position of current agent
                self.population[i, j] = (X1 + X2 + X3) / 3  # Eq. (3.7)

            # Apply transformation function on the updated greywolf
            for j in range(self.num_features):
                trans_value = self.trans_function(self.population[i, j])
                if (np.random.random() < trans_value):
                    self.population[i, j] = 1
                else:
                    self.population[i, j] = 0

    # main loop
    def next(self):
        self.print('\n================================================================================')
        self.print('                          Iteration - {}'.format(self.cur_iter + 1))
        self.print('================================================================================\n')

        self.update_wolves()
        self.algo_params['a'] = 2 - self.cur_iter * ((2) / self.max_iter)
        self.update_positions()

        self.cur_iter += 1


############# for testing purpose ################
if __name__ == '__main__':
    data = datasets.load_digits()
    algo = GWO(num_agents=20, max_iter=20, train_data=data.data, train_label=data.target, default_mode=True)
    solution = algo.run()
############# for testing purpose ################
