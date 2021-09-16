"""

Programmer: Shameem Ahmed
Date of Development: 9/10/2020
This code has been developed according to the procedures mentioned in the following research article:
"Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer.
Advances in engineering software, 69, 46-61."

"""
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets

import numpy as np
from Py_FS.datasets import get_dataset
from wrapper.nature_inspired.algorithm import Algorithm
from wrapper.nature_inspired._utilities_test import compute_accuracy, compute_fitness, initialize, sort_agents
from wrapper.nature_inspired._transfer_functions import get_trans_function


class GWO(Algorithm):

    # Grey Wolf Optimizer
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of greywolves                                          #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #
    #   obj_function: the function to maximize while doing feature selection      #
    #   trans_function_shape: shape of the transfer function used                 #
    #   save_conv_graph: boolean value for saving convergence graph               #
    #                                                                             #
    ###############################################################################

    def __init__(self,
                 num_agents,
                 max_iter,
                 train_data,
                 train_label,
                 save_conv_graph=False,
                 seed=0):

        super().__init__(num_agents=num_agents,
                         max_iter=max_iter,
                         train_data=train_data,
                         train_label=train_label,
                         save_conv_graph=save_conv_graph,
                         seed=seed)

        self.algo_name = 'GWO'
        self.agent_name = 'Greywolf'
        self.trans_function = None
        self.algo_params = {}
        self.alpha, self.beta, self.delta = None, None, None
        self.alpha_fit, self.beta_fit, self.delta_fit = None, None, None

    def initialize(self):
        super(GWO, self).initialize()
        self.alpha, self.beta, self.delta = np.zeros((1, self.num_features)), np.zeros(
            (1, self.num_features)), np.zeros((1, self.num_features))
        self.alpha_fit, self.beta_fit, self.delta_fit = float("-inf"), float("-inf"), float("-inf")

    def user_input(self):
        # accept the parameters as user inputs
        self.algo_params['trans_function'] = input('Shape of Transfer Function [s/v/u]: ') or 's'
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
        print('\n================================================================================')
        print('                          Iteration - {}'.format(self.cur_iter + 1))
        print('================================================================================\n')

        self.update_wolves()
        self.algo_params['a'] = 2 - self.cur_iter * ((2) / self.max_iter)
        self.update_positions()

        self.cur_iter += 1


############# for testing purpose ################
if __name__ == '__main__':
    data = datasets.load_digits()
    algo = GWO(num_agents=20, max_iter=100, train_data=data.data, train_label=data.target, save_conv_graph=True)
    algo.run()
