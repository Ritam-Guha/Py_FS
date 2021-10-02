"""

Programmer: Bitanu Chatterjee
Date of Development: 22/10/2020
This code has been developed according to the procedures mentioned in the following research article:
"Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi.
'GSA: a gravitational search algorithm.'' Information sciences 179.13 (2009): 2232-2248"

"""
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets

from Py_FS.datasets import get_dataset
from Py_FS.wrapper.nature_inspired.algorithm import Algorithm
from Py_FS.wrapper.nature_inspired._utilities_test import compute_accuracy, compute_fitness, initialize, sort_agents
from Py_FS.wrapper.nature_inspired._transfer_functions import get_trans_function


class GSA(Algorithm):

    # Gravitational Search Algorithm
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of particles                                           #
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

        self.algo_name = 'GSA'
        self.agent_name = 'Particle'
        self.trans_function = None
        self.algo_params = {}
        self.F = None
        self.R = None
        self.force = None
        self.acc = None
        self.velocity = None
        self.kBest = None

    def user_input(self):
        # accept the parameters as user inputs
        self.algo_params['trans_function'] = input('Shape of Transfer Function [s/v/u]: ') or 's'
        self.trans_function = get_trans_function(self.algo_params['trans_function'])

    def initialize(self):
        super(GSA, self).initialize()
        self.algo_params['eps'] = 0.00001
        self.algo_params['G_ini'] = 6
        self.algo_params['kBest'] = range(5)
        self.F = np.zeros((self.num_agents, self.num_agents, self.num_features))
        self.R = np.zeros((self.num_agents, self.num_agents))
        self.force = np.zeros((self.num_agents, self.num_features))
        self.acc = np.zeros((self.num_agents, self.num_features))
        self.velocity = np.zeros((self.num_agents, self.num_features))
        self.kBest = range(5)


    # main loop
    def next(self):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(self.cur_iter + 1))
        print('================================================================================\n')

        # updating value of G
        G = self.algo_params['G_ini'] - self.cur_iter * (self.algo_params['G_ini'] / self.max_iter)  # Eq. (13)

        # finding mass of each particle
        best_fitness = self.fitness[0]
        worst_fitness = self.fitness[-1]
        m = (self.fitness - worst_fitness) / (best_fitness - worst_fitness + self.algo_params['eps'])  # Eq. (15)
        sum_fitness = np.sum(m)
        mass = m / sum_fitness  # Eq. (16)

        # finding force acting between each pair of particles
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                for k in range(self.num_features):
                    self.R[i][j] += abs(self.population[i][k] - self.population[j][k])  # Eq. (8)
                self.F[i][j] = G * (mass[i] * mass[j]) / (self.R[i][j] + self.algo_params['eps']) * (self.population[j] - self.population[i])  # Eq. (7)

        # finding net force acting on each particle
        for i in range(self.num_agents):
            for j in self.kBest:
                if i != j:
                    self.force[i] += np.random.random() * self.F[i][j]  # Eq. (9)

        # finding acceleration of each particle
        for i in range(self.num_agents):
            self.acc[i] = self.force[i] / (mass[i] + self.algo_params['eps'])  # Eq. (10)

        # updating velocity of each particle
        self.velocity = np.random.random() * self.velocity + self.acc  # Eq. (11)

        # apply transformation function on the velocity
        for i in range(self.num_agents):
            for j in range(self.num_features):
                trans_value = self.trans_function(self.velocity[i][j])
                if np.random.random() < trans_value:
                    self.population[i][j] = 1
                else:
                    self.population[i][j] = 0
            if np.sum(self.population[i]) == 0:
                x = np.random.randint(0, self.num_features - 1)
                self.population[i][x] = 1

        self.cur_iter += 1



############# for testing purpose ################

if __name__ == '__main__':
    data = datasets.load_digits()
    algo = GSA(num_agents=20, max_iter=100, train_data=data.data, train_label=data.target, save_conv_graph=True)
    algo.run()
############# for testing purpose ################
