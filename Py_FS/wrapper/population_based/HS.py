# Programmer : Khalid Hassan

import random
import numpy as np
from sklearn import datasets

from Py_FS.wrapper.nature_inspired.algorithm import Algorithm
from Py_FS.wrapper.population_based._utilities import compute_fitness, sort_agents, compute_accuracy


class HS(Algorithm):
    # Harmony Search Algorithm
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of harmonies                                           #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #
    #   obj_function: the function to maximize while doing feature selection      #
    #   trans_function_shape: shape of the transfer function used                 #
    #   save_conv_graph: boolean value for saving convergence graph               #
    #                                                                             #
    ###############################################################################

    # <STEPS OF HARMOMY SEARCH ALGORITH>
    # Step 1. Initialize a Harmony Memory (HM).
    # Step 2. Improvise a new harmony from HM.
    # Step 3. If the new harmony is better than minimum harmony in HM, include the new harmony in HM, and exclude the minimum harmony from HM.
    # Step 4. If stopping criteria are not satisfied, go to Step 2.

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

        self.algo_name = 'HS'
        self.agent_name = 'Harmony'
        self.algo_params = {}

    def user_input(self):
        self.algo_params['HMCR'] = float(
            input('HMCR [0-1]: ') or 0.9)
        self.algo_params['PAR'] = float(
            input('PAR [0-1]: ') or 0.3)

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
        print('\n================================================================================')
        print('                          Iteration - {}'.format(self.cur_iter+1))
        print('================================================================================\n')

        # perform improvisation, replacement
        self.improvise()

        self.cur_iter += 1


if __name__ == '__main__':
    data = datasets.load_digits()
    algo = HS(num_agents=20, max_iter=5,
              train_data=data.data, train_label=data.target)
    solution = algo.run()
