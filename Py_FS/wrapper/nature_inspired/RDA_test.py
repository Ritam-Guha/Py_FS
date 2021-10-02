"""

Programmer: Bitanu Chatterjee
Date of Development: 14/10/2020
This code has been developed according to the procedures mentioned in the following research article:
"Fathollahi-Fard, Amir Mohammad, Mostafa Hajiaghaei-Keshteli, and Reza Tavakkoli-Moghaddam.
'Red deer algorithm (RDA): a new nature-inspired meta-heuristic.''" Soft Computing (2020): 1-29."

"""

import math
import random

import numpy as np
from sklearn import datasets
from wrapper.nature_inspired._transfer_functions import get_trans_function
from wrapper.nature_inspired._utilities_test import sort_agents
from wrapper.nature_inspired.algorithm import Algorithm


class RDA(Algorithm):

    # Red Deer Algorithm
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of red deer                                           #
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

        self.algo_name = 'RDA'
        self.agent_name = 'RedDeer'
        self.trans_function = None
        self.algo_params = {}
        self.num_males = None
        self.num_hinds = None
        self.num_coms = None
        self.num_stags = None
        self.num_harems = None
        self.males = None
        self.hinds = None
        self.coms = None
        self.stags = None
        self.harem = None
        self.population_pool = None

    def initialize(self):
        super(RDA, self).initialize()

    def user_input(self):
        # accept the parameters as user inputs
        self.algo_params['trans_function'] = input('Shape of Transfer Function [s/v/u]: ') or 's'
        self.trans_function = get_trans_function(self.algo_params['trans_function'])
        self.algo_params['UB'] = 5  # Upper bound
        self.algo_params['LB'] = -5  # Lower bound
        self.algo_params['gamma'] = 0.5  # Fraction of total number of males who are chosen as commanders
        self.algo_params[
            'alpha'] = 0.2  # Fraction of total number of hinds in a harem who mate with the commander of their harem
        self.algo_params[
            'beta'] = 0.1  # Fraction of total number of hinds in a harem who mate with the commander of a different harem

    def roar(self):
        # roaring of male deer
        for i in range(self.num_males):
            r1 = np.random.random()  # r1 is a random number in [0, 1]
            r2 = np.random.random()  # r2 is a random number in [0, 1]
            r3 = np.random.random()  # r3 is a random number in [0, 1]
            new_male = self.males[i].copy()
            if r3 >= 0.5:  # Eq. (3)
                new_male += r1 * (((self.algo_params['UB'] - self.algo_params['LB']) * r2) + self.algo_params['LB'])
            else:
                new_male -= r1 * (((self.algo_params['UB'] - self.algo_params['LB']) * r2) + self.algo_params['LB'])

            # apply transformation function on the new male
            for j in range(self.num_features):
                trans_value = self.trans_function(new_male[j])
                if np.random.random() < trans_value:
                    new_male[j] = 1
                else:
                    new_male[j] = 0

            if self.obj_function(new_male, self.training_data) < self.obj_function(self.males[i], self.training_data):
                self.males[i] = new_male

    def fight(self):
        # fight between male commanders and stags
        for i in range(self.num_coms):
            chosen_com = self.coms[i].copy()
            chosen_stag = random.choice(self.stags)
            r1 = np.random.random()
            r2 = np.random.random()
            new_male_1 = (chosen_com + chosen_stag) / 2 + r1 * (
                        ((self.algo_params['UB'] - self.algo_params['LB']) * r2) + self.algo_params['LB'])  # Eq. (6)
            new_male_2 = (chosen_com + chosen_stag) / 2 - r1 * (
                        ((self.algo_params['UB'] - self.algo_params['LB']) * r2) + self.algo_params['LB'])  # Eq. (7)

            # apply transformation function on new_male_1
            for j in range(self.num_features):
                trans_value = self.trans_function(new_male_1[j])
                if (np.random.random() < trans_value):
                    new_male_1[j] = 1
                else:
                    new_male_1[j] = 0

            # apply transformation function on new_male_2
            for j in range(self.num_features):
                trans_value = self.trans_function(new_male_2[j])
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
                self.coms[i] = chosen_stag.copy()
            elif fitness[0] < fitness[2] and fitness[2] == bestfit:
                self.coms[i] = new_male_1.copy()
            elif fitness[0] < fitness[3] and fitness[3] == bestfit:
                self.coms[i] = new_male_2.copy()

    def form_harems(self):
        # formation of harems
        com_fitness = self.obj_function(self.coms, self.training_data)
        self.coms, com_fitness = sort_agents(agents=self.coms, fitness=com_fitness)
        norm = np.linalg.norm(com_fitness)
        normal_fit = com_fitness / norm
        total = np.sum(normal_fit)
        power = normal_fit / total  # Eq. (9)
        self.num_harems = [int(x * self.num_hinds) for x in power]  # Eq.(10)
        max_harem_size = np.max(self.num_harems)
        self.harem = np.empty(shape=(self.num_coms, max_harem_size, self.num_features))
        random.shuffle(self.hinds)
        itr = 0
        for i in range(self.num_coms):
            harem_size = self.num_harems[i]
            for j in range(harem_size):
                self.harem[i][j] = self.hinds[itr]
                itr += 1

    def mate(self):
        # mating of commander with hinds in his harem
        num_harem_mate = [int(x * self.algo_params['alpha']) for x in self.num_harems]  # Eq. (11)
        self.population_pool = list(self.population)
        for i in range(self.num_coms):
            random.shuffle(self.harem[i])
            for j in range(num_harem_mate[i]):
                r = np.random.random()  # r is a random number in [0, 1]
                offspring = (self.coms[i] + self.harem[i][j]) / 2 + (self.algo_params['UB'] - self.algo_params['LB']) * r  # Eq. (12)

                # apply transformation function on offspring
                for k in range(self.num_features):
                    trans_value = self.trans_function(offspring[k])
                    if (np.random.random() < trans_value):
                        offspring[k] = 1
                    else:
                        offspring[k] = 0
                self.population_pool.append(list(offspring))

                # if number of commanders is greater than 1, inter-harem mating takes place
                if self.num_coms > 1:
                    # mating of commander with hinds in another harem
                    k = i
                    while k == i:
                        k = random.choice(range(self.num_coms))

                    num_mate = int(self.num_harems[k] * self.algo_params['beta'])  # Eq. (13)

                    np.random.shuffle(self.harem[k])
                    for k in range(num_mate):
                        r = np.random.random()  # r is a random number in [0, 1]
                        offspring = (self.coms[i] + self.harem[k][j]) / 2 + (self.algo_params['UB'] - self.algo_params['LB']) * r
                        # apply transformation function on offspring
                        for j in range(self.num_features):
                            trans_value = self.trans_function(offspring[j])
                            if (np.random.random() < trans_value):
                                offspring[j] = 1
                            else:
                                offspring[j] = 0
                        self.population_pool.append(list(offspring))

        # mating of stag with nearest hind
        for stag in self.stags:
            dist = np.zeros(self.num_hinds)
            for i in range(self.num_hinds):
                dist[i] = math.sqrt(np.sum((stag - self.hinds[i]) * (stag - self.hinds[i])))
            min_dist = np.min(dist)
            for i in range(self.num_hinds):
                distance = math.sqrt(np.sum((stag - self.hinds[i]) * (stag - self.hinds[i])))  # Eq. (14)
                if (distance == min_dist):
                    r = np.random.random()  # r is a random number in [0, 1]
                    offspring = (stag + self.hinds[i]) / 2 + (self.algo_params['UB'] - self.algo_params['LB']) * r

                    # apply transformation function on offspring
                    for j in range(self.num_features):
                        trans_value = self.trans_function(offspring[j])
                        if (np.random.random() < trans_value):
                            offspring[j] = 1
                        else:
                            offspring[j] = 0
                    self.population_pool.append(list(offspring))

                    break

    def select_next_generation(self):
        # selection of the next generation
        self.population_pool = np.array(self.population_pool)
        fitness = self.obj_function(self.population_pool, self.training_data)
        self.population_pool, fitness = sort_agents(agents=self.population_pool, fitness=fitness)
        maximum = sum([f for f in fitness])
        selection_probs = [f / maximum for f in fitness]
        indices = np.random.choice(len(self.population_pool), size=self.num_agents, replace=True, p=selection_probs)
        deer = self.population_pool[indices]

    # main loop
    def next(self):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(self.cur_iter + 1))
        print('================================================================================\n')

        # selection of males and hinds
        self.num_males = int(0.25 * self.num_agents)
        self.num_hinds = self.num_agents - self.num_males
        self.males = self.population[:self.num_males, :]
        self.hinds = self.population[self.num_males:, :]

        self.roar()

        # selection of male commanders and stags
        self.num_coms = int(self.num_males * self.algo_params['gamma'])  # Eq. (4)
        self.num_stags = self.num_males - self.num_coms  # Eq. (5)
        self.coms = self.males[:self.num_coms, :]
        self.stags = self.males[self.num_coms:, :]

        self.fight()
        self.form_harems()
        self.mate()
        self.select_next_generation()

        self.cur_iter += 1


############# for testing purpose ################
if __name__ == '__main__':
    data = datasets.load_digits()
    algo = RDA(num_agents=20, max_iter=100, train_data=data.data, train_label=data.target, save_conv_graph=True)
    algo.run()
