# Programmer : Khalid Hassan

import random
import time
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from Py_FS.wrapper.nature_inspired.algorithm import Algorithm
from Py_FS.wrapper.population_based._utilities import Data, compute_fitness, initialize, sort_agents, compute_accuracy, call_counter


class BBA(Algorithm):
    # Binary Bat Algorithm (BBA)
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of bats                                                #
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

        self.algo_name = 'BBA'
        self.agent_name = 'Bat'
        self.algo_params = {}

        # self.num_features = self.population[0, :].shape[0]
        # self.velocity = np.zeros([self.num_agents, self.num_features])

    def trans_function(self, val, shape='s'):
        if (shape.lower() == 's'):
            if val < 0:
                return 1 - 1/(1 + np.exp(val))
            else:
                return 1/(1 + np.exp(-val))

        elif (shape.lower() == 'v'):
            return abs(val/(np.sqrt(1 + val*val)))

        elif(shape.lower() == 'u'):
            alpha, beta = 2, 1.5
            return abs(alpha * np.power(abs(val), beta))

        else:
            print(
                '\n[Error!] We don\'t currently support {}-shaped transfer functions...\n'.format(shape))
            exit(1)

    def user_input(self):
        self.algo_params['minFrequency'] = float(
            input('Min Frequency : ') or 0)
        self.algo_params['maxFrequency'] = float(
            input('Max Frequency: ') or 2)
        self.algo_params['loudness'] = float(input('Loudness: ') or 1.00)
        self.algo_params['pulseEmissionRate'] = float(
            input('Pulse emission rate : ') or 0.15)

        self.algo_params['alpha'] = float(
            input('Alpha value [0-1] : ') or 0.95)

        self.algo_params['gamma'] = float(
            input('Gamma value [0-1] : ') or 0.5)

        self.algo_params['constantLoudness'] = (True if input(
            "Constant Loudness (T/F): ").lower() == "t" else False)

        # self.A_t = self.algo_params['loudness']
        # self.r_t = self.algo_params['pulseEmissionRate']

    def initialize(self):
        # set the objective function
        self.val_size = float(
            input('Percentage of data for valdiation [0-100]: ') or 30)/100
        self.weight_acc = float(
            input('Weight for the classification accuracy [0-1]: ') or 0.9)
        self.obj_function = call_counter(compute_fitness(self.weight_acc))

        # start timer
        self.start_time = time.time()
        np.random.seed(self.seed)

        # data preparation
        self.training_data = Data()
        self.train_data, self.train_label = np.array(
            self.train_data), np.array(self.train_label)
        self.training_data.train_X, self.training_data.val_X, self.training_data.train_Y, self.training_data.val_Y = train_test_split(
            self.train_data, self.train_label, stratify=self.train_label, test_size=self.val_size)

        # create initial population
        num_features = self.train_data.shape[1]
        self.population = initialize(
            num_agents=self.num_agents, num_features=num_features)
        self.fitness = self.obj_function(self.population, self.training_data)
        self.population, self.fitness = sort_agents(
            agents=self.population, fitness=self.fitness)
        self.accuracy = compute_accuracy(
            agents=self.population, data=self.training_data)

        # Create velocity class
        self.num_features = self.population[0, :].shape[0]
        self.velocity = np.zeros([self.num_agents, self.num_features])

        # Initialize other parameters
        self.A_t = self.algo_params['loudness']
        self.r_t = self.algo_params['pulseEmissionRate']

        # Leader Agent Feature
        self.Leader_agent = self.population[0, :]
        self.Leader_fitness = self.fitness[0]
        self.Leader_accuracy = self.accuracy[0]

    def bat(self):
        if self.algo_params['constantLoudness'] == False:
            self.A_t *= self.algo_params['alpha']
            self.r_t = self.algo_params['pulseEmissionRate'] * \
                (1 - np.exp(-1*self.algo_params['gamma']*self.cur_iter))

        for agentNumber in range(self.num_agents):
            # frequency for i-th agent or bat
            fi = self.algo_params['minFrequency'] + \
                (self.algo_params['maxFrequency'] -
                 self.algo_params['minFrequency'])*np.random.rand()

            # update velocity equation number 1 in paper
            self.velocity[agentNumber, :] = self.velocity[agentNumber, :] + \
                (self.population[agentNumber, :] - self.Leader_agent)*fi

            # updating the bats for bat number = agentNumber
            newPos = np.zeros([1, self.num_features])

            for featureNumber in range(self.num_features):
                transferValue = self.trans_function(
                    self.velocity[agentNumber, featureNumber])

                # change complement bats value at dimension number = featureNumber
                if np.random.rand() < transferValue:
                    newPos[0, featureNumber] = 1 - \
                        self.population[agentNumber, featureNumber]
                else:
                    newPos[0, featureNumber] = self.population[agentNumber,
                                                               featureNumber]

                # considering the current pulse rate
                if np.random.rand() > self.r_t:
                    newPos[0, featureNumber] = self.Leader_agent[featureNumber]

            # calculate fitness for new bats
            newFit = self.obj_function(
                newPos, self.training_data)

            # update better solution for indivisual bat
            if self.fitness[agentNumber] <= newFit and np.random.rand() <= self.A_t:
                self.fitness[agentNumber] = newFit
                self.population[agentNumber, :] = newPos[0, :]

        self.population, self.fitness = sort_agents(
            self.population, self.fitness)

        # update (global) best solution for all bats
        if self.fitness[0] > self.Leader_fitness:
            self.Leader_fitness = self.fitness[0]
            self.Leader_agent = self.population[0, :]

    def next(self):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(self.cur_iter+1))
        print('================================================================================\n')

        # perform improvisation, replacement
        self.bat()

        self.cur_iter += 1


if __name__ == '__main__':
    data = datasets.load_digits()
    algo = BBA(num_agents=20, max_iter=5,
               train_data=data.data, train_label=data.target)
    solution = algo.run()
