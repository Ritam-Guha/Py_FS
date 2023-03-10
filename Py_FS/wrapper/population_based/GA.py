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

class GA(Algorithm):
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

        self.algo_name = 'GA'
        self.agent_name = 'Chromosome'

    def user_input(self):
        # first set the default values for the attributes
        self.default_vals["prob_cross"] = 0.7
        self.default_vals["prob_mut"] = 0.3
        self.default_vals["cross_limit"] = 5

        # accept the parameters as user inputs (if default_mode not set)
        if self.default_mode:
            self.set_default()
        else:
            self.algo_params['prob_cross'] = float(input(f'Probability of crossover [0-1] (default={self.default_vals["prob_cross"]}): ') or self.default_vals["prob_cross"])
            self.algo_params['prob_mut'] = float(input(f'Probability of mutation [0-1] (default={self.default_vals["prob_mut"]}): ') or self.default_vals["prob_mut"])
            self.algo_params['cross_limit'] = float(input(f'Max crossover in every Generation [5-10] (default={self.default_vals["cross_limit"]}): ') or self.default_vals["cross_limit"])
        
    def crossover(self, parent_1, parent_2):
        # perform crossover with crossover probability prob_cross
        num_features = parent_1.shape[0]
        child_1 = parent_1.copy()
        child_2 = parent_2.copy()

        for i in range(num_features):
            if(np.random.rand() < self.algo_params['prob_cross']):
                child_1[i] = parent_2[i]
                child_2[i] = parent_1[i]

        return child_1, child_2

    def mutation(self, chromosome):
        # perform mutation with mutation probability prob_mut
        num_features = chromosome.shape[0]
        mut_chromosome = chromosome.copy()

        for i in range(num_features):
            if(np.random.rand() < self.algo_params['prob_mut']):
                mut_chromosome[i] = 1-mut_chromosome[i]
        
        return mut_chromosome

    def roulette_wheel(self, fitness):
        # perform roulette wheel selection
        maximum = sum([f for f in fitness])
        selection_probs = [f/maximum for f in fitness]
        return np.random.choice(len(fitness), p=selection_probs)

    def cross_mut(self):
        # perform crossover, mutation and replacement
        count = 0
        self.print('Crossover-Mutation phase starting....')

        while(count < self.algo_params['cross_limit']):
            self.print('\nCrossover no. {}'.format(count+1))
            id_1 = self.roulette_wheel(self.fitness)
            id_2 = self.roulette_wheel(self.fitness)

            if(id_1 != id_2):
                child_1, child_2 = self.crossover(self.population[id_1, :], self.population[id_2, :])
                child_1 = self.mutation(child_1)
                child_2 = self.mutation(child_2)

                child = np.array([child_1, child_2])
                child_fitness = self.obj_function(child, self.training_data)
                child, child_fitness = sort_agents(child, child_fitness)

                for i in range(2):
                    for j in reversed(range(self.num_agents)):
                        if(child_fitness[i] > self.fitness[j]):
                            self.print('child {} replaced with chromosome having id {}'.format(i+1, j+1))
                            self.population[j, :] = child[i]
                            self.fitness[j] = child_fitness[i]
                            break
                
                count = count+1

            else:
                self.print('Crossover failed....')
                self.print('Restarting crossover....\n')

    def next(self):
        self.print('\n================================================================================')
        self.print('                          Iteration - {}'.format(self.cur_iter+1))
        self.print('================================================================================\n')

        # perform crossover, mutation and replacement
        self.cross_mut()

        self.cur_iter += 1

############# for testing purpose ################
if __name__ == '__main__':
    data = datasets.load_digits()
    algo = GA(num_agents=20, max_iter=20, train_data=data.data, train_label=data.target, default_mode=True)
    solution = algo.run()
############# for testing purpose ################
