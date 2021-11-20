"""

Programmer: Ali Hussain Khan
Date of Development: 15/10/2020
This code has been developed according to the procedures mentioned in the following research article:
"Khanesar, M. A., Teshnehlab, M., & Shoorehdeli, M. A. (2007, June). 
A novel binary particle swarm optimization. 
In 2007 Mediterranean conference on control & automation (pp. 1-6). IEEE."

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

class PSO(Algorithm):
    # Particle Swarm Optimization (PSO)
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

        self.algo_name = 'PSO'
        self.agent_name = 'Particle'

    def user_input(self):
         # first set the default values for the attributes
        self.default_vals["trans_function"] = 's'

        # accept the parameters as user inputs (if default_mode not set)
        if self.default_mode:
            self.set_default()
        else:    
            self.algo_params['trans_function'] = input(f"Shape of Transfer Function [s/v/u] (default={self.default_vals['trans_function']}):") or self.default_vals["trans_function"]
        self.trans_function = get_trans_function(self.algo_params['trans_function'])
        
    def initialize(self):
        super().initialize()
        self.global_best_particle = [0 for i in range(self.num_features)]
        self.global_best_fitness = float("-inf")
        self.local_best_particle = [ [ 0 for i in range(self.num_features) ] for j in range(self.num_agents) ] 
        self.local_best_fitness = [float("-inf") for i in range(self.num_agents) ]
        self.weight = 1.0 
        self.velocity = [ [ 0.0 for i in range(self.num_features) ] for j in range(self.num_agents) ]

    def next(self):
        self.print('\n================================================================================')
        self.print('                          Iteration - {}'.format(self.cur_iter+1))
        self.print('================================================================================\n')

        # update weight
        self.weight = 1.0 - (self.cur_iter / self.max_iter)
        
        # update the velocity
        for i in range(self.num_agents):
            for j in range(self.num_features):
                self.velocity[i][j] = (self.weight*self.velocity[i][j])
                r1, r2 = np.random.random(2)
                self.velocity[i][j] = self.velocity[i][j] + (r1 * (self.local_best_particle[i][j] - self.population[i][j]))
                self.velocity[i][j] = self.velocity[i][j] + (r2 * (self.global_best_particle[j] - self.population[i][j]))
       
        # updating position of particles
        for i in range(self.num_agents):
            for j in range(self.num_features):
                trans_value = self.trans_function(self.velocity[i][j])
                if (np.random.random() < trans_value): 
                    self.population[i][j] = 1
                else:
                    self.population[i][j] = 0

        # updating fitness of particles
        self.fitness = self.obj_function(self.population, self.training_data)
        self.population, self.fitness = sort_agents(agents=self.population, fitness=self.fitness)

        # updating the global best and local best particles
        for i in range(self.num_agents):
            if self.fitness[i]>self.local_best_fitness[i]:
                self.local_best_fitness[i]=self.fitness[i]
                self.local_best_particle[i]=self.population[i][:]

            if self.fitness[i]>self.global_best_fitness:
                self.global_best_fitness=self.fitness[i]
                self.global_best_particle=self.population[i][:]

        self.cur_iter += 1


############# for testing purpose ################
if __name__ == '__main__':
    data = datasets.load_digits()
    algo = PSO(num_agents=20, max_iter=20, train_data=data.data, train_label=data.target, default_mode=True)
    solution = algo.run()
############# for testing purpose ################
