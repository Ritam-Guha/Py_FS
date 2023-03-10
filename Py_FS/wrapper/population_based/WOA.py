"""

Programmer: Ritam Guha
Date of Development: 8/10/2020
This code has been developed according to the procedures mentioned in the following research article:
"Mafarja, M., & Mirjalili, S. (2018). Whale optimization approaches for wrapper feature selection. 
Applied Soft Computing, 62, 441-453."

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

class WOA(Algorithm):
    # Whale Optimization Algorithm (WOA)
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

        self.algo_name = 'WOA'
        self.agent_name = 'Whale'
        self.trans_function = None
    
    def user_input(self):
        # first set the default values for the attributes
        self.default_vals["trans_function"] = 's'

        # accept the parameters as user inputs (if default_mode not set)
        if self.default_mode:
            self.set_default()
        else:
            self.algo_params['trans_function'] = input('Shape of Transfer Function [s/v/u] (default=s): ') or 's'
        
        self.trans_function = get_trans_function(self.algo_params['trans_function'])
    
    def forage(self):
        a = 2 - self.cur_iter * (2/self.max_iter)  # a decreases linearly fron 2 to 0
        # update the position of each whale
        for i in range(self.num_agents):
            # update the parameters
            r = np.random.rand() # r is a random number in [0, 1]
            A = (2 * a * r) - a  # Eq. (3)
            C = 2 * r  # Eq. (4)
            l = np.random.uniform(-1, 1)   # l is a random number in [-1, 1]
            p = np.random.random()  # p is a random number in [0, 1]
            b = 1  # defines shape of the spiral               
            
            if p<0.5:
                # Shrinking Encircling mechanism
                if abs(A)>=1:
                    rand_agent_index = np.random.randint(0, self.num_agents)
                    rand_agent = self.population[rand_agent_index, :]
                    mod_dist_rand_agent = abs(C * rand_agent - self.population[i, :]) 
                    self.population[i, :] = rand_agent - (A * mod_dist_rand_agent)   # Eq. (9)
                    
                else:
                    mod_dist_Leader = abs(C * self.Leader_agent - self.population[i, :]) 
                    self.population[i, :] = self.Leader_agent - (A * mod_dist_Leader)  # Eq. (2)
                
            else:
                # Spiral-Shaped Attack mechanism
                dist_Leader = abs(self.Leader_agent - self.population[i, :])
                self.population[i, :] = dist_Leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + self.Leader_agent

            # Apply transformation function on the updated whale
            for j in range(self.num_features):
                trans_value = self.trans_function(self.population[i, j])
                if (np.random.rand() < trans_value): 
                    self.population[i, j] = 1
                else:
                    self.population[i, j] = 0

    def next(self):
        self.print('\n================================================================================')
        self.print('                          Iteration - {}'.format(self.cur_iter+1))
        self.print('================================================================================\n')

        # perform bubble-net foraging actions
        self.forage()

        self.cur_iter += 1

############# for testing purpose ################

if __name__ == '__main__':
    data = datasets.load_digits()
    algo = WOA(num_agents=20, max_iter=20, train_data=data.data, train_label=data.target, default_mode=True)
    solution = algo.run()
############# for testing purpose ################
