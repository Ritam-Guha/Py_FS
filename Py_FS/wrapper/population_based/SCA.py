"""
Programmer: Shameem Ahmed
Date of Development: 19/10/2020
This code has been developed according to the procedures mentioned in the following research article:
"Mirjalili, S. (2016). Sine Cosine Algorithm.
Knowledge Based Systems, 96, 120-133."
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

class SCA(Algorithm):
    # Sine Cosine Algorithm (SCA)
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

        self.algo_name='SCA'
        self.agent_name='Agent'
        self.trans_function=None
	
    def user_input(self):
        # first set the default values for the attributes
        self.default_vals["a"] = 3
        self.default_vals["trans_function"] = 's'

        # accept the parameters as user inputs (if default_mode not set)
        if self.default_mode:
            self.set_default()
        else:    
            self.algo_params['trans_function'] = input(f'Shape of Transfer Function [s/v/u] (default={self.default_vals["trans_function"]}): ') or self.default_vals["trans_function"]
            self.algo_params['a'] = float(input(f"Value of constant a (default={self.default_vals['a']}): ") or self.default_vals['a'])
        
        self.trans_function = get_trans_function(self.algo_params['trans_function'])
	
    def next(self):
        self.print('\n================================================================================')
        self.print('                          Iteration - {}'.format(self.cur_iter+1))
        self.print('================================================================================\n')
		
        # Eq. (3.4)
        r1 = self.algo_params['a'] * (1 - self.cur_iter / self.max_iter)  # r1 decreases linearly from a to 0
		
        # update the Position of search agents
        for i in range(self.num_agents):
            for j in range(self.num_features):

                # update r2, r3, and r4 for Eq. (3.3)
                r2 = (2 * np.pi) * np.random.random()
                r3 = 2 * np.random.random()
                r4 = np.random.random()

                # Eq. (3.3)
                if r4 < 0.5:
                    # Eq. (3.1)
                    self.population[i, j] = self.population[i, j] + \
                        (r1*np.sin(r2)*abs(r3*self.Leader_agent[j]-self.population[i, j]))
                else:
                    # Eq. (3.2)
                    self.population[i, j] = self.population[i, j] + \
                        (r1*np.cos(r2)*abs(r3*self.Leader_agent[j]-self.population[i, j]))

                temp = self.trans_function(self.population[i, j])
                if temp > np.random.random():
                    self.population[i, j] = 1
                else:
                    self.population[i, j] = 0
                
        self.cur_iter += 1
		
############# for testing purpose ################
if __name__ == '__main__':
    data = datasets.load_digits()
    algo = SCA(num_agents=20, max_iter=20, train_data=data.data, train_label=data.target, default_mode=True)
    solution = algo.run()
############# for testing purpose ################