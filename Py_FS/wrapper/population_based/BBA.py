"""
Programmer: Khalid Hassan
Date of Development: 20/10/2020
This code has been developed according to the procedures mentioned in the following research article:
Mirjalili, S., Mirjalili, S. M., & Yang, X. S. (2014). Binary bat algorithm. 
Neural Computing and Applications, 25(3-4), 663-681.

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

class BBA(Algorithm):
    # Binary Bat Algorithm (BBA)
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

        self.algo_name = 'BBA'
        self.agent_name = 'Bat'
        self.trans_function=None

    def user_input(self):
        # first set the default values for the attributes
        self.default_vals["min_frequency"] = 0
        self.default_vals["max_frequency"] = 2
        self.default_vals["loudness"] = 1.0
        self.default_vals["pulse_emission_rate"] = 0.15
        self.default_vals["alpha"] = 0.95
        self.default_vals["gamma"] = 0.5
        self.default_vals["constant_loudness"] = True
        self.default_vals["trans_function"] = 's'

        # accept the parameters as user inputs (if default_mode not set)
        if self.default_mode:
            self.set_default()
        else:   
            self.algo_params["min_frequency"] = float(input(f"Min Frequency (default={self.default_vals['min_frequency']}): ") or self.default_vals['min_frequency'])
            self.algo_params["max_frequency"] = float(input(f"Max Frequency (default={self.default_vals['max_frequency']}): ") or self.default_vals['max_frequency'])
            self.algo_params["loudness"] = float(input(f"Loudness (default={self.default_vals['loudness']}): ") or self.default_vals['loudness'])
            self.algo_params["pulse_emission_rate"] = float(input(f"Pulse emission rate (default={self.default_vals['pulse_emission_rate']}): ") or self.default_vals['pulse_emission_rate'])
            self.algo_params["alpha"] = float(input(f"Alpha value [0-1] (default={self.default_vals['alpha']}): ") or self.default_vals['alpha'])
            self.algo_params["gamma"] = float(input(f"Gamma value [0-1] (default={self.default_vals['gamma']}): ") or self.default_vals['gamma'])
            self.algo_params["constant_loudness"] = (True if input(f"Constant Loudness (True/False) (default={self.default_vals['constant_loudness']}): ").lower() == "true" else False)
            self.algo_params['trans_function'] = input(f'Shape of Transfer Function [s/v/u] (default={self.default_vals["trans_function"]}): ') or self.default_vals["trans_function"]
        
        self.trans_function = get_trans_function(self.algo_params['trans_function'])

    def initialize(self):
        super().initialize()        

        # Create velocity class
        self.velocity = np.zeros([self.num_agents, self.num_features])

        # Initialize other parameters
        self.A_t = self.algo_params['loudness']
        self.r_t = self.algo_params['pulse_emission_rate']

        # Leader Agent Feature
        self.Leader_agent = self.population[0, :]
        self.Leader_fitness = self.fitness[0]
        self.Leader_accuracy = self.accuracy[0]

    def bat(self):
        if self.algo_params['constant_loudness'] == False:
            self.A_t *= self.algo_params['alpha']
            self.r_t = self.algo_params['pulse_emission_rate'] * \
                (1 - np.exp(-1*self.algo_params['gamma']*self.cur_iter))

        for agentNumber in range(self.num_agents):
            # frequency for i-th agent or bat
            fi = self.algo_params['min_frequency'] + \
                (self.algo_params['max_frequency'] -
                 self.algo_params['min_frequency'])*np.random.rand()

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
        self.print('\n================================================================================')
        self.print('                          Iteration - {}'.format(self.cur_iter+1))
        self.print('================================================================================\n')

        # perform improvisation, replacement
        self.bat()

        self.cur_iter += 1

############# for testing purpose ################
if __name__ == '__main__':
    data = datasets.load_digits()
    algo = BBA(num_agents=20, max_iter=20, train_data=data.data, train_label=data.target, default_mode=True)
    solution = algo.run()
############# for testing purpose ################
