"""

Programmer: Ritam Guha
Date of Development: 18/10/2020
This code has been developed according to the procedures mentioned in the following research article:
"Faramarzi, A., Heidarinejad, M., Stephens, B., & Mirjalili, S. (2020). Equilibrium optimizer: A novel optimization algorithm. 
Knowledge-Based Systems, 191, 105190."

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

class EO(Algorithm):
    # Equilibrium Optimizer (EO)
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

        self.algo_name='EO'
        self.agent_name='Particle'
        self.trans_function=None
        self.algo_params["a2"] = 1
        self.algo_params["a1"] = 2
        self.algo_params["GP"] = 0.5
        
    def user_input(self):
        # first set the default values for the attributes
        self.default_vals["trans_function"] = 's'
        self.default_vals["pool_size"] = 4
        
        # accept the parameters as user inputs (if default_mode not set)
        if self.default_mode:
            self.set_default()
        else:    
            self.algo_params['trans_function'] = input(f'Shape of Transfer Function [s/v/u] (default={self.default_vals["trans_function"]}): ') or self.default_vals["trans_function"]
            self.algo_params['pool_size'] = int(input(f'Fraction of nests to be replaced (default={self.default_vals["pool_size"]}): ') or self.default_vals["pool_size"])
            
        self.trans_function = get_trans_function(self.algo_params['trans_function'])
        
    def initialize(self):
        #call the base class function
        super().initialize()
        
        # pool initialization
        self.eq_pool = np.zeros((self.algo_params["pool_size"]+1, self.num_features))
        self.eq_fitness = np.zeros(self.algo_params["pool_size"])
        self.eq_fitness[:] = float("-inf") 

    def avg_concentration(self, eq_pool, pool_size, dimension): 
        # function to compute average concentration of the equilibrium pool   
        avg = np.sum(eq_pool[0:pool_size,:], axis=0)         
        avg = avg/pool_size
        return avg    

    def next(self):
        self.print('\n================================================================================')
        self.print('                          Iteration - {}'.format(self.cur_iter+1))
        self.print('================================================================================\n')
        
        # replacements in the pool
        for i in range(self.num_agents):
            for j in range(self.algo_params["pool_size"]):                 
                if self.fitness[i] <= self.eq_fitness[j]:
                    self.eq_fitness[j] = self.fitness[i].copy()
                    self.eq_pool[j, :] = self.population[i, :].copy()
                    break
        

        best_particle = self.eq_pool[0,:]
                
        Cave = self.avg_concentration(self.eq_pool, self.algo_params["pool_size"], self.num_features)
        self.eq_pool[self.algo_params["pool_size"]] = Cave.copy()

        t = (1 - (self.cur_iter/self.max_iter)) ** (self.algo_params["a2"]*(self.cur_iter/self.max_iter))
        
        for i in range(self.num_agents):
            
            # randomly choose one candidate from the equillibrium pool
            inx = np.random.randint(0,self.algo_params["pool_size"])
            Ceq = np.array(self.eq_pool[inx])

            lambda_vec = np.zeros(np.shape(Ceq))
            r_vec = np.zeros(np.shape(Ceq))
            for j in range(self.num_features):
                lambda_vec[j] = np.random.random()
                r_vec[j] = np.random.random()

            F_vec = np.zeros(np.shape(Ceq))
            for j in range(self.num_features):
                x = -1*lambda_vec[j]*t 
                x = np.exp(x) - 1
                x = self.algo_params["a1"] * np.sign(r_vec[j] - 0.5) * x

            r1, r2 = np.random.random(2)
            if r2 < self.algo_params["GP"]:
                GCP = 0
            else:
                GCP = 0.5 * r1
            G0 = np.zeros(np.shape(Ceq))
            G = np.zeros(np.shape(Ceq))
            for j in range(self.num_features):
                G0[j] = GCP * (Ceq[j] - lambda_vec[j]*self.population[i][j])
                G[j] = G0[j]*F_vec[j]
            
            # use transfer function to map continuous->binary
            for j in range(self.num_features):
                temp = Ceq[j] + (self.population[i][j] - Ceq[j])*F_vec[j] + G[j]*(1 - F_vec[j])/lambda_vec[j]                
                temp = self.trans_function(temp)                
                if temp>np.random.random():
                    self.population[i][j] = 1 - self.population[i][j]
                else:
                    self.population[i][j] = self.population[i][j] 
        
        self.cur_iter += 1
        
############# for testing purpose ################
if __name__ == '__main__':
    data = datasets.load_digits()
    algo = EO(num_agents=20, max_iter=20, train_data=data.data, train_label=data.target, default_mode=True)
    solution = algo.run()
############# for testing purpose ################