"""
Programmer: Trinav Bhattacharyya
Date of Development: 18/10/2020
This code has been developed according to the procedures mentioned in the following research article:
X.-S. Yang, S. Deb, “Cuckoo search via Levy flights”, in: Proc. of
World Congress on Nature & Biologically Inspired Computing (NaBIC 2009),
December 2009, India. IEEE Publications, USA, pp. 210-214 (2009).

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

class CS(Algorithm):
    # Cuckoo Search (CS)
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

        self.algo_name='CS'
        self.agent_name='Cuckoo'
        self.trans_function=None
        
    def user_input(self):
        # first set the default values for the attributes
        self.default_vals["trans_function"] = 's'
        self.default_vals["p_a"] = 0.25

        # accept the parameters as user inputs (if default_mode not set)
        if self.default_mode:
            self.set_default()
        else:    
            self.algo_params['trans_function'] = input(f'Shape of Transfer Function [s/v/u] (default={self.default_vals["trans_function"]}): ') or self.default_vals["trans_function"]
            self.algo_params['p_a'] = float(input(f'Fraction of nests to be replaced [0-1] (default={self.default_vals["p_a"]}): ') or self.default_vals["p_a"])
            
        self.trans_function = get_trans_function(self.algo_params['trans_function'])
        
    def initialize(self):
        #call the base class function
        super().initialize()
        
        self.levy_flight = np.random.uniform(low=-2, high=2, size=(self.num_features))
        self.cuckoo = np.random.randint(low=0, high=2, size=(self.num_features))
        self.cuckoo_fitness = self.obj_function(self.cuckoo,self.training_data)
        
        #rank initial nests
        self.fitness = self.obj_function(self.population,self.training_data)
        self.population,self.fitness = sort_agents(self.population,self.fitness)
        
    def get_cuckoo(self,agent, alpha=np.random.randint(-2,3)):
        features = len(agent)
        lamb = np.random.uniform(low=-3, high=-1, size=(features))
        levy = np.zeros((features))
        get_test_value = 1/(np.power((np.random.normal(0,1)),2))
        for j in range(features):
            levy[j] = np.power(get_test_value, lamb[j])
        for j in range(features):
            agent[j]+=(alpha*levy[j])

        return agent
        
    def replace_worst(self,agent, fraction):
        pop, features = agent.shape
        for i in range(int((1-fraction)*pop), pop):
            agent[i] = np.random.randint(low=0, high=2, size=(features))
            if np.sum(agent[i])==0:
                agent[i][np.random.randint(0,features)]=1

        return agent
        
    def next(self):
        self.print('\n================================================================================')
        self.print('                          Iteration - {}'.format(self.cur_iter+1))
        self.print('================================================================================\n')
        
        # get new cuckoo
        self.levy_flight = self.get_cuckoo(self.levy_flight)
        for j in range(self.num_features):
            if self.trans_function(self.levy_flight[j]) > np.random.random():
                self.cuckoo[j]=1
            else:
                self.cuckoo[j]=0
        self.cuckoo_fitness = self.obj_function(self.cuckoo,self.training_data)
        
        # check if a nest needs to be replaced
        j = np.random.randint(0,self.num_agents)
        if self.cuckoo_fitness > self.fitness[j]:
            self.population[j] = self.cuckoo.copy()
            self.fitness[j] = self.cuckoo_fitness
            
        self.fitness = self.obj_function(self.population,self.training_data)
        self.population, self.fitness = sort_agents(self.population,self.fitness)
        
        # eliminate worse nests and generate new ones
        self.population = self.replace_worst(self.population, self.algo_params['p_a'])
        
        self.cur_iter += 1
        
############# for testing purpose ################
if __name__ == '__main__':
    data = datasets.load_digits()
    algo = CS(num_agents=20, max_iter=20, train_data=data.data, train_label=data.target, default_mode=True)
    algo.run()
############# for testing purpose ################