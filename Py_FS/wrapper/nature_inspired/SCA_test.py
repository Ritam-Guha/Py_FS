"""
Programmer: Shameem Ahmed
Date of Development: 19/10/2020
This code has been developed according to the procedures mentioned in the following research article:
"Mirjalili, S. (2016). Sine Cosine Algorithm.
Knowledge Based Systems, 96, 120-133."
"""
import numpy as np
from sklearn import datasets

from Py_FS.datasets import get_dataset
from Py_FS.wrapper.nature_inspired.algorithm import Algorithm
from Py_FS.wrapper.nature_inspired._transfer_functions import get_trans_function

class SCA(Algorithm):
	
	# Sine Cosine Algorithm
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of agents                                              #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #                
    #   obj_function: the function to maximize while doing feature selection      #
    #   trans_function_shape: shape of the transfer function used                 #
    #   save_conv_graph: boolean value for saving convergence graph               #
    #                                                                             #
    ###############################################################################
	
	def __init__(self,num_agents, max_iter, train_data, train_label, trans_function_shape='s',  save_conv_graph=False, seed=0):
        super().__init__(num_agents=num_agents,max_iter=max_iter,train_data=train_data,train_label=train_label,save_conv_graph=save_conv_graph,seed=seed)
        self.algo_name='SCA'
        self.agent_name='Agent'
        self.trans_function=None
        self.algo_params={}
		
	def user_input(self):
        self.algo_params['trans_function'] = input('Shape of Transfer Function [s/v/u]: ') or 's'
        self.trans_function = get_trans_function(self.algo_params['trans_function'])
        self.algo_params['a'] = float(input('Value of constant a: ') or 3)
	
	def next(self):
		
        print('\n================================================================================')
        print('                          Iteration - {}'.format(self.cur_iter+1))
        print('================================================================================\n')
		
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

                temp = self.population[i, j].copy()
                if temp > np.random.random():
                    self.population[i, j] = 1
                else:
                    self.population[i, j] = 0
                
        self.cur_iter += 1
		
if __name__ == '__main__':
    data = datasets.load_digits()
    algo = SCA(num_agents=20, max_iter=30, train_data=data.data, train_label=data.target, save_conv_graph=True)
    algo.run()