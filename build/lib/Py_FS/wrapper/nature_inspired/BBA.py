"""
Programmer: Khalid Hassan
Date of Development: 20/10/2020
This code has been developed according to the procedures mentioned in the following research article:
Mirjalili, S., Mirjalili, S. M., & Yang, X. S. (2014). Binary bat algorithm. 
Neural Computing and Applications, 25(3-4), 663-681.

"""

import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
# from sklearn import datasets

from Py_FS.wrapper.nature_inspired._utilities import Solution, Data, initialize, sort_agents, display, compute_accuracy
from Py_FS.wrapper.nature_inspired._transfer_functions import get_trans_function
# from _utilities import Solution, Data, initialize, sort_agents, display, compute_accuracy
# from _transfer_functions import get_trans_function

def BBA(num_agents, max_iter, train_data, train_label, obj_function=compute_accuracy, trans_function_shape='s', constantLoudness = True, save_conv_graph = False):
    
    # Binary Bat Algorithm (BBA)
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of chromosomes                                         #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #                
    #   obj_function: the function to maximize while doing feature selection      #
    #   trans_function_shape: shape of the transfer function used                 #
    #   save_conv_graph: boolean value for saving convergence graph               #
    #                                                                             #
    ###############################################################################

    agent_name = "Bat"
    short_name = "BBA"
    train_data, train_label = np.array(train_data), np.array(train_label)
    num_features = train_data.shape[1]
    trans_function = get_trans_function(trans_function_shape)

    # initialize positions of particles and Leader (the agent with the max fitness)
    position = initialize(num_agents, num_features)
    velocity = np.zeros([num_agents, num_features])
    fitness = np.zeros(num_agents)
    Leader_agent = np.zeros((1, num_features))
    Leader_fitness = float("-inf")

    # initialize some important parameters
    minFrequency = 0    # min freq, const if constantLoudness  == True
    maxFrequency = 2    # max freq, const if constantLoudness  == True
    A = 1.00    # loudness
    r = 0.15    # pulse emission rate 

    # initialize convergence curves
    convergence_curve = {}
    convergence_curve['fitness'] = np.zeros(max_iter)
    convergence_curve['feature_count'] = np.zeros(max_iter)

    # initialize data class
    data = Data()
    data.train_X, data.val_X, data.train_Y, data.val_Y = train_test_split(train_data, train_label, stratify=train_label, test_size=0.2)

    # create a solution object
    solution = Solution()
    solution.num_agents = num_agents
    solution.max_iter = max_iter
    solution.num_features = num_features
    solution.obj_function = obj_function

    # start timer
    start_time = time.time()

    position, fitness = sort_agents(position, obj_function, data)

    Leader_agent = position[0, :]
    Leader_fitness = fitness[0]

    alpha = 0.95
    gamma = 0.5
    A_t = A
    r_t = r

    for iterCount in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iterCount + 1))
        print('================================================================================\n')

        if constantLoudness == False:
            A_t *= alpha
            r_t = r*(1 - np.exp(-1*gamma*iterCount))

        for agentNumber in range(num_agents):
            fi = minFrequency + (maxFrequency - minFrequency)*np.random.rand() # frequency for i-th agent or bat

            # update velocity equation number 1 in paper
            velocity[agentNumber, :] = velocity[agentNumber, :] + (position[agentNumber, :] - Leader_agent)*fi


            # updating the position for bat number = agentNumber
            newPos = np.zeros([1, num_features])

            for featureNumber in range(num_features):
                transferValue = trans_function(velocity[agentNumber, featureNumber])

                # change complement position value at dimension number = featureNumber 
                if np.random.rand() < transferValue:
                    newPos[0, featureNumber] = 1 - position[agentNumber, featureNumber]
                else:
                    newPos[0, featureNumber] = position[agentNumber, featureNumber]


                # considering the current pulse rate
                if np.random.rand() > r_t:
                    newPos[0, featureNumber] = Leader_agent[featureNumber]


            ## calculate fitness for new position
            newFit = obj_function(newPos, data.train_X, data.val_X, data.train_Y, data.val_Y)


            ## update better solution for indivisual bat
            if fitness[agentNumber] <= newFit and np.random.rand() <= A_t:
                fitness[agentNumber] = newFit
                position[agentNumber, :] = newPos[0, :]

            ## update (global) best solution for all bats
            if newFit >= Leader_fitness:
                Leader_fitness = newFit
                Leader_agent = newPos[0, :]

        
        convergence_curve['fitness'][iterCount] = Leader_fitness
        convergence_curve['feature_count'][iterCount] = int(np.sum(Leader_agent))

        # display current agents
        display(position, fitness, agent_name)



    # stop timer
    end_time = time.time()
    exec_time = end_time - start_time

    # plot convergence curves
    iters = np.arange(max_iter)+1
    fig, axes = plt.subplots(2, 1)
    fig.tight_layout(pad=5)
    fig.suptitle('Convergence Curves')

    axes[0].set_title('Convergence of Fitness over Iterations')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Fitness')
    axes[0].plot(iters, convergence_curve['fitness'])

    axes[1].set_title('Convergence of Feature Count over Iterations')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Number of Selected Features')
    axes[1].plot(iters, convergence_curve['feature_count'])

    if(save_conv_graph):
        plt.savefig('convergence_graph_'+ short_name + '.jpg')
    plt.show()


    solution.best_agent = Leader_agent
    solution.best_fitness = Leader_fitness
    solution.convergence_curve = convergence_curve
    solution.final_population = position
    solution.final_fitness = fitness
    solution.execution_time = exec_time


    return solution


if __name__ == '__main__':

    iris = datasets.load_iris()
    BBA(10, 20, iris.data, iris.target, compute_accuracy, save_conv_graph=True)











