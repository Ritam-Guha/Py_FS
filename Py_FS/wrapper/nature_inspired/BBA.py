"""
    Programmer: Khalid Hassan
    Github: //khalid0007
     
    Paper: Binary bat algorithm
    Authors: Seyedali Mirjalili, Seyed Mohammad Mirjalili, Xin-She Yang
"""

import numpy as np
import matplotlib.pyplot as plt
import math, time, sys, random

from sklearn.model_selection import train_test_split
from sklearn import datasets

from _utilities import Solution, Data, initialize, sort_agents, display, compute_accuracy

def vShapedTransferFunction(val):
    return abs((2/math.pi)*math.atan((math.pi)*0.5*val))

def BBA(num_agents, max_iter, train_data, train_label, obj_function = compute_accuracy, minFrequency = 0, maxFrequency = 2, A = 1.00, r = 0.15, constantLoudness = True, save_conv_graph = False):
    # Binary Bat Algorithm (BBA)
    # Parameters
    ############################### Parameters ####################################
    #   num_agents: number of harmonies                                           #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #                
    #   obj_function: the function to maximize while doing feature selection      #
    #   A: loudness                                                               #
    #   r: pulse emission rate                                                    #
    #   minFrequency: Minimum Frequency                                           #
    #   maxFrequency: Maximum Frequency                                           #
    ###############################################################################

    # A is the loudness  (constant or decreasing) ## const if constantLoudness  == True
    # r is the pulse rate (constant or decreasing) ## const if constantLoudness  == True


    train_data, train_label = np.array(train_data), np.array(train_label)
    num_features = train_data.shape[1]

    short_name = "BBA"
    agent_name = "Binary Bat Algorithm"

    # Intialisation
    position = initialize(num_agents, num_features);
    velocity = np.zeros([num_agents, num_features]);
    fitness = np.zeros(num_agents)
    Leader_agent = np.zeros((1, num_features))
    Leader_fitness = float("-inf")

    # initialize convergence curves
    convergence_curve = {}
    convergence_curve['fitness'] = np.zeros(max_iter)
    convergence_curve['feature_count'] = np.zeros(max_iter)

    # initialise data class
    data = Data()
    data.train_X, data.val_X, data.train_Y, data.val_Y = train_test_split(train_data, train_label, stratify=train_label, test_size=0.2)

    # Initialise solution class
    solution = Solution()
    solution.num_agents = num_agents
    solution.max_iter = max_iter
    solution.num_features = num_features
    solution.obj_function = obj_function

    # start timer
    start_time = time.time()

    position, fitness = sort_agents(position, obj_function, data)

    Leader_agent = position[0, :];
    Leader_fitness = fitness[0];

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
            r_t = r*(1 - math.exp(-1*gamma*iterCount))

        for agentNumber in range(num_agents):
            fi = minFrequency + (maxFrequency - minFrequency)*np.random.rand() # frequency for i-th agent or bat

            # Update velocity equation number 1 in paper
            velocity[agentNumber, :] = velocity[agentNumber, :] + (position[agentNumber, :] - Leader_agent)*fi


            ## Updating the position for bat number = agentNumber
            newPos = np.zeros([1, num_features])

            for featureNumber in range(num_features):
                transferValue = vShapedTransferFunction(velocity[agentNumber, featureNumber])

                # change complement position value at dimension number = featureNumber 
                if np.random.rand() < transferValue:
                    newPos[0, featureNumber] = 1 - position[agentNumber, featureNumber]
                else:
                    newPos[0, featureNumber] = position[agentNumber, featureNumber]


                # Considering the current pulse rate
                if np.random.rand() > r_t:
                    newPos[0, featureNumber] = Leader_agent[featureNumber]


            ## Calculate fitness for new position
            newFit = obj_function(newPos, data.train_X, data.val_X, data.train_Y, data.val_Y)


            ## update better solution for indivisual bat
            if fitness[agentNumber] <= newFit and np.random.rand() <= A_t:
                fitness[agentNumber] = newFit
                position[agentNumber, :] = newPos[0, :]

            ## Update (global) best solution for all bats
            if newFit >= Leader_fitness:
                Leader_fitness = newFit
                Leader_agent = newPos[0, :]

        
        convergence_curve['fitness'][iterCount] = Leader_fitness;
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
        plt.savefig('convergence_graph_'+ short_name + '.png')
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
    BBA(10, 20, iris.data, iris.target, compute_accuracy, constantLoudness = True, save_conv_graph=True)











