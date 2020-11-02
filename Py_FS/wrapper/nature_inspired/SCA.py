"""
Programmer: Shameem Ahmed
Date of Development: 19/10/2020
This code has been developed according to the procedures mentioned in the following research article:
"Mirjalili, S. (2016). Sine Cosine Algorithm.
Knowledge Based Systems, 96, 120-133."
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets

from Py_FS.wrapper.nature_inspired._utilities import Solution, Data, initialize, sort_agents, display, compute_fitness, compute_accuracy
from Py_FS.wrapper.nature_inspired._transfer_functions import get_trans_function
# from _utilities import Solution, Data, initialize, sort_agents, display, compute_fitness, compute_accuracy
# from _transfer_functions import get_trans_function



def SCA(num_agents, max_iter, train_data, train_label, obj_function=compute_fitness, trans_func_shape='s', save_conv_graph=False):
    
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

    short_name = 'SCA'
    agent_name = 'Agent'
    train_data, train_label = np.array(train_data), np.array(train_label)
    num_features = train_data.shape[1]
    trans_function = get_trans_function(trans_func_shape)

    # initialize agents and Leader (the agent with the max fitness)
    population = initialize(num_agents, num_features)
    fitness = np.zeros(num_agents)
    accuracy = np.zeros(num_agents)
    Leader_agent = np.zeros((1, num_features))
    Leader_fitness = float("-inf")
    Leader_accuracy = float("-inf")

    # initialize convergence curves
    convergence_curve = {}
    convergence_curve['fitness'] = np.zeros(max_iter)
    convergence_curve['feature_count'] = np.zeros(max_iter)

    # format the data
    data = Data()
    data.train_X, data.val_X, data.train_Y, data.val_Y = train_test_split(
        train_data, train_label, stratify=train_label, test_size=0.2)

    # create a solution object
    solution = Solution()
    solution.num_agents = num_agents
    solution.max_iter = max_iter
    solution.num_features = num_features
    solution.obj_function = obj_function

    # rank initial population
    population, fitness = sort_agents(population, obj_function, data)
    Leader_agent = population[0].copy()
    Leader_fitness = fitness[0].copy()

    # start timer
    start_time = time.time()

    # Eq. (3.4)
    a = 3

    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no+1))
        print('================================================================================\n')

        # Eq. (3.4)
        r1 = a-iter_no*((a)/max_iter)  # r1 decreases linearly from a to 0

        # update the Position of search agents
        for i in range(num_agents):
            for j in range(num_features):

                # update r2, r3, and r4 for Eq. (3.3)
                r2 = (2 * np.pi) * np.random.random()
                r3 = 2 * np.random.random()
                r4 = np.random.random()

                # Eq. (3.3)
                if r4 < 0.5:
                    # Eq. (3.1)
                    population[i, j] = population[i, j] + \
                        (r1*np.sin(r2)*abs(r3*Leader_agent[j]-population[i, j]))
                else:
                    # Eq. (3.2)
                    population[i, j] = population[i, j] + \
                        (r1*np.cos(r2)*abs(r3*Leader_agent[j]-population[i, j]))

                temp = population[i, j].copy()
                temp = trans_function(temp)
                if temp > np.random.random():
                    population[i, j] = 1
                else:
                    population[i, j] = 0


        # update final information
        population, fitness = sort_agents(population, obj_function, data)
        display(population, fitness)

        if fitness[0] > Leader_fitness:
            Leader_agent = population[0].copy()
            Leader_fitness = fitness[0].copy()


        convergence_curve['fitness'][iter_no] = Leader_fitness
        convergence_curve['feature_count'][iter_no] = int(np.sum(Leader_agent))

    # compute final accuracy
    Leader_agent, Leader_accuracy = sort_agents(Leader_agent, compute_accuracy, data)
    population, accuracy = sort_agents(population, compute_accuracy, data)

    print('\n================================================================================')
    print('                                    Final Result                                  ')
    print('================================================================================\n')
    print('Leader ' + agent_name + ' Dimension : {}'.format(int(np.sum(Leader_agent))))
    print('Leader ' + agent_name + ' Fitness : {}'.format(Leader_fitness))
    print('Leader ' + agent_name + ' Classification Accuracy : {}'.format(Leader_accuracy))
    print('\n================================================================================\n')

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
        plt.savefig('convergence_graph_' + short_name + '.jpg')
    plt.show()

    # update attributes of solution
    solution.best_agent = Leader_agent
    solution.best_fitness = Leader_fitness
    solution.best_accuracy = Leader_accuracy
    solution.convergence_curve = convergence_curve
    solution.final_particles = population
    solution.final_fitness = fitness
    solution.final_accuracy = accuracy
    solution.execution_time = exec_time

    return solution


if __name__ == '__main__':

    iris = datasets.load_iris()
    SCA(10, 20, iris.data, iris.target, save_conv_graph=True)
