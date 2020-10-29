"""

Programmer: Ritam Guha
Date of Development: 8/10/2020
This code has been developed according to the procedures mentioned in the following research article:
"Mafarja, M., & Mirjalili, S. (2018). Whale optimization approaches for wrapper feature selection. 
Applied Soft Computing, 62, 441-453."

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


def WOA(num_agents, max_iter, train_data, train_label, obj_function=compute_fitness, trans_function_shape='s', save_conv_graph=False):

    # Whale Optimization Algorithm
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of whales                                              #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #                
    #   obj_function: the function to maximize while doing feature selection      #
    #   trans_function_shape: shape of the transfer function used                 #
    #   save_conv_graph: boolean value for saving convergence graph               #
    #                                                                             #
    ###############################################################################
    
    short_name = 'WOA'
    agent_name = 'Whale'
    train_data, train_label = np.array(train_data), np.array(train_label)
    num_features = train_data.shape[1]
    cross_limit = 5
    trans_function = get_trans_function(trans_function_shape)

    # initialize whales and Leader (the agent with the max fitness)
    whales = initialize(num_agents, num_features)
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
    data.train_X, data.val_X, data.train_Y, data.val_Y = train_test_split(train_data, train_label, stratify=train_label, test_size=0.2)

    # create a solution object
    solution = Solution()
    solution.num_agents = num_agents
    solution.max_iter = max_iter
    solution.num_features = num_features
    solution.obj_function = obj_function

    # rank initial population
    whales, fitness = sort_agents(whales, obj_function, data)

    # start timer
    start_time = time.time()

    # main loop
    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no+1))
        print('================================================================================\n')

        a = 2 - iter_no * (2/max_iter)  # a decreases linearly fron 2 to 0
        # update the position of each whale
        for i in range(num_agents):
            # update the parameters
            r = np.random.random() # r is a random number in [0, 1]
            A = (2 * a * r) - a  # Eq. (3)
            C = 2 * r  # Eq. (4)
            l = -1 + (np.random.random() * 2)   # l is a random number in [-1, 1]
            p = np.random.random()  # p is a random number in [0, 1]
            b = 1  # defines shape of the spiral               
            
            if p<0.5:
                # Shrinking Encircling mechanism
                if abs(A)>=1:
                    rand_agent_index = np.random.randint(0, num_agents)
                    rand_agent = whales[rand_agent_index, :]
                    mod_dist_rand_agent = abs(C * rand_agent - whales[i,:]) 
                    whales[i,:] = rand_agent - (A * mod_dist_rand_agent)   # Eq. (9)
                    
                else:
                    mod_dist_Leader = abs(C * Leader_agent - whales[i,:]) 
                    whales[i,:] = Leader_agent - (A * mod_dist_Leader)  # Eq. (2)
                
            else:
                # Spiral-Shaped Attack mechanism
                dist_Leader = abs(Leader_agent - whales[i,:])
                whales[i,:] = dist_Leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + Leader_agent

            # Apply transformation function on the updated whale
            for j in range(num_features):
                trans_value = trans_function(whales[i,j])
                if (np.random.random() < trans_value): 
                    whales[i,j] = 1
                else:
                    whales[i,j] = 0

        # update final information
        whales, fitness = sort_agents(whales, obj_function, data)
        display(whales, fitness, agent_name)
        if fitness[0]>Leader_fitness:
            Leader_agent = whales[0].copy()
            Leader_fitness = fitness[0].copy()
        convergence_curve['fitness'][iter_no] = Leader_fitness
        convergence_curve['feature_count'][iter_no] = int(np.sum(Leader_agent))

    # compute final accuracy
    Leader_agent, Leader_accuracy = sort_agents(Leader_agent, compute_accuracy, data)
    whales, accuracy = sort_agents(whales, compute_accuracy, data)

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
    fig.tight_layout(pad = 5) 
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

    # update attributes of solution
    solution.best_agent = Leader_agent
    solution.best_fitness = Leader_fitness
    solution.best_accuracy = Leader_accuracy
    solution.convergence_curve = convergence_curve
    solution.final_population = whales
    solution.final_fitness = fitness
    solution.final_accuracy = accuracy
    solution.execution_time = exec_time

    return solution





############# for testing purpose ################

if __name__ == '__main__':
    iris = datasets.load_iris()
    WOA(10, 20, iris.data, iris.target, save_conv_graph=True)
############# for testing purpose ################
