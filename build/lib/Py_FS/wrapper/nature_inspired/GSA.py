
"""

Programmer: Bitanu Chatterjee
Date of Development: 22/10/2020
This code has been developed according to the procedures mentioned in the following research article:
"Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi. 
'GSA: a gravitational search algorithm.'' Information sciences 179.13 (2009): 2232-2248"

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


def GSA(num_agents, max_iter, train_data, train_label, obj_function=compute_fitness, trans_function_shape='s', save_conv_graph=False):

    # Gravitational Search Algorithm
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of particles                                           #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #                
    #   obj_function: the function to maximize while doing feature selection      #
    #   trans_function_shape: shape of the transfer function used                 #
    #   save_conv_graph: boolean value for saving convergence graph               #
    #                                                                             #
    ###############################################################################
    
    agent_name = 'Particle'
    short_name = 'GSA'
    train_data, train_label = np.array(train_data), np.array(train_label)
    num_features = train_data.shape[1]
    trans_function = get_trans_function(trans_function_shape)

    # initialize positionss of particles and Leader (the agent with the max fitness)
    positions = initialize(num_agents, num_features)
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
    
    # initializing parameters
    eps = 0.00001
    G_ini = 6
    F = np.zeros((num_agents, num_agents, num_features))
    R = np.zeros((num_agents, num_agents))
    force = np.zeros((num_agents, num_features))
    acc = np.zeros((num_agents, num_features))
    velocity = np.zeros((num_agents, num_features))
    kBest = range(5)
    
    # rank initial population
    positions, fitness = sort_agents(positions, obj_function, data)
    
    # start timer
    start_time = time.time()

    # main loop
    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no+1))
        print('================================================================================\n')
        
        # updating value of G
        G = G_ini - iter_no * (G_ini / max_iter) # Eq. (13)
        
        # finding mass of each particle
        best_fitness = fitness[0]
        worst_fitness = fitness[-1]
        m = (fitness - worst_fitness) / (best_fitness - worst_fitness + eps) # Eq. (15)
        sum_fitness = np.sum(m) 
        mass = m / sum_fitness # Eq. (16)
                      
        # finding force acting between each pair of particles
        for i in range(num_agents):
            for j in range(num_agents):
                for k in range(num_features):
                    R[i][j] += abs(positions[i][k] - positions[j][k]) # Eq. (8)
                F[i][j] = G * (mass[i] * mass[j]) / (R[i][j] + eps) * (positions[j] - positions[i]) # Eq. (7)
        
        # finding net force acting on each particle
        for i in range(num_agents):
            for j in kBest:
                if i != j:
                    force[i] += np.random.random() * F[i][j] # Eq. (9)
        
        # finding acceleration of each particle
        for i in range(num_agents):
            acc[i] = force[i] / (mass[i] + eps) # Eq. (10)
               
        # updating velocity of each particle
        velocity = np.random.random() * velocity + acc # Eq. (11)
        
        # apply transformation function on the velocity
        for i in range(num_agents):
            for j in range(num_features):
                trans_value = trans_function(velocity[i][j])
                if (np.random.random() < trans_value): 
                    positions[i][j] = 1
                else:
                    positions[i][j] = 0
            if np.sum(positions[i]) == 0:
                x = np.random.randint(0,num_features-1)
                positions[i][x] = 1
                    
        
               
        # update final information
        positions, fitness = sort_agents(positions, obj_function, data)
        display(positions, fitness, agent_name)
        if fitness[0] > Leader_fitness:
            Leader_agent = positions[0].copy()
            Leader_fitness = fitness[0].copy()
        convergence_curve['fitness'][iter_no] = Leader_fitness
        convergence_curve['feature_count'][iter_no] = int(np.sum(Leader_agent))

    # compute final accuracy
    Leader_agent, Leader_accuracy = sort_agents(Leader_agent, compute_accuracy, data)
    positions, accuracy = sort_agents(positions, compute_accuracy, data)

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
    solution.final_population = positions
    solution.final_fitness = fitness
    solution.final_accuracy = accuracy
    solution.execution_time = exec_time
    return solution





############# for testing purpose ################

if __name__ == '__main__':
    iris = datasets.load_iris()
    GSA(10, 20, iris.data, iris.target, save_conv_graph=True)
############# for testing purpose ################


