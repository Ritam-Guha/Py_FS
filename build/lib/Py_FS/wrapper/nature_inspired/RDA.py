"""

Programmer: Bitanu Chatterjee
Date of Development: 14/10/2020
This code has been developed according to the procedures mentioned in the following research article:
"Fathollahi-Fard, Amir Mohammad, Mostafa Hajiaghaei-Keshteli, and Reza Tavakkoli-Moghaddam. 
'Red deer algorithm (RDA): a new nature-inspired meta-heuristic.''" Soft Computing (2020): 1-29."

"""

import numpy as np
import time
import matplotlib.pyplot as plt
import random, math

from sklearn.model_selection import train_test_split
from sklearn import datasets

from Py_FS.wrapper.nature_inspired._utilities import Solution, Data, initialize, sort_agents, display, compute_fitness, compute_accuracy
from Py_FS.wrapper.nature_inspired._transfer_functions import get_trans_function
# from _utilities import Solution, Data, initialize, sort_agents, display, compute_fitness, compute_accuracy
# from _transfer_functions import get_trans_function


def RDA(num_agents, max_iter, train_data, train_label, obj_function=compute_fitness, trans_function_shape='s', save_conv_graph=False):

    # Red Deer Algorithm
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of red deers                                           #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #                
    #   obj_function: the function to maximize while doing feature selection      #
    #   trans_function_shape: shape of the transfer function used                 #
    #   save_conv_graph: boolean value for saving convergence graph               #
    #                                                                             #
    ###############################################################################
    
    short_name = 'RDA'
    agent_name = 'RedDeer'
    train_data, train_label = np.array(train_data), np.array(train_label)
    num_features = train_data.shape[1]
    trans_function = get_trans_function(trans_function_shape)

    # initialize red deers and Leader (the agent with the max fitness)
    deer = initialize(num_agents, num_features)
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
    UB = 5 # Upper bound
    LB = -5 # Lower bound
    gamma = 0.5 # Fraction of total number of males who are chosen as commanders
    alpha = 0.2 # Fraction of total number of hinds in a harem who mate with the commander of their harem
    beta = 0.1 # Fraction of total number of hinds in a harem who mate with the commander of a different harem

    # start timer
    start_time = time.time()

    # main loop
    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no+1))
        print('================================================================================\n')
        
        deer, fitness = sort_agents(deer, obj_function, data)
        num_males = int(0.25 * num_agents)
        num_hinds = num_agents - num_males
        males = deer[:num_males,:]
        hinds = deer[num_males:,:]
        
        # roaring of male deer
        for i in range(num_males):
            r1 = np.random.random() # r1 is a random number in [0, 1]
            r2 = np.random.random() # r2 is a random number in [0, 1]
            r3 = np.random.random() # r3 is a random number in [0, 1]
            new_male = males[i].copy()
            if r3 >= 0.5:                                    # Eq. (3)
                new_male += r1 * (((UB - LB) * r2) + LB)
            else:
                new_male -= r1 * (((UB - LB) * r2) + LB)
                 
            # apply transformation function on the new male
            for j in range(num_features):
                trans_value = trans_function(new_male[j])
                if (np.random.random() < trans_value): 
                    new_male[j] = 1
                else:
                    new_male[j] = 0
                    
            if obj_function(new_male, data.train_X, data.val_X, data.train_Y, data.val_Y) < obj_function(males[i], data.train_X, data.val_X, data.train_Y, data.val_Y):
                males[i] = new_male
        
        
        # selection of male commanders and stags
        num_coms = int(num_males * gamma) # Eq. (4)
        num_stags = num_males - num_coms # Eq. (5)

        coms = males[:num_coms,:]
        stags = males[num_coms:,:]
        
        # fight between male commanders and stags       
        for i in range(num_coms):
            chosen_com = coms[i].copy()
            chosen_stag = random.choice(stags)
            r1 = np.random.random()
            r2 = np.random.random()
            new_male_1 = (chosen_com + chosen_stag) / 2 + r1 * (((UB - LB) * r2) + LB) # Eq. (6)
            new_male_2 = (chosen_com + chosen_stag) / 2 - r1 * (((UB - LB) * r2) + LB) # Eq. (7)
            
            # apply transformation function on new_male_1
            for j in range(num_features):
                trans_value = trans_function(new_male_1[j])
                if (np.random.random() < trans_value): 
                    new_male_1[j] = 1
                else:
                    new_male_1[j] = 0
             
            # apply transformation function on new_male_2
            for j in range(num_features):
                trans_value = trans_function(new_male_2[j])
                if (np.random.random() < trans_value): 
                    new_male_2[j] = 1
                else:
                    new_male_2[j] = 0
                    
            fitness = np.zeros(4)
            fitness[0] = obj_function(chosen_com, data.train_X, data.val_X, data.train_Y, data.val_Y)
            fitness[1] = obj_function(chosen_stag, data.train_X, data.val_X, data.train_Y, data.val_Y)
            fitness[2] = obj_function(new_male_1, data.train_X, data.val_X, data.train_Y, data.val_Y)
            fitness[3] = obj_function(new_male_2, data.train_X, data.val_X, data.train_Y, data.val_Y)
            
            bestfit = np.max(fitness)
            if fitness[0] < fitness[1] and fitness[1] == bestfit:
                coms[i] = chosen_stag.copy()
            elif fitness[0] < fitness[2] and fitness[2] == bestfit:
                coms[i] = new_male_1.copy()
            elif fitness[0] < fitness[3] and fitness[3] == bestfit:
                coms[i] = new_male_2.copy()

        # formation of harems
        coms, fitness = sort_agents(coms, obj_function, data)
        norm = np.linalg.norm(fitness)
        normal_fit = fitness / norm
        total = np.sum(normal_fit)
        power = normal_fit / total # Eq. (9)
        num_harems = [int(x * num_hinds) for x in power] # Eq.(10)
        max_harem_size = np.max(num_harems)
        harem = np.empty(shape=(num_coms, max_harem_size, num_features))
        random.shuffle(hinds)
        itr = 0
        for i in range(num_coms):
            harem_size = num_harems[i]
            for j in range(harem_size):
                harem[i][j] = hinds[itr]
                itr += 1
        
        # mating of commander with hinds in his harem
        num_harem_mate = [int(x * alpha) for x in num_harems] # Eq. (11)
        population_pool = list(deer)
        for i in range(num_coms):
            random.shuffle(harem[i])
            for j in range(num_harem_mate[i]):
                r = np.random.random() # r is a random number in [0, 1]
                offspring = (coms[i] + harem[i][j]) / 2 + (UB - LB) * r # Eq. (12)
                
                # apply transformation function on offspring
                for j in range(num_features):
                    trans_value = trans_function(offspring[j])
                    if (np.random.random() < trans_value): 
                        offspring[j] = 1
                    else:
                        offspring[j] = 0
                population_pool.append(list(offspring))
                
                # if number of commanders is greater than 1, inter-harem mating takes place
                if num_coms > 1:
                    # mating of commander with hinds in another harem
                    k = i 
                    while k == i:
                        k = random.choice(range(num_coms))

                    num_mate = int(num_harems[k] * beta) # Eq. (13)

                    np.random.shuffle(harem[k])
                    for j in range(num_mate):
                        r = np.random.random() # r is a random number in [0, 1]
                        offspring = (coms[i] + harem[k][j]) / 2 + (UB - LB) * r 
                        # apply transformation function on offspring
                        for j in range(num_features):
                            trans_value = trans_function(offspring[j])
                            if (np.random.random() < trans_value): 
                                offspring[j] = 1
                            else:
                                offspring[j] = 0
                        population_pool.append(list(offspring))
        
        # mating of stag with nearest hind
        for stag in stags:
            dist = np.zeros(num_hinds)
            for i in range(num_hinds):
                dist[i] = math.sqrt(np.sum((stag-hinds[i])*(stag-hinds[i])))
            min_dist = np.min(dist)
            for i in range(num_hinds):
                distance = math.sqrt(np.sum((stag-hinds[i])*(stag-hinds[i]))) # Eq. (14)
                if(distance == min_dist):
                    r = np.random.random() # r is a random number in [0, 1]
                    offspring = (stag + hinds[i])/2 + (UB - LB) * r
                    
                    # apply transformation function on offspring
                    for j in range(num_features):
                        trans_value = trans_function(offspring[j])
                        if (np.random.random() < trans_value): 
                            offspring[j] = 1
                        else:
                            offspring[j] = 0
                    population_pool.append(list(offspring))
                    
                    break
        
        # selection of the next generation
        population_pool = np.array(population_pool)            
        population_pool, fitness = sort_agents(population_pool, obj_function, data)
        maximum = sum([f for f in fitness])
        selection_probs = [f/maximum for f in fitness]
        indices = np.random.choice(len(population_pool), size=num_agents, replace=True, p=selection_probs)          
        deer = population_pool[indices]
        
        # update final information
        deer, fitness = sort_agents(deer, obj_function, data)
        display(deer, fitness, agent_name)
        if fitness[0] > Leader_fitness:
            Leader_agent = deer[0].copy()
            Leader_fitness = fitness[0].copy()
        convergence_curve['fitness'][iter_no] = Leader_fitness
        convergence_curve['feature_count'][iter_no] = int(np.sum(Leader_agent))

    # compute final accuracy
    Leader_agent, Leader_accuracy = sort_agents(Leader_agent, compute_accuracy, data)
    deer, accuracy = sort_agents(deer, compute_accuracy, data)

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
    solution.final_population = deer
    solution.final_fitness = fitness
    solution.final_accuracy = accuracy
    solution.execution_time = exec_time
    return solution





############# for testing purpose ################

if __name__ == '__main__':
    iris = datasets.load_iris()
    RDA(10, 20, iris.data, iris.target, save_conv_graph=True)
############# for testing purpose ################
