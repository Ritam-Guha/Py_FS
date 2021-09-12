"""

Programmer: Ritam Guha
Date of Development: 6/10/2020

"""
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets

from Py_FS.wrapper.nature_inspired._utilities import Solution, Data, initialize, sort_agents, display, compute_fitness, Conv_plot


def GA(num_agents, max_iter, train_data, train_label, obj_function=compute_fitness, prob_cross=0.4, prob_mut=0.3, save_conv_graph=False, seed=0):

    # Genetic Algorithm
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of chromosomes                                         #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #                
    #   obj_function: the function to maximize while doing feature selection      #
    #   prob_cross: probability of crossover                                      #
    #   prob_mut: probability of mutation                                         #
    #   save_conv_graph: boolean value for saving convergence graph               #
    #                                                                             #
    ###############################################################################

    short_name = 'GA'
    agent_name = 'Chromosome'
    train_data, train_label = np.array(train_data), np.array(train_label)
    num_features = train_data.shape[1]
    cross_limit = 5
    np.random.seed(seed)

    # setting up the objectives
    weight_acc = None
    if(obj_function==compute_fitness):
        weight_acc = float(input('Weight for the classification accuracy [0-1]: '))

    obj = (obj_function, weight_acc)
    compute_accuracy = (compute_fitness, 1) # compute_accuracy is just compute_fitness with accuracy weight as 1

    # initialize chromosomes and Leader (the agent with the max fitness)
    chromosomes = initialize(num_agents, num_features)
    fitness = np.zeros(num_agents)
    accuracy = np.zeros(num_agents)
    Leader_agent = np.zeros((1, num_features))
    Leader_fitness = float("-inf")
    Leader_accuracy = float("-inf")

    # initialize convergence curves
    convergence_curve = {}
    convergence_curve['fitness'] = np.zeros(max_iter)

    # initialize data class
    data = Data()
    val_size = float(input('Enter the percentage of data wanted for valdiation [0, 100]: '))/100
    data.train_X, data.val_X, data.train_Y, data.val_Y = train_test_split(train_data, train_label, stratify=train_label, test_size=val_size)

    # create a solution object
    solution = Solution()
    solution.num_agents = num_agents
    solution.max_iter = max_iter
    solution.num_features = num_features
    solution.obj_function = obj_function

    # rank initial population
    chromosomes, fitness = sort_agents(chromosomes, obj, data)

    # start timer
    start_time = time.time()

    # main loop
    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no+1))
        print('================================================================================\n')

        # perform crossover, mutation and replacement
        chromosomes, fitness = cross_mut(chromosomes, fitness, obj, data, prob_cross, cross_limit, prob_mut)
        
        # update final information
        chromosomes, fitness = sort_agents(chromosomes, obj, data, fitness)
        display(chromosomes, fitness, agent_name)

        if fitness[0]>Leader_fitness:
            Leader_agent = chromosomes[0].copy()
            Leader_fitness = fitness[0].copy()

        convergence_curve['fitness'][iter_no] = np.mean(fitness)

    # compute final accuracy
    Leader_agent, Leader_accuracy = sort_agents(Leader_agent, compute_accuracy, data)
    chromosomes, accuracy = sort_agents(chromosomes, compute_accuracy, data)

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

    # plot convergence graph
    fig, axes = Conv_plot(convergence_curve)
    if(save_conv_graph):
        plt.savefig('convergence_graph_'+ short_name + '.jpg')
    plt.show()

    # update attributes of solution
    solution.best_agent = Leader_agent
    solution.best_fitness = Leader_fitness
    solution.best_accuracy = Leader_accuracy
    solution.convergence_curve = convergence_curve
    solution.final_population = chromosomes
    solution.final_fitness = fitness
    solution.final_accuracy = accuracy
    solution.execution_time = exec_time

    return solution


def crossover(parent_1, parent_2, prob_cross):
    # perform crossover with crossover probability prob_cross
    num_features = parent_1.shape[0]
    child_1 = parent_1.copy()
    child_2 = parent_2.copy()

    for i in range(num_features):
        if(np.random.rand()<prob_cross):
            child_1[i] = parent_2[i]
            child_2[i] = parent_1[i]

    return child_1, child_2


def mutation(chromosome, prob_mut):
    # perform mutation with mutation probability prob_mut
    num_features = chromosome.shape[0]
    mut_chromosome = chromosome.copy()

    for i in range(num_features):
        if(np.random.rand()<prob_mut):
            mut_chromosome[i] = 1-mut_chromosome[i]
    
    return mut_chromosome


def roulette_wheel(fitness):
    # Perform roulette wheel selection
    maximum = sum([f for f in fitness])
    selection_probs = [f/maximum for f in fitness]
    return np.random.choice(len(fitness), p=selection_probs)


def cross_mut(chromosomes, fitness, obj, data, prob_cross, cross_limit, prob_mut):
    # perform crossover, mutation and replacement
    count = 0
    num_agents = chromosomes.shape[0]
    print('Crossover-Mutation phase starting....')

    while(count<cross_limit):
        print('\nCrossover no. {}'.format(count+1))
        id_1 = roulette_wheel(fitness)
        id_2 = roulette_wheel(fitness)

        if(id_1 != id_2):
            child_1, child_2 = crossover(chromosomes[id_1], chromosomes[id_2], prob_cross)
            child_1 = mutation(child_1, prob_mut)
            child_2 = mutation(child_2, prob_mut)

            child = np.array([child_1, child_2])
            child, child_fitness = sort_agents(child, obj, data)

            for i in range(2):
                for j in range(num_agents):
                    if(child_fitness[i] > fitness[j]):
                        print('child {} replaced with chromosome having id {}'.format(i+1, j+1))
                        chromosomes[j] = child[i]
                        fitness[j] = child_fitness[i]
                        break

            count = count+1
            

        else:
            print('Crossover failed....')
            print('Restarting crossover....\n')

    return chromosomes, fitness



############# for testing purpose ################

if __name__ == '__main__':
    data = datasets.load_digits()
    GA(20, 100, data.data, data.target, save_conv_graph=True)
############# for testing purpose ################
