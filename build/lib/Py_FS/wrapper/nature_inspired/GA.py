"""

Programmer: Ritam Guha
Date of Development: 6/10/2020

"""

import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets

from Py_FS.wrapper.nature_inspired._utilities import Solution, Data, initialize, sort_agents, display, compute_fitness, compute_accuracy
# from _utilities import Solution, Data, initialize, sort_agents, display, compute_fitness, compute_accuracy


def GA(num_agents, max_iter, train_data, train_label, obj_function=compute_fitness, prob_cross=0.4, prob_mut=0.3, save_conv_graph=False):

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
    chromosomes, fitness = sort_agents(chromosomes, obj_function, data)

    # start timer
    start_time = time.time()

    # main loop
    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no+1))
        print('================================================================================\n')

        # perform crossover, mutation and replacement
        cross_mut(chromosomes, fitness, obj_function, data, prob_cross, cross_limit, prob_mut)

        # update final information
        chromosomes, fitness = sort_agents(chromosomes, obj_function, data)
        display(chromosomes, fitness, agent_name)
        if fitness[0]>Leader_fitness:
            Leader_agent = chromosomes[0].copy()
            Leader_fitness = fitness[0].copy()
        convergence_curve['fitness'][iter_no] = Leader_fitness
        convergence_curve['feature_count'][iter_no] = int(np.sum(Leader_agent))

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


def cross_mut(chromosomes, fitness, obj_function, data, prob_cross, cross_limit, prob_mut):
    # perform crossover, mutation and replacement
    count = 0
    num_agents = chromosomes.shape[0]
    train_X, val_X, train_Y, val_Y = data.train_X, data.val_X, data.train_Y, data.val_Y
    print('Crossover-Mutation phase starting....')

    while(count<cross_limit):
        print('\nCrossover no. {}'.format(count+1))
        id_1 = roulette_wheel(fitness)
        id_2 = roulette_wheel(fitness)

        if(id_1 != id_2):
            child_1, child_2 = crossover(chromosomes[id_1], chromosomes[id_2], prob_cross)
            child_1 = mutation(child_1, prob_mut)
            child_2 = mutation(child_2, prob_mut)
            fitness_1 = obj_function(child_1, train_X, val_X, train_Y, val_Y)
            fitness_2 = obj_function(child_2, train_X, val_X, train_Y, val_Y)

            if(fitness_1 < fitness_2):
                temp = child_1, fitness_1
                child_1, fitness_1 = child_2, fitness_2
                child_2, fitness_2 = temp

            for i in range(num_agents):
                if(fitness_1 > fitness[i]):
                    print('1st child replaced with chromosome having id {}'.format(i+1))
                    chromosomes[i] = child_1
                    fitness[i] = fitness_1
                    break

            for i in range(num_agents):
                if(fitness_2 > fitness[i]):
                    print('2nd child replaced with chromosome having id {}'.format(i+1))
                    chromosomes[i] = child_2
                    fitness[i] = fitness_2
                    break

            count = count+1

        else:
            print('Crossover failed....')
            print('Restarting crossover....\n')





############# for testing purpose ################

if __name__ == '__main__':
    iris = datasets.load_iris()
    GA(10, 20, iris.data, iris.target, save_conv_graph=True)
############# for testing purpose ################
