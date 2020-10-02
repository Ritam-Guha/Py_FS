import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Py_FS.wrapper.nature_inspired._utilities import Solution, Data, initialize, sort_agents, display, compute_accuracy
from sklearn import datasets

def GA(num_agents, max_iter, train_data, train_label, obj_function=compute_accuracy, prob_cross=0.4, prob_mut=0.3):

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
    #                                                                             #
    ###############################################################################

    num_features = train_data.shape[1]
    cross_limit = 5

    # initialize chromosomes and Leader (the agent with the max fitness)
    chromosomes = initialize(num_agents, num_features)
    fitness = np.zeros(num_agents)
    Leader_agent = np.zeros((1, num_features))
    Leader_fitness = float("-inf")

    # initialize convergence curves
    convergence_curve = {}
    convergence_curve['fitness'] = np.zeros(max_iter)
    convergence_curve['feature_count'] = np.zeros(max_iter)

    # format the data 
    data = Data()
    data.train_X, data.val_X, data.train_Y, data.val_Y = train_test_split(train_data, train_label, stratify=train_label, test_size=0.2)

    # create a Solution object
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
        display(chromosomes, fitness)
        Leader_agent = chromosomes[0].copy()
        Leader_fitness = fitness[0].copy()
        convergence_curve['fitness'][iter_no] = Leader_fitness
        convergence_curve['feature_count'][iter_no] = int(np.sum(Leader_agent))

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

    plt.show()

    # update attributes of solution
    solution.best_agent = Leader_agent
    solution.best_fitness = Leader_fitness
    solution.convergence_curve = convergence_curve
    solution.final_population = chromosomes
    solution.final_fitness = fitness
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
    maximum = sum([f for f in fitness])
    selection_probs = [f/maximum for f in fitness]
    return np.random.choice(len(fitness), p=selection_probs)


def cross_mut(chromosomes, fitness, obj_function, data, prob_cross, cross_limit, prob_mut):
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
            fitness_1 = obj_function(child_1, data)
            fitness_2 = obj_function(child_2, data)

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
    GA(10, 20, iris.data, iris.target, compute_accuracy)
    # a = np.array([1, 0, 0, 1, 1])
    # b = np.array([0, 1, 1, 0, 0])
    # c, d = crossover(a, b, 0.4)
    # print(a)
    # print(b)

    # print(c)
    # print(d)

############# for testing purpose ################
