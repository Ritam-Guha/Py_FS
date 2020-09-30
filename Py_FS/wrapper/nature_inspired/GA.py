import random
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utilities import Solution, Data, initialize, sort_agents, display
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

def GA(num_agents, max_iter, obj_function, train_data, train_label):

    # Genetic Algorithm
    num_features = train_data.shape[1]

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

         ################# GA #################
        
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

############# for testing purpose ################

def compute_accuracy(agents, data): 
    num_agents = agents.shape[0]
    acc = np.zeros(num_agents)

    for id, agent in enumerate(agents):
        cols=np.flatnonzero(agent)     
        if(cols.shape[0]==0):
            return 0    

        clf=KNeighborsClassifier(n_neighbors=5)

        train_data=data.train_X[:,cols]
        train_label = data.train_Y
        val_data=data.val_X[:,cols]
        val_label = data.val_Y

        clf.fit(train_data,train_label)
        acc[id]=clf.score(val_data,val_label)
    return acc

if __name__ == '__main__':
    iris = datasets.load_iris()
    GA(10, 20, compute_accuracy, iris.data, iris.target)

############# for testing purpose ################
