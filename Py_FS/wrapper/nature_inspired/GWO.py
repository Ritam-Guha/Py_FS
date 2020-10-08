from Py_FS.wrapper.nature_inspired._utilities import Solution, Data, initialize, sort_agents, display, compute_accuracy
import time, random, math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from _transformation_functions import get_trans_function
import numpy as np



def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))


def GWO(num_agents, max_iter, train_data, train_label, obj_function=compute_accuracy):
    # Grey Wolf Optimizer
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: population size                                               #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #
    #   obj_function: the function to maximize while doing feature selection      #
    #                                                                             #
    ###############################################################################

    num_features = train_data.shape[1]

    # initialize chromosomes and Leader (the agent with the max fitness)
    population = initialize(num_agents, num_features)
    fitness = np.zeros(num_agents)
    Leader_agent = np.zeros((1, num_features))
    Leader_fitness = float("-inf")

    # initialize convergence curves
    convergence_curve = {}
    convergence_curve['fitness'] = np.zeros(max_iter)
    convergence_curve['feature_count'] = np.zeros(max_iter)

    # format the data
    data = Data()
    data.train_X, data.val_X, data.train_Y, data.val_Y = train_test_split(
        train_data, train_label, stratify=train_label, test_size=0.2)

    # create a Solution object
    solution = Solution()
    solution.num_agents = num_agents
    solution.max_iter = max_iter
    solution.num_features = num_features
    solution.obj_function = obj_function

    # rank initial population
    population, fitness = sort_agents(population, obj_function, data)

    # start timer
    start_time = time.time()

    # initialize the alpha, beta and delta grey wolves and their fitness
    alpha, beta, delta = np.zeros((1, num_features)), np.zeros((
        1, num_features)), np.zeros((1, num_features))
    alpha_fit, beta_fit, delta_fit = float(
        "-inf"), float("-inf"), float("-inf")

    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no+1))
        print('================================================================================\n')

        # update the alpha, beta and delta grey wolves
        for i in range(num_agents):

            # update alpha, beta, delta
            if fitness[i] > alpha_fit:
                delta_fit = beta_fit
                delta = beta.copy()
                beta_fit = alpha_fit
                beta = alpha.copy()
                alpha_fit = fitness[i]
                alpha = population[i, :].copy()

            # update beta, delta
            elif fitness[i] > beta_fit:
                delta_fit = beta_fit
                delta = beta.copy()
                beta_fit = fitness[i]
                beta = population[i, :].copy()

            # update delta
            elif fitness[i] > delta_fit:
                delta_fit = fitness[i]
                delta = population[i, :].copy()

        # a decreases linearly fron 2 to 0
        a = 2-iter_no*((2)/max_iter)

        for i in range(num_agents):
            for j in range(num_features):

                random.seed(time.time()%10)
                # r1 is a random number in [0,1]
                r1 = np.random.random()

                random.seed(time.time()%7)
                # r2 is a random number in [0,1]
                r2 = np.random.random()

                # calculate A1
                A1 = 2*a*r1-a
                # calculate C1
                C1 = 2*r2

                # find D_alpha
                D_alpha = abs(C1*alpha[j]-population[i, j])
                # calculate distance between alpha and current agent
                X1 = alpha[j]-A1*D_alpha

                random.seed(time.time()%9)
                # r1 is a random number in [0,1]
                r1 = np.random.random()

                # r2 is a random number in [0,1]
                r2 = np.random.random()

                # calculate A2
                A2 = 2*a*r1-a
                # calculate C2
                C2 = 2*r2

                # find D_beta
                D_beta = abs(C2*beta[j]-population[i, j])
                # calculate distance between beta and current agent
                X2 = beta[j]-A2*D_beta

                random.seed(time.time() % 11)
                # r1 is a random number in [0,1]
                r1 = np.random.random()

                # r2 is a random number in [0,1]
                r2 = np.random.random()

                # calculate A3
                A3 = 2*a*r1-a
                # calculate C3
                C3 = 2*r2

                # find D_delta
                D_delta = abs(C3*delta[j]-population[i, j])
                # calculate distance between delta and current agent
                X3 = delta[j]-A3*D_delta

                # update the position of current agent
                population[i, j] = (X1 + X2 + X3) / 3


                # convert to binary values
                random.seed(time.time()*iter_no)

                if (sigmoid(population[i][j]) > np.random.random()):
                    population[i][j] = 1
                else:
                    population[i][j] = 0



        # update final information
        population, fitness = sort_agents(population, obj_function, data)
        display(population, fitness)
        
        # update Leader (best agent)
        if fitness[0] > Leader_fitness:
            Leader_agent = population[0].copy()
            Leader_fitness = fitness[0].copy()

        if alpha_fit > Leader_fitness:
            Leader_fitness = alpha_fit
            Leader_agent = alpha.copy()


        convergence_curve['fitness'][iter_no] = Leader_fitness
        convergence_curve['feature_count'][iter_no] = int(np.sum(Leader_agent))


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

    plt.show()

    # update attributes of solution
    solution.best_agent = Leader_agent
    solution.best_fitness = Leader_fitness
    solution.convergence_curve = convergence_curve
    solution.final_population = population
    solution.final_fitness = fitness
    solution.execution_time = exec_time

    return solution


if __name__ == '__main__':

    iris = datasets.load_iris()
    GWO(10, 20, iris.data, iris.target, compute_accuracy)
