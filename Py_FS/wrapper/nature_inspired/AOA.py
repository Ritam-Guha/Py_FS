"""

Programmer: Soumitri Chattopadhyay
Date of Development: 09/05/2021
This code has been developed according to the procedures mentioned in the following research article:
"Hashim, F.A., Hussain, K., Houssein, E.H. et al. Archimedes Optimization Algorithm.
Applied Intelligence, 51, 1531–1551 (2021)"

"""

import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets

from _utilities import Solution, Data, initialize, sort_agents, display, compute_fitness, compute_accuracy
from _transfer_functions import get_trans_function


def AOA(num_agents, max_iter, train_data, train_label, obj_function=compute_fitness,
                        trans_func_shape='s', save_conv_graph=False):
    # Name of the optimizer
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

    short_name = 'AOA'
    agent_name = 'Particles'
    train_data, train_label = np.array(train_data), np.array(train_label)
    num_features = train_data.shape[1]
    trans_function = get_trans_function(trans_func_shape)

    # setting up the objectives
    weight_acc = None
    if (obj_function == compute_fitness):
        weight_acc = float(input('Weight for the classification accuracy [0-1]: '))
    obj = (obj_function, weight_acc)
    compute_accuracy = (compute_fitness, 1)  # compute_accuracy is just compute_fitness with accuracy weight as 1

    # initialize agents and Leader (the agent with the max fitness)
    agents = initialize(num_agents, num_features)
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
    val_size = float(input('Enter the percentage of data wanted for valdiation [0, 100]: ')) / 100
    data.train_X, data.val_X, data.train_Y, data.val_Y = train_test_split(train_data, train_label, stratify=train_label,
                                                                          test_size=val_size)

    # create a solution object
    solution = Solution()
    solution.num_agents = num_agents
    solution.max_iter = max_iter
    solution.num_features = num_features
    solution.obj_function = obj_function

    # initializing parameters
    C1, C2, C3, C4 = (2, 6, 2, 0.5)
    upper = 0.9
    lower = 0.1

    # initializing agent attributes
    position = np.random.rand(num_agents, num_features)
    volume = np.random.rand(num_agents, num_features)
    density = np.random.rand(num_agents, num_features)
    acceleration = np.random.rand(num_agents, num_features)

    # initializing leader agent attributes
    Leader_position = np.zeros((1, num_features))
    Leader_volume = np.zeros((1, num_features))
    Leader_density = np.zeros((1, num_features))
    Leader_acceleration = np.zeros((1, num_features))

    # rank initial agents
    agents, position, volume, density, acceleration, fitness = sort_agents_(agents, position, volume, density,
                                                                               acceleration, obj, data)
    Leader_agent = agents[0].copy()
    Leader_fitness = fitness[0].copy()
    Leader_position = position[0].copy()
    Leader_volume = volume[0].copy()
    Leader_density = density[0].copy()
    Leader_acceleration = acceleration[0].copy()

    # start timer
    start_time = time.time()

    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no + 1))
        print('================================================================================\n')

        # weight factors
        Tf = np.exp((iter_no - max_iter) / max_iter)
        Df = np.exp((max_iter - iter_no) / max_iter) - (iter_no / max_iter)

        # updating densities and volumes
        for i in range(num_agents):
            for j in range(num_features):
                r1, r2 = np.random.random(2)
                # update density
                density[i][j] = density[i][j] + r1 * (Leader_density[j] - density[i][j])
                # update volume
                volume[i][j] = volume[i][j] + r2 * (Leader_volume[j] - volume[i][j])

        # Exploration phase
        if Tf <= 0.5:
            for i in range(num_agents):
                for j in range(num_features):
                    # update acceleration
                    rand_vol, rand_density, rand_accn = np.random.random(3)
                    acceleration[i][j] = (rand_density + rand_vol * rand_accn) / (density[i][j] * volume[i][j])
                    # update position
                    r1, rand_pos = np.random.random(2)
                    position[i][j] = position[i][j] + C1 * r1 * Df * (rand_pos - position[i][j])

        # Exploitation phase
        else:
            for i in range(num_agents):
                for j in range(num_features):
                    # update acceleration
                    acceleration[i][j] = (Leader_density[j] + Leader_volume[j] * Leader_acceleration[j]) / (
                                density[i][j] * volume[i][j])
                    # update position
                    r2, r3 = np.random.random(2)
                    T_ = C3 * Tf
                    P = 2 * r3 - C4
                    F = 1 if P <= 0.5 else -1
                    position[i][j] = position[i][j] + F * C2 * r2 * acceleration[i][j] * Df * (
                                (T_ * Leader_position[j]) - position[i][j])

        # Normalize accelerations
        for i in range(num_agents):
            max_accn = np.amax(acceleration[i])
            min_accn = np.amin(acceleration[i])
            for j in range(num_features):
                acceleration[i][j] = lower + (acceleration[i][j] - min_accn) / (max_accn - min_accn) * upper

        # Convert to binary: lower acceleration => closer to equilibrium
        for i in range(num_agents):
            for j in range(num_features):
                if np.random.random() < trans_function(acceleration[i][j]):
                    agents[i][j] = 1
                else:
                    agents[i][j] = 0

        ###########################################################################

        # update final information
        agents, position, volume, density, acceleration, fitness = sort_agents_(agents, position, volume, density, acceleration, obj, data)
        display(agents, fitness, agent_name)

        # update Leader (best agent)
        if fitness[0] > Leader_fitness:
            Leader_agent = agents[0].copy()
            Leader_fitness = fitness[0].copy()
            Leader_position = position[0].copy()
            Leader_volume = volume[0].copy()
            Leader_density = density[0].copy()
            Leader_acceleration = acceleration[0].copy()

        convergence_curve['fitness'][iter_no] = Leader_fitness
        convergence_curve['feature_count'][iter_no] = int(np.sum(Leader_agent))

    # compute final accuracy
    Leader_agent, Leader_accuracy = sort_agents(Leader_agent, compute_accuracy, data)
    agents, accuracy = sort_agents(agents, compute_accuracy, data)

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
    iters = np.arange(max_iter) + 1
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

    if (save_conv_graph):
        plt.savefig('convergence_graph_' + short_name + '.jpg')
    plt.show()

    # update attributes of solution
    solution.best_agent = Leader_agent
    solution.best_fitness = Leader_fitness
    solution.best_accuracy = Leader_accuracy
    solution.convergence_curve = convergence_curve
    solution.final_agents = agents
    solution.final_fitness = fitness
    solution.final_accuracy = accuracy
    solution.execution_time = exec_time

    return solution

def sort_agents_(agents, position, volume, density, acceleration, obj, data):
    # sort the agents according to fitness
    train_X, val_X, train_Y, val_Y = data.train_X, data.val_X, data.train_Y, data.val_Y
    (obj_function, weight_acc) = obj

    # if there is only one agent
    if len(agents.shape) == 1:
        num_agents = 1
        fitness = obj_function(agents, train_X, val_X, train_Y, val_Y, weight_acc)
        return agents, position, volume, density, acceleration, fitness

    # for multiple agents
    else:
        num_agents = agents.shape[0]
        fitness = np.zeros(num_agents)
        for id, agent in enumerate(agents):
            fitness[id] = obj_function(agent, train_X, val_X, train_Y, val_Y, weight_acc)
        idx = np.argsort(-fitness)
        sorted_agents = agents[idx].copy()
        sorted_fitness = fitness[idx].copy()
        sorted_position = position[idx].copy()
        sorted_density = density[idx].copy()
        sorted_volume = volume[idx].copy()
        sorted_acceleration = acceleration[idx].copy()

    return sorted_agents, sorted_position, sorted_volume, sorted_density, sorted_acceleration, sorted_fitness

if __name__ == '__main__':
    iris = datasets.load_iris()
    AOA(10, 20, iris.data, iris.target, save_conv_graph=False)
