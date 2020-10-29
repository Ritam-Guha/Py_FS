"""

Programmer: Ritam Guha
Date of Development: 18/10/2020
This code has been developed according to the procedures mentioned in the following research article:
"Faramarzi, A., Heidarinejad, M., Stephens, B., & Mirjalili, S. (2020). Equilibrium optimizer: A novel optimization algorithm. 
Knowledge-Based Systems, 191, 105190."

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


def EO(num_agents, max_iter, train_data, train_label, obj_function=compute_fitness, trans_func_shape='s', save_conv_graph=False):
    
    # Equilibrium Optimizer
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
    
    short_name = 'EO'
    agent_name = 'Particle'
    train_data, train_label = np.array(train_data), np.array(train_label)
    num_features = train_data.shape[1]
    trans_function = get_trans_function(trans_func_shape)

    # initialize particles and Leader (the agent with the max fitness)
    particles = initialize(num_agents, num_features)
    fitness = np.zeros(num_agents)
    accuracy = np.zeros(num_agents)
    Leader_agent = np.zeros((1, num_features))
    Leader_fitness = float("-inf")
    Leader_accuracy = float("-inf")
    pool_size = 4
    omega = 0.9                 
    a2 = 1
    a1 = 2
    GP = 0.5

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

    # rank initial particles
    particles, fitness = sort_agents(particles, obj_function, data)

    # start timer
    start_time = time.time()

    # pool initialization
    eq_pool = np.zeros((pool_size+1, num_features))
    eq_fitness = np.zeros(pool_size)
    eq_fitness[:] = float("-inf")

    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no+1))
        print('================================================================================\n')     

        # replacements in the pool
        for i in range(num_agents):
            for j in range(pool_size):                 
                if fitness[i] <= eq_fitness[j]:
                    eq_fitness[j] = fitness[i].copy()
                    eq_pool[j, :] = particles[i, :].copy()
                    break
        

        best_particle = eq_pool[0,:]
                
        Cave = avg_concentration(eq_pool, pool_size, num_features)
        eq_pool[pool_size] = Cave.copy()

        t = (1 - (iter_no/max_iter)) ** (a2*iter_no/max_iter)
        
        for i in range(num_agents):
            
            # randomly choose one candidate from the equillibrium pool
            inx = np.random.randint(0,pool_size)
            Ceq = np.array(eq_pool[inx])

            lambda_vec = np.zeros(np.shape(Ceq))
            r_vec = np.zeros(np.shape(Ceq))
            for j in range(num_features):
                lambda_vec[j] = np.random.random()
                r_vec[j] = np.random.random()

            F_vec = np.zeros(np.shape(Ceq))
            for j in range(num_features):
                x = -1*lambda_vec[j]*t 
                x = np.exp(x) - 1
                x = a1 * sign_func(r_vec[j] - 0.5) * x

            r1, r2 = np.random.random(2)
            if r2 < GP:
                GCP = 0
            else:
                GCP = 0.5 * r1
            G0 = np.zeros(np.shape(Ceq))
            G = np.zeros(np.shape(Ceq))
            for j in range(num_features):
                G0[j] = GCP * (Ceq[j] - lambda_vec[j]*particles[i][j])
                G[j] = G0[j]*F_vec[j]
            
            # use transfer function to map continuous->binary
            for j in range(num_features):
                temp = Ceq[j] + (particles[i][j] - Ceq[j])*F_vec[j] + G[j]*(1 - F_vec[j])/lambda_vec[j]                
                temp = trans_function(temp)                
                if temp>np.random.random():
                    particles[i][j] = 1 - particles[i][j]
                else:
                    particles[i][j] = particles[i][j]          

        # update final information
        particles, fitness = sort_agents(particles, obj_function, data)
        display(particles, fitness, agent_name)
        
        # update Leader (best agent)
        if fitness[0] > Leader_fitness:
            Leader_agent = particles[0].copy()
            Leader_fitness = fitness[0].copy()

        convergence_curve['fitness'][iter_no] = Leader_fitness
        convergence_curve['feature_count'][iter_no] = int(np.sum(Leader_agent))

    # compute final accuracy
    Leader_agent, Leader_accuracy = sort_agents(Leader_agent, compute_accuracy, data)
    particles, accuracy = sort_agents(particles, compute_accuracy, data)

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
        plt.savefig('convergence_graph_'+ short_name + '.jpg')
    plt.show()

    # update attributes of solution
    solution.best_agent = Leader_agent
    solution.best_fitness = Leader_fitness
    solution.best_accuracy = Leader_accuracy
    solution.convergence_curve = convergence_curve
    solution.final_particles = particles
    solution.final_fitness = fitness
    solution.final_accuracy = accuracy
    solution.execution_time = exec_time

    return solution


def avg_concentration(eq_pool, pool_size, dimension): 
    # function to compute average concentration of the equilibrium pool   
    avg = np.sum(eq_pool[0:pool_size,:], axis=0)         
    avg = avg/pool_size
    return avg


def sign_func(x): 
    # signum function
    if x<0:
        return -1
    return 1

if __name__ == '__main__':

    iris = datasets.load_iris()
    EO(10, 20, iris.data, iris.target, save_conv_graph=True)
