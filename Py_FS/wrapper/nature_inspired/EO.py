"""

Programmer: Ritam Guha
Date of Development: 9/10/2020
This code has been developed according to the procedures mentioned in the following research article:
"Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. 
Advances in engineering software, 69, 46-61."

"""

import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets

# from Py_FS.wrapper.nature_inspired._utilities import Solution, Data, initialize, sort_agents, display, compute_accuracy
# from Py_FS.wrapper.nature_inspired._transfer_functions import get_trans_function
from _utilities import Solution, Data, initialize, sort_agents, display, compute_accuracy
from _transfer_functions import get_trans_function


def EO(num_agents, max_iter, train_data, train_label, obj_function=compute_accuracy, trans_func_shape='s', save_conv_graph=False):
    
    # Equilibrium Optimizer
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of chromosomes                                         #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #                
    #   obj_function: the function to maximize while doing feature selection      #
    #   trans_func_shape: shape of the transfer function used                     #
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
    Leader_agent = np.zeros((1, num_features))
    Leader_fitness = float("-inf")
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

    # create a Solution object
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
    eq_fitness = np.zeros((1, pool_size))
    eq_fitness[0, :] = 100

    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no+1))
        print('================================================================================\n')

        pop_new = np.zeros((num_agents, num_features))        

        # replacements in the pool
        for i in range(num_agents):
            for j in range(pool_size):                 
                if fitness[0, i] <= eq_fitness[0, j]:
                    eq_fitness[0, j] = fitness[0, i].copy()
                    eq_pool[j, :] = particles[i, :].copy()
                    break
        

        best_particle = eq_pool[0,:]
                
        Cave = avg_concentration(eq_pool,pool_size,num_features)
        eq_pool[pool_size] = Cave.copy()

        t = (1 - (iter_no/max_iter)) ** (a2*iter_no/max_iter)
        
        for i in range(num_agents):
            
            # randomly choose one candidate from the equillibrium pool
            random.seed(time.time() + 100 + 0.02*i)
            inx = random.randint(0,pool_size)
            Ceq = np.array(eq_pool[inx])

            lambda_vec = np.zeros(np.shape(Ceq))
            r_vec = np.zeros(np.shape(Ceq))
            for j in range(num_features):
                random.seed(time.time() + 1.1)
                lambda_vec[j] = random.random()
                random.seed(time.time() + 10.01)
                r_vec[j] = random.random()

            F_vec = np.zeros(np.shape(Ceq))
            for j in range(num_features):
                x = -1*lambda_vec[j]*t 
                x = math.exp(x) - 1
                x = a1 * sign_func(r_vec[j] - 0.5) * x

            random.seed(time.time() + 200)
            r1 = random.random()
            random.seed(time.time() + 20)
            r2 = random.random()
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
                    pop_new[i][j] = 1 - particles[i][j]
                else:
                    pop_new[i][j] = particles[i][j]          

        # update final information
        particles, fitness = sort_agents(pop_new, obj_function, data)
        display(particles, fitness, agent_name)
        
        # update Leader (best agent)
        if fitness[0] > Leader_fitness:
            Leader_agent = particles[0].copy()
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

    if(save_conv_graph):
        plt.savefig('convergence_graph_'+ short_name + '.jpg')
    plt.show()

    # update attributes of solution
    solution.best_agent = Leader_agent
    solution.best_fitness = Leader_fitness
    solution.convergence_curve = convergence_curve
    solution.final_particles = particles
    solution.final_fitness = fitness
    solution.execution_time = exec_time

    return solution


if __name__ == '__main__':

    iris = datasets.load_iris()
    EO(10, 20, iris.data, iris.target, compute_accuracy, save_conv_graph=True)
