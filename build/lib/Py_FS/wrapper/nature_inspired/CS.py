"""
Programmer: Trinav Bhattacharyya
Date of Development: 18/10/2020
This code has been developed according to the procedures mentioned in the following research article:
X.-S. Yang, S. Deb, “Cuckoo search via L´evy flights”, in: Proc. of
World Congress on Nature & Biologically Inspired Computing (NaBIC 2009),
December 2009, India. IEEE Publications, USA, pp. 210-214 (2009).

"""

import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets

from Py_FS.wrapper.nature_inspired._utilities import Solution, Data, initialize, sort_agents, display, compute_fitness, compute_accuracy
from Py_FS.wrapper.nature_inspired._transfer_functions import get_trans_function,sigmoid
# from _utilities import Solution, Data, initialize, sort_agents, display, compute_fitness, compute_accuracy
# from _transfer_functions import get_trans_function

def CS (num_nests, max_iter, train_data, train_label, obj_function=compute_fitness, trans_function_shape='s', save_conv_graph=False):
    
    # Cuckoo Search Algorithm
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

    short_name = 'CS'
    agent_name = 'Agent'
    num_features = train_data.shape[1]
    trans_function = get_trans_function(trans_function_shape)
    num_agents = num_nests

    # initializing cuckoo and host nests
    levy_flight = np.random.uniform(low=-2, high=2, size=(num_features))
    cuckoo = np.random.randint(low=0, high=2, size=(num_features))
    nest = initialize(num_nests, num_features)
    nest_fitness = np.zeros(num_nests)
    nest_accuracy = np.zeros(num_nests)
    cuckoo_fitness = float("-inf")
    Leader_agent = np.zeros((num_features))
    Leader_fitness = float("-inf")
    Leader_accuracy = float("-inf")
    p_a=0.25    # fraction of nests to be replaced   

    # initialize convergence curves
    convergence_curve = {}
    convergence_curve['fitness'] = np.zeros(max_iter)
    convergence_curve['feature_count'] = np.zeros(max_iter)

    data = Data()
    data.train_X, data.val_X, data.train_Y, data.val_Y = train_test_split(train_data, train_label, stratify=train_label, test_size=0.2)

    # create a solution object
    solution = Solution()
    solution.num_agents = num_agents
    solution.max_iter = max_iter
    solution.num_features = num_features
    solution.obj_function = obj_function

    # rank initial nests
    nest, nest_fitness = sort_agents(nest, obj_function, data)

    # start timer
    start_time = time.time()

    # main loop
    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no+1))
        print('================================================================================\n')

        # updating leader nest
        if nest_fitness[0] > Leader_fitness:
            Leader_agent = nest[0].copy()
            Leader_fitness = nest_fitness[0]

        # get new cuckoo
        levy_flight = get_cuckoo(levy_flight)
        for j in range(num_features):
            if trans_function(levy_flight[j]) > np.random.random():
                cuckoo[j]=1
            else:
                cuckoo[j]=0

        # check if a nest needs to be replaced
        j = np.random.randint(0,num_nests)
        if cuckoo_fitness > nest_fitness[j]:
            nest[j] = cuckoo.copy()
            nest_fitness[j] = cuckoo_fitness

        nest, nest_fitness = sort_agents(nest, obj_function, data)

        # eliminate worse nests and generate new ones
        nest = replace_worst(nest, p_a)

        nest, nest_fitness = sort_agents(nest, obj_function, data)

        # update final information
        display(nest, nest_fitness, agent_name)

        if nest_fitness[0]>Leader_fitness:
            Leader_agent = nest[0].copy()
            Leader_fitness = nest_fitness[0].copy()

        convergence_curve['fitness'][iter_no] = Leader_fitness
        convergence_curve['feature_count'][iter_no] = int(np.sum(Leader_agent))

    # compute final accuracy
    Leader_agent, Leader_accuracy = sort_agents(Leader_agent, compute_accuracy, data)
    nest, nest_accuracy = sort_agents(nest, compute_accuracy, data)

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
    solution.final_population = nest
    solution.final_fitness = nest_fitness
    solution.final_accuracy = nest_accuracy
    solution.execution_time = exec_time

    return solution

def get_cuckoo(agent, alpha=np.random.randint(-2,3)):
    features = len(agent)
    lamb = np.random.uniform(low=-3, high=-1, size=(features))
    levy = np.zeros((features))
    get_test_value = 1/(np.power((np.random.normal(0,1)),2))
    for j in range(features):
        levy[j] = np.power(get_test_value, lamb[j])   #Eq 2
    for j in range(features):
        agent[j]+=(alpha*levy[j])    #Eq 1

    return agent

def replace_worst(agent, fraction):
    pop, features = agent.shape
    for i in range(int((1-fraction)*pop), pop):
        agent[i] = np.random.randint(low=0, high=2, size=(features))
        if np.sum(agent[i])==0:
            agent[i][np.random.randint(0,features)]=1

    return agent

if __name__ == '__main__':
    iris = datasets.load_iris()
    CS(10, 20, iris.data, iris.target, save_conv_graph=True)