"""
Programmer: Trinav Bhattacharyya
Date of Development: 18/10/2020
This code has been developed according to the procedures mentioned in the following research article:
X.-S. Yang, S. Deb, “Cuckoo search via Levy flights”, in: Proc. of
World Congress on Nature & Biologically Inspired Computing (NaBIC 2009),
December 2009, India. IEEE Publications, USA, pp. 210-214 (2009).

"""
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets

from Py_FS.wrapper.nature_inspired._utilities import Solution, Data, initialize, sort_agents, display, compute_fitness, Conv_plot
from Py_FS.wrapper.nature_inspired._transfer_functions import get_trans_function

def CS (num_agents, max_iter, train_data, train_label, obj_function=compute_fitness, trans_function_shape='s', save_conv_graph=False):
    
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

    # setting up the objectives
    weight_acc = None
    if(obj_function==compute_fitness):
        weight_acc = float(input('Weight for the classification accuracy [0-1]: '))
    obj = (obj_function, weight_acc)
    compute_accuracy = (compute_fitness, 1) # compute_accuracy is just compute_fitness with accuracy weight as 1

    # initializing cuckoo and host nests
    levy_flight = np.random.uniform(low=-2, high=2, size=(num_features))
    cuckoo = np.random.randint(low=0, high=2, size=(num_features))
    nest = initialize(num_agents, num_features)
    fitness = np.zeros(num_agents)
    nest_accuracy = np.zeros(num_agents)
    cuckoo_fitness = float("-inf")
    Leader_agent = np.zeros((num_features))
    Leader_fitness = float("-inf")
    Leader_accuracy = float("-inf")
    p_a=0.25    # fraction of nests to be replaced   

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

    # rank initial nests
    nest, fitness = sort_agents(nest, obj, data)
    cuckoo,cuckoo_fitness = sort_agents(cuckoo,obj,data)

    # start timer
    start_time = time.time()

    # main loop
    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no+1))
        print('================================================================================\n')

        # updating leader nest
        if fitness[0] > Leader_fitness:
            Leader_agent = nest[0].copy()
            Leader_fitness = fitness[0]

        # get new cuckoo
        levy_flight = get_cuckoo(levy_flight)
        for j in range(num_features):
            if trans_function(levy_flight[j]) > np.random.random():
                cuckoo[j]=1
            else:
                cuckoo[j]=0
        cuckoo,cuckoo_fitness = sort_agents(cuckoo,obj,data)
        
        # check if a nest needs to be replaced
        j = np.random.randint(0,num_agents)
        if cuckoo_fitness > fitness[j]:
            nest[j] = cuckoo.copy()
            fitness[j] = cuckoo_fitness

        nest, fitness = sort_agents(nest, obj, data)

        # eliminate worse nests and generate new ones
        nest = replace_worst(nest, p_a)

        nest, fitness = sort_agents(nest, obj, data)

        # update final information
        display(nest, fitness, agent_name)

        if fitness[0]>Leader_fitness:
            Leader_agent = nest[0].copy()
            Leader_fitness = fitness[0].copy()

        convergence_curve['fitness'][iter_no] = np.mean(fitness)

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
    solution.final_population = nest
    solution.final_fitness = fitness
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
    data = datasets.load_digits()
    CS(20, 100, data.data, data.target, save_conv_graph=True)