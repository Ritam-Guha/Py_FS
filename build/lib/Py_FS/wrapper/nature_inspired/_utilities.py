import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
import matplotlib.pyplot as plt

class Solution():    
    #structure of the solution 
    def __init__(self):
        self.num_features = None
        self.num_agents = None
        self.max_iter = None
        self.obj_function = None
        self.execution_time = None
        self.convergence_curve = {}
        self.best_agent = None
        self.best_fitness = None
        self.best_accuracy = None
        self.final_population = None
        self.final_fitness = None
        self.final_accuracy = None


class Data():
    # structure of the training data
    def __init__(self):
        self.train_X = None
        self.train_Y = None
        self.val_X = None
        self.val_Y = None



def initialize(num_agents, num_features):
    # define min and max number of features
    min_features = int(0.3 * num_features)
    max_features = int(0.6 * num_features)

    # initialize the agents with zeros
    agents = np.zeros((num_agents, num_features))

    # select random features for each agent
    for agent_no in range(num_agents):

        # find random indices
        cur_count = np.random.randint(min_features, max_features)
        temp_vec = np.random.rand(1, num_features)
        temp_idx = np.argsort(temp_vec)[0][0:cur_count]

        # select the features with the ranom indices
        agents[agent_no][temp_idx] = 1   

    return agents



def sort_agents(agents, obj, data, fitness=None):
    # sort the agents according to fitness
    train_X, val_X, train_Y, val_Y = data.train_X, data.val_X, data.train_Y, data.val_Y
    (obj_function, weight_acc) = obj
   
    if fitness is None:
        # if there is only one agent
        if len(agents.shape) == 1:
            num_agents = 1
            fitness = obj_function(agents, train_X, val_X, train_Y, val_Y, weight_acc)
            return agents, fitness

        # for multiple agents
        else:
            num_agents = agents.shape[0]
            fitness = np.zeros(num_agents)
            for id, agent in enumerate(agents):
                fitness[id] = obj_function(agent, train_X, val_X, train_Y, val_Y, weight_acc)

    idx = np.argsort(-fitness)
    sorted_agents = agents[idx].copy()
    sorted_fitness = fitness[idx].copy()

    return sorted_agents, sorted_fitness



def display(agents, fitness, agent_name='Agent'):
    # display the population
    print('\nNumber of agents: {}'.format(agents.shape[0]))
    print('\n------------- Best Agent ---------------')
    print('Fitness: {}'.format(fitness[0]))
    print('Number of Features: {}'.format(int(np.sum(agents[0]))))
    print('----------------------------------------\n')

    for id, agent in enumerate(agents):
        print('{} {} - Fitness: {}, Number of Features: {}'.format(agent_name, id+1, fitness[id], int(np.sum(agent))))

    print('================================================================================\n')



def compute_accuracy(agent, train_X, test_X, train_Y, test_Y): 
    # compute classification accuracy of the given agents
    cols = np.flatnonzero(agent)     
    if(cols.shape[0] == 0):
        return 0    
    clf = KNN()

    train_data = train_X[:,cols]
    train_label = train_Y
    test_data = test_X[:,cols]
    test_label = test_Y

    clf.fit(train_data,train_label)
    acc = clf.score(test_data,test_label)

    return acc
        

def compute_fitness(agent, train_X, test_X, train_Y, test_Y, weight_acc=0.9):
    # compute a basic fitness measure
    if(weight_acc == None):
        weight_acc = 0.9

    weight_feat = 1 - weight_acc
    num_features = agent.shape[0]
    
    acc = compute_accuracy(agent, train_X, test_X, train_Y, test_Y)
    feat = (num_features - np.sum(agent))/num_features

    fitness = weight_acc * acc + weight_feat * feat
    return fitness


def Conv_plot(convergence_curve):
    # plot convergence curves
    num_iter = len(convergence_curve['fitness'])
    iters = np.arange(num_iter) + 1
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

    return fig, axes