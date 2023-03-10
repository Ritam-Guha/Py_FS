import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets


class Solution():    
    #structure of the solution 
    def __init__(self):
        self.num_features = None
        self.num_agents = None
        self.max_iter = None
        self.obj_function = None
        self.execution_time = None
        self.best_agent = None
        self.best_fitness = None
        self.best_accuracy = None
        self.final_population = None
        self.final_fitness = None
        self.final_accuracy = None
        self.history = None


class Data():
    # structure of the training data
    def __init__(self):
        self.train_X = None
        self.train_Y = None
        self.val_X = None
        self.val_Y = None



def initialize(num_agents, num_features):  
    # create the population
  
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



def sort_agents(agents, fitness):
    # sort the agents according to fitness
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



def compute_accuracy(agents, data): 
    # compute classification accuracy of the given agents
    if len(agents.shape) == 1:
        # handles with zero-rank arrays
        agents = np.array([agents])

    (num_agents, num_features) = agents.shape
    acc = np.zeros(num_agents)   
    clf = KNN()
    
    for (i, agent) in enumerate(agents):
        cols = np.flatnonzero(agent) 

        if cols.size > 0:    
            train_data = data.train_X[:,cols]
            train_label = data.train_Y
            test_data = data.val_X[:,cols]
            test_label = data.val_Y

            clf.fit(train_data,train_label)
            acc[i] = clf.score(test_data,test_label)

    return acc
        

def compute_fitness(weight_acc):
    def _compute_fitness(agents, data):
        # compute a basic fitness measure
        
        if len(agents.shape) == 1:
        # handles with zero-rank arrays
            agents = np.array([agents])

        weight_feat = 1 - weight_acc
        (num_agents, num_features) = agents.shape
        fitness = np.zeros(num_agents)

        acc = compute_accuracy(agents, data)

        for (i, agent) in enumerate(agents):
            if np.sum(agents[i])!=0:
                feat = (num_features - np.sum(agents[i]))/num_features
                fitness[i] = weight_acc * acc[i] + weight_feat * feat

        return fitness
    
    return _compute_fitness



def call_counter(func):
    # meta function to count the number of calls to another function
	def helper(*args, **kwargs):
		helper.cur_evals += 1	
		func_val = func(*args, **kwargs)
		return func_val

	helper.cur_evals = 0
	helper.__name__= func.__name__
	return helper

