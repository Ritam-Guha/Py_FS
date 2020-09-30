import numpy as np

class Solution():    
    #structure of the solution 
    def __init__(self):
        self.num_features = 0
        self.num_agents = 0
        self.max_iter = 0
        self.obj_function = None
        self.execution_time = 0
        self.convergence_curve = {}
        self.best_agent = None
        self.best_fitness = 0
        self.final_population = None
        self.final_fitness = None

class Data():
    # structure of the dataset
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

def sort_agents(agents, obj_function, data):
    # sort the agents according to fitness
    fitness = obj_function(agents, data)
    idx = np.argsort(-fitness)
    sorted_agents = agents[idx].copy()
    sorted_fitness = fitness[idx].copy()

    return sorted_agents, sorted_fitness

def display(agents, fitness):
    # display the population
    print('Number of agents: {}'.format(agents.shape[0]))
    print('\n------------- Best Agent -------------')
    print('Fitness: {}'.format(fitness[0]))
    print('Number of Features: {}'.format(int(np.sum(agents[0]))))
    print('----------------------------------------\n')

    for id, agent in enumerate(agents):
        print('Agent {} - Fitness: {}, Number of Features: {}'.format(id, fitness[id], int(np.sum(agent))))

    print('================================================================================\n')
