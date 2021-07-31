from abc import abstractmethod
import copy
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets

from _utilities_test import Solution, Data, compute_accuracy, compute_fitness, initialize, sort_agents, display, call_counter


class Algorithm():
    def __init__(self,
                 num_agents,
                 max_iter,
                 train_data,
                 train_label,
                 test_data=None,
                 test_label=None,
                 val_size=30,
                 seed=0,
                 save_conv_graph=True,
                 max_evals=np.float("inf"),
                 algo_name=None,
                 agent_name=None):

        # essential user-defined variables
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.training_data = None
        self.train_data = train_data
        self.train_label = train_label

        # user-defined or default variables
        self.test_data = test_data
        self.test_label = test_label
        self.val_size = val_size
        self.weight_acc = None
        self.seed = seed
        self.save_conv_graph = save_conv_graph

        # algorithm internal variables
        self.population = None
        self.fitness = None
        self.accuracy = None
        self.Leader_agent = None
        self.Leader_fitness = float("-inf")
        self.Leader_accuracy = float("-inf")
        self.history = []
        self.cur_iter = 0
        self.max_evals = max_evals
        self.start_time = None
        self.end_time = None
        self.exec_time = None
        self.solution = None
        self.algo_name = algo_name
        self.agent_name = agent_name


    @abstractmethod
    def user_input(self):
        pass


    @abstractmethod
    def next(self):
        pass


    def initialize(self):
        # set the objective function
        self.val_size = float(input('Percentage of data for valdiation [0-100]: ') or 30)/100
        self.weight_acc = float(input('Weight for the classification accuracy [0-1]: ') or 0.9)
        self.obj_function = call_counter(compute_fitness(self.weight_acc))

        # start timer
        self.start_time = time.time()
        np.random.seed(self.seed)

        # data preparation
        self.training_data = Data()
        self.train_data, self.train_label = np.array(self.train_data), np.array(self.train_label)
        self.training_data.train_X, self.training_data.val_X, self.training_data.train_Y, self.training_data.val_Y = train_test_split(self.train_data, self.train_label, stratify=self.train_label, test_size=self.val_size)

        # create initial population
        num_features = self.train_data.shape[1]
        self.population = initialize(num_agents=self.num_agents, num_features=num_features)
        self.fitness = self.obj_function(self.population, self.training_data)
        self.population, self.fitness = sort_agents(agents=self.population, fitness=self.fitness)
        self.accuracy = compute_accuracy(agents=self.population, data=self.training_data)


    def check_end(self):
        # checks if the algorithm has met the end criterion
        return (self.cur_iter >= self.max_iter) or (self.obj_function.cur_evals >= self.max_evals)


    def save_details(self):
        # save some details of every generation
        cur_obj = {
            'population': self.population,
            'fitness': self.fitness,
            'accurcay': self.accuracy,
        }
        self.history.append(cur_obj)


    def display(self):
        # display the current generation details
        display(agents=self.population, fitness=self.fitness, agent_name=self.agent_name)


    def plot(self):
        # plot the convergence graph
        fig = plt.figure(figsize=(10, 8))
        avg_fitness = []
        for cur in self.history:
            avg_fitness.append(np.mean(cur['fitness']))

        plt.plot(np.arange(len(avg_fitness)), avg_fitness)
        plt.xlabel('Number of Generations')
        plt.ylabel('Average Fitness')
        plt.title('Convergence Curve')

        plt.show()

        return fig

    
    def post_processing(self):
        # post processing steps
        self.population, self.fitness = sort_agents(agents=self.population, fitness=self.fitness)
        self.accuracy = compute_accuracy(agents=self.population, data=self.training_data)
        
        if(self.fitness[0] > self.Leader_fitness):
            self.Leader_fitness = self.fitness[0]
            self.Leader_agent = self.population[0, :]
            self.Leader_accuracy = self.accuracy[0]


    def save_solution(self):
        # create a solution object
        self.solution = Solution()
        self.solution.num_agents = self.num_agents
        self.solution.max_iter = self.max_iter
        self.solution.num_features = self.train_data.shape[1]
        self.solution.obj_function = self.obj_function

        # update attributes of solution
        self.solution.best_agent = self.Leader_agent
        self.solution.best_fitness = self.Leader_fitness
        self.solution.best_accuracy = self.Leader_accuracy
        self.solution.final_population = self.population
        self.solution.final_fitness = self.fitness
        self.solution.final_accuracy = self.accuracy
        self.solution.execution_time = self.exec_time


    def run(self):
        # the main algorithm run
        print('\n************    Please enter the values of the following paramters or press newline for using default values    ************\n')
        self.user_input()   # take the user inputs
        self.initialize()   # initialize the algorithm
        print('\n*****************************************************    Thank You    ******************************************************\n')

        while(not self.check_end()):    # while the end criterion is not met
            self.next()                     # do one step of the algorithm
            self.post_processing()          # do the post processing steps
            self.display()                  # display the details of 1 iteration
            self.save_details()             # save the details

        self.end_time = time.time()     
        self.exec_time = self.end_time - self.start_time

        if self.test_data:          # if there is a test data, test the final solution on that 
            temp_data = Data()
            temp_data.train_X = self.train_data
            temp_data.train_Y = self.train_label
            temp_data.val_X = self.test_data
            temp_data.val_Y = self.test_label

            self.Leader_fitness = compute_fitness(self.Leader_agent, temp_data)
            self.Leader_accuracy = compute_accuracy(self.Leader_agent, temp_data)

        self.save_solution()
        fig = self.plot()

        if(self.save_conv_graph):
            fig.savefig('convergence_curve_' + self.algo_name + '.jpg')

        return self.solution

 





