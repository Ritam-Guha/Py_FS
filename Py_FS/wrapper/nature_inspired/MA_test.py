"""

Programmer: Trinav Bhattacharyya
Date of Development: 13/10/2020
This code has been developed according to the procedures mentioned in the following research article:
Zervoudakis, K., Tsafarakis, S., A mayfly optimization algorithm, Computers &
Industrial Engineering (2020)

"""
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets

from Py_FS.datasets import get_dataset
from Py_FS.wrapper.nature_inspired.algorithm import Algorithm
from Py_FS.wrapper.nature_inspired._utilities_test import compute_accuracy, compute_fitness, initialize, sort_agents
from Py_FS.wrapper.nature_inspired._transfer_functions import get_trans_function

class MA(Algorithm):

    # Mayfly Algorithm
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of mayflies                                            #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #
    #   obj_function: the function to maximize while doing feature selection      #
    #   trans_function_shape: shape of the transfer function used                 #    
    #   save_conv_graph: boolean value for saving convergence graph               #
    #                                                                             #
    ###############################################################################
    
    def __init__(self,
                num_agents, 
                max_iter, 
                train_data, 
                train_label, 
                save_conv_graph=False, 
                seed=0):

        super().__init__(num_agents=num_agents,
                        max_iter=max_iter,
                        train_data=train_data,
                        train_label=train_label,
                        save_conv_graph=save_conv_graph,
                        seed=seed)

        self.algo_name='MA'
        self.agent_name='Mayfly'
        self.trans_function=None
        
    def user_input(self):
        # first set the default values for the attributes
        self.default_vals["prob_mut"] = 0.3
        self.default_vals["trans_function"] = 's'
        self.default_vals["a1"] = 1
        self.default_vals["a2"] = 1.5
        self.default_vals["d"] = 0.1
        self.default_vals["fl"] = 0.1
        self.default_vals["g"] = 0.8
        self.default_vals["beta"] = 2
        self.default_vals["delta"] = 0.9

        # accept the parameters as user inputs (if default_mode not set)
        if self.default_mode:
            self.set_default()
        else:
            self.algo_params['prob_mut'] = float(input(f'Probability of mutation [0-1] (default={self.default_vals["prob_mut"]}): ') or self.default_vals["prob_mut"])
            self.algo_params['trans_function'] = input(f'Shape of Transfer Function [s/v/u] (default={self.default_vals["trans_function"]}): ') or self.default_vals["trans_function"]
            self.algo_params['a1'] = float(input(f'Value of first attractive constant [1-3] (default={self.default_vals["a1"]}): ') or self.default_vals["a1"])
            self.algo_params['a2'] = float(input(f'Value of second attractive constant [1-3] (default={self.default_vals["a2"]}): ') or self.default_vals["a2"])
            self.algo_params['d'] = float(input(f'Value of nuptial dance coefficient [0-1] (default={self.default_vals["d"]}): ') or self.default_vals["d"])
            self.algo_params['fl'] = float(input(f'Value of random walk coefficient [0-1] (default={self.default_vals["fl"]}): ') or self.default_vals["fl"])
            self.algo_params['g'] = float(input(f'Value of gravity constant (0-1] (default={self.default_vals["g"]}): ') or self.default_vals["g"])
            self.algo_params['beta'] = float(input(f'Value of visibility coefficient [1-3] (default={self.default_vals["beta"]}): ') or self.default_vals["beta"])
            self.algo_params['delta'] = float(input(f'Value of delta [0-1] (default={self.default_vals["delta"]}): ') or self.default_vals["delta"])
            self.trans_function = get_trans_function(self.algo_params['trans_function'])
        
    def initialize(self):
        #call the base class function
        super().initialize()
        
        # initialize position and velocities of male and female mayflies' and Leader (the agent with the max fitness)
        self.male_pos = initialize(self.num_agents, self.num_features)
        self.female_pos = initialize(self.num_agents, self.num_features)
        self.male_vel = np.random.uniform(low = -1, high = 1, size = (self.num_agents, self.num_features))
        self.female_vel = np.random.uniform(low = -1, high = 1, size = (self.num_agents, self.num_features))
        self.male_fitness = np.zeros(self.num_agents)
        self.male_accuracy = np.zeros(self.num_agents)
        self.female_fitness = np.zeros(self.num_agents)
        self.male_personal_best = np.zeros((self.num_agents, self.num_features))
        self.male_personal_best_fit = np.zeros(self.num_agents)
        self.male_offspring = np.zeros((self.num_agents, self.num_features))
        self.female_offspring = np.zeros((self.num_agents, self.num_features))
        self.vmax_male = np.zeros(self.num_features)
        self.vmax_female = np.zeros(self.num_features)
        
        # rank initial population
        self.male_fitness = self.obj_function(self.male_pos,self.training_data)
        self.female_fitness = self.obj_function(self.female_pos,self.training_data)
        self.male_pos, self.male_fitness = sort_agents(self.male_pos,self.male_fitness)
        self.female_pos, self.female_fitness = sort_agents(self.female_pos,self.female_fitness)
        
        # create initial population
        self.population = self.male_pos
        self.fitness = self.male_fitness
        self.population, self.fitness = sort_agents(agents=self.population, fitness=self.fitness)
        self.accuracy = compute_accuracy(agents=self.population, data=self.training_data)
        self.Leader_agent, self.Leader_fitness = self.population[0], self.fitness[0]
        

    def update_max_velocity(self,male,female):
        size, length = male.shape
        agent1 = []
        agent2 = []
        r = np.random.normal(0,1 , size=(length))
        for j in range(length):
            r[j] *= 2
            agent1.append((male[0][j]-male[size-1][j])*r[j])
            agent2.append((female[0][j]-female[size-1][j])*r[j])
        
        return (agent1, agent2)

    def update_velocity(self,m_pos, f_pos, m_vel, f_vel, Leader_agent, pbest, a1, a2, d, fl, g, b, i, fitness1, fitness2):
        tot_features = m_pos.shape[0]
        agent1 = np.zeros((tot_features))
        agent2 = np.zeros((tot_features))
        tot_features = len(m_pos)
        if i==0:
            for j in range(tot_features):
                agent1[j] = m_vel[j] + d * np.random.uniform(-1,1)
        else:
            sum = 0    
            for j in range(tot_features):
                sum = sum+(m_pos[j]-Leader_agent[j])*(m_pos[j]-Leader_agent[j])
            rg = np.sqrt(sum)
            sum = 0
            for j in range(tot_features):
                sum = sum+(m_pos[j]-pbest[j])*(m_pos[j]-pbest[j])
            rp = np.sqrt(sum)
            for j in range(tot_features):
                agent1[j] = g*m_vel[j]+a1*np.exp(-b*rp*rp)*(pbest[j]-m_pos[j])+a2*np.exp(-b*rg*rg)*(Leader_agent[j]-m_pos[j])
        if fitness1 >= fitness2:
            sum = 0
            for j in range(tot_features):
                sum = sum+(m_pos[j]-f_pos[j])*(m_pos[j]-f_pos[j])
            rmf = np.sqrt(sum)
            agent2[j] = g*f_vel[j]+a2*np.exp(-b*rmf*rmf)*(m_pos[j]-f_pos[j])
        else:
            for j in range(tot_features):
                agent2[j] = g*f_vel[j]+fl*np.random.uniform(-1,1)
                
        return (agent1, agent2)

    def check_velocity_limits(self,m_vel, f_vel, vmax_m, vmax_f):
        tot_features = len(m_vel)
        for j in range(tot_features):
            m_vel[j] = np.minimum(m_vel[j], vmax_m[j])
            m_vel[j] = np.maximum(m_vel[j], -vmax_m[j])
            f_vel[j] = np.minimum(f_vel[j], vmax_f[j])
            f_vel[j] = np.maximum(f_vel[j], -vmax_f[j])
        
        return (m_vel, f_vel)

    def cross_mut(self,m_pos, f_pos,prob_mut):
        tot_features = len(m_pos)
        offspring1 = np.zeros((tot_features))
        offspring2 = np.zeros((tot_features))
        # partition defines the midpoint of the crossover
        partition = np.random.randint(tot_features//4, np.floor((3*tot_features//4)+1))

        # starting crossover
        for i in range(partition):
            offspring1[i] = m_pos[i]
            offspring2[i] = f_pos[i]

        for i in  range(partition, tot_features):
            offspring1[i] = f_pos[i]
            offspring2[i] = m_pos[i]
        # crossover ended


        # starting mutation
        if np.random.random() <= prob_mut:
            percent = 0.2
            numChange = int(tot_features*percent)
            pos = np.random.randint(0,tot_features-1,numChange)
            
            for j in pos:
                offspring1[j] = 1-offspring1[j]
            pos=np.random.randint(0,tot_features-1,numChange)
            for j in pos:
                offspring2[j] = 1-offspring2[j]

        # mutation ended
        
        if np.random.random() >= 0.5:
            return (offspring1, offspring2)
        else:
            return (offspring2, offspring1)


    def compare_and_replace(self,pos, off, fit):
        agents, features = pos.shape
        newfit = np.zeros((agents))
        temp_pos = np.zeros((agents, features))
        pos, fit = sort_agents(pos,fit)
        # finding fitnesses of offsprings
        newfit = self.obj_function(off,self.training_data)
        off, newfit = sort_agents(off,newfit)
        i=0
        j=0
        cnt=0
        # merging offsprings and parents and finding the next generation of mayflies
        while(cnt < agents):
            if fit[i] > newfit[j]:
                temp_pos[cnt] = pos[i].copy()
                i+=1
            else:
                temp_pos[cnt] = off[i].copy()
                j+=1
            cnt+=1
        return temp_pos

        #main loop
    def next(self):
    
        print('\n================================================================================')
        print('                          Iteration - {}'.format(self.cur_iter+1))
        print('================================================================================\n')
        
        #updating velocity limits
        self.vmax_male, self.vmax_female = self.update_max_velocity(self.male_pos,self.female_pos)
        
        self.male_personal_best_fit = self.obj_function(self.male_personal_best,self.training_data)
        
        for agent in range(self.num_agents):
            
            if self.male_fitness[agent] > self.male_personal_best_fit[agent]:
                self.male_personal_best[agent] = self.male_pos[agent]

            #update velocities of male and female mayflies
            self.male_vel[agent], self.female_vel[agent] = self.update_velocity(self.male_pos[agent], self.female_pos[agent], self.male_vel[agent], self.female_vel[agent], self.Leader_agent, self.male_personal_best[agent], self.algo_params['a1'], self.algo_params['a2'], self.algo_params['d'], self.algo_params['fl'], self.algo_params['g'], self.algo_params['beta'], agent, self.male_fitness[agent], self.female_fitness[agent])
            
            #check boundary condition of velocities of male and female mayflies
            self.male_vel[agent], self.female_vel[agent] = self.check_velocity_limits(self.male_vel[agent], self.female_vel[agent], self.vmax_male, self.vmax_female)
            
            #applying transfer functions to update positions of male and female mayflies
            #the updation is done based on their respective velocity values
            for j in range(self.num_features):
                trans_value = self.trans_function(self.male_vel[agent][j])
                if trans_value > np.random.normal(0,1):
                    self.male_pos[agent][j]=1
                else:
                    self.male_pos[agent][j]=0

                trans_value = self.trans_function(self.female_vel[agent][j])
                if trans_value > np.random.random():
                    self.female_pos[agent][j]=1
                else:
                    self.female_pos[agent][j]=0
                    
        #sorting 
        self.male_pos, self.male_fitness = sort_agents(self.male_pos, self.male_fitness)
        self.female_pos, self.female_fitness = sort_agents(self.female_pos, self.female_fitness)
        
        for agent in range(self.num_agents):
            
            #generation of offsprings by crossover and mutation between male and female parent mayflies
            self.male_offspring[agent], self.female_offspring[agent] = self.cross_mut(self.male_pos[agent], self.female_pos[agent],self.algo_params['prob_mut'])
            
        #comparing parents and offsprings and replacing parents wherever necessary
        self.male_pos = self.compare_and_replace(self.male_pos, self.male_offspring, self.male_fitness)
        self.female_pos = self.compare_and_replace(self.female_pos, self.female_offspring, self.female_fitness)
        
        #updating fitness values
        self.male_pos, self.male_fitness = sort_agents(self.male_pos, self.male_fitness)
        self.female_pos, self.female_fitness = sort_agents(self.female_pos, self.female_fitness)
        
        #updating values of nuptial dance
        self.algo_params['d'] = self.algo_params['d'] * self.algo_params['delta']
        self.algo_params['fl'] = self.algo_params['fl'] * self.algo_params['delta']
        
        self.population = self.male_pos
        self.fitness = self.male_fitness
        
        self.cur_iter += 1


if __name__ == '__main__':
    data = datasets.load_digits()
    algo = MA(num_agents=20, max_iter=30, train_data=data.data, train_label=data.target, save_conv_graph=True)
    algo.run()