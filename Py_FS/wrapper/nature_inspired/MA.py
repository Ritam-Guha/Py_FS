"""

Programmer: Trinav Bhattacharyya
Date of Development: 13/10/2020
This code has been developed according to the procedures mentioned in the following research article:
Zervoudakis, K., Tsafarakis, S., A mayfly optimization algorithm, Computers &
Industrial Engineering (2020)

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

def MA(num_agents, max_iter, train_data, train_label, obj_function=compute_fitness, trans_function_shape='s',  prob_mut=0.2,  save_conv_graph=False):
    
    # Mayfly Algorithm
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of mayflies                                            #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #                
    #   obj_function: the function to maximize while doing feature selection      #
    #   prob_mut: probability of mutation                                         #
    #   trans_function_shape: shape of the transfer function used                 #
    #   save_conv_graph: boolean value for saving convergence graph               #
    #                                                                             #
    ###############################################################################

    short_name = 'MA'
    agent_name = 'Mayfly'
    train_data, train_label = np.array(train_data), np.array(train_label)
    num_features = train_data.shape[1]
    trans_function = get_trans_function(trans_function_shape)
    
    # control parameters
    a1 = 1
    a2 = 1.5
    d = 0.1
    fl = 0.1
    g = 0.8
    beta = 2
    delta = 0.9
    
    # initialize position and velocities of male and female mayflies' and Leader (the agent with the max fitness)
    male_pos = initialize(num_agents, num_features)
    female_pos = initialize(num_agents, num_features)
    male_vel = np.random.uniform(low = -1, high = -1, size = (num_agents, num_features))
    female_vel = np.random.uniform(low = -1, high = -1, size = (num_agents, num_features))
    male_fitness = np.zeros((num_agents))
    male_accuracy = np.zeros(num_agents)
    female_fitness = np.zeros((num_agents))
    Leader_agent = np.zeros((num_features))
    Leader_fitness = float("-inf")
    Leader_accuracy = float("-inf")
    male_personal_best = np.zeros((num_agents, num_features))
    male_offspring = np.zeros((num_agents, num_features))
    female_offspring = np.zeros((num_agents, num_features))
    vmax_male = np.zeros((num_features))
    vmax_female = np.zeros((num_features))
    
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
    
    # rank initial population
    male_pos, male_fitness = sort_agents(male_pos, obj_function, data)
    female_pos, female_fitness = sort_agents(female_pos, obj_function, data)
    
    # start timer
    start_time = time.time()
    
    # main loop
    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no+1))
        print('================================================================================\n')
        
        #updating velocity limits
        vmax_male, vmax_female = update_max_velocity(male_pos, female_pos)
        
        for agent in range(num_agents):
            
            #updating Leader fitness and personal best fitnesses
            if male_fitness[agent] > Leader_fitness:
                Leader_fitness = male_fitness[agent]
                Leader_agent = male_pos[agent]
            
            if male_fitness[agent] > obj_function(male_personal_best[agent], data.train_X, data.val_X, data.train_Y, data.val_Y):
                male_personal_best[agent] = male_pos[agent]

            #update velocities of male and female mayflies
            male_vel[agent], female_vel[agent] = update_velocity(male_pos[agent], female_pos[agent], male_vel[agent], female_vel[agent], Leader_agent, male_personal_best[agent], a1, a2, d, fl, g, beta, agent, data, obj_function)
            
            #check boundary condition of velocities of male and female mayflies
            male_vel[agent], female_vel[agent] = check_velocity_limits(male_vel[agent], female_vel[agent], vmax_male, vmax_female)
            
            #applying transfer functions to update positions of male and female mayflies
            #the updation is done based on their respective velocity values
            for j in range(num_features):
                trans_value = trans_function(male_vel[agent][j])
                if trans_value > np.random.normal(0,1):
                    male_pos[agent][j]=1
                else:
                    male_pos[agent][j]=0

                trans_value = trans_function(female_vel[agent][j])
                if trans_value > np.random.random():
                    female_pos[agent][j]=1
                else:
                    female_pos[agent][j]=0
        
        #sorting 
        male_pos, male_fitness = sort_agents(male_pos, obj_function, data)
        female_pos, female_fitness = sort_agents(female_pos, obj_function, data)
        
        for agent in range(num_agents):
            
            #generation of offsprings by crossover and mutation between male and female parent mayflies
            male_offspring[agent], female_offspring[agent] = cross_mut(male_pos[agent], female_pos[agent])
            
        #comparing parents and offsprings and replacing parents wherever necessary
        male_pos = compare_and_replace(male_pos, male_offspring, male_fitness, data, obj_function)
        female_pos = compare_and_replace(female_pos, female_offspring, female_fitness, data, obj_function)
        
        #updating fitness values
        male_pos, male_fitness = sort_agents(male_pos, obj_function, data)
        female_pos, female_fitness = sort_agents(female_pos, obj_function, data)
        
        #updating values of nuptial dance
        d = d * delta
        fl = fl * delta
        
        #update final information
        display(male_pos, male_fitness, agent_name)
        if(male_fitness[0] > Leader_fitness):
            Leader_agent = male_pos[0].copy()
            Leader_fitness = male_fitness[0].copy()
        convergence_curve['fitness'][iter_no] = Leader_fitness
        convergence_curve['feature_count'][iter_no] = int(np.sum(Leader_agent))
    
    # compute final accuracy
    Leader_agent, Leader_accuracy = sort_agents(Leader_agent, compute_accuracy, data)
    male_pos, male_accuracy = sort_agents(male_pos, compute_accuracy, data)

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
    solution.final_population = male_pos
    solution.final_fitness = male_fitness
    solution.final_accuracy = male_accuracy
    solution.execution_time = exec_time

    return solution


def update_max_velocity(male, female):
    size, length = male.shape
    agent1 = []
    agent2 = []
    r = np.random.normal(0,1 , size=(length))
    for j in range(length):
        r[j] *= 2
        agent1.append((male[0][j]-male[size-1][j])*r[j])
        agent2.append((female[0][j]-female[size-1][j])*r[j])
    
    return (agent1, agent2)

def update_velocity(m_pos, f_pos, m_vel, f_vel, Leader_agent, pbest, a1, a2, d, fl, g, b, i, data, obj_function):
    tot_features = m_pos.shape[0]
    agent1 = np.zeros((tot_features))
    agent2 = np.zeros((tot_features))
    tot_features = len(m_pos)
    if i==0:
        for j in range(tot_features):
            agent1[j] = m_vel[j]+d*np.random.uniform(-1,1)
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
    if obj_function(m_pos, data.train_X, data.val_X, data.train_Y, data.val_Y) >= obj_function(f_pos, data.train_X, data.val_X, data.train_Y, data.val_Y):
        sum = 0
        for j in range(tot_features):
            sum = sum+(m_pos[j]-f_pos[j])*(m_pos[j]-f_pos[j])
        rmf = np.sqrt(sum)
        agent2[j] = g*f_vel[j]+a2*np.exp(-b*rmf*rmf)*(m_pos[j]-f_pos[j])
    else:
        for j in range(tot_features):
            agent2[j] = g*f_vel[j]+fl*np.random.uniform(-1,1)
            
    return (agent1, agent2)

def check_velocity_limits(m_vel, f_vel, vmax_m, vmax_f):
    tot_features = len(m_vel)
    for j in range(tot_features):
        m_vel[j] = np.minimum(m_vel[j], vmax_m[j])
        m_vel[j] = np.maximum(m_vel[j], -vmax_m[j])
        f_vel[j] = np.minimum(f_vel[j], vmax_f[j])
        f_vel[j] = np.maximum(f_vel[j], -vmax_f[j])
    
    return (m_vel, f_vel)

def cross_mut(m_pos, f_pos):
    tot_features = len(m_pos)
    offspring1 = np.zeros((tot_features))
    offspring2 = np.zeros((tot_features))
    #partition defines the midpoint of the crossover
    partition=np.random.randint(tot_features//4,np.floor((3*tot_features//4)+1))
    #starting crossover
    for i in range(partition):
        offspring1 = m_pos[i]
        offspring2 = f_pos[i]
    for i in  range(partition,tot_features):
        offspring1 = f_pos[i]
        offspring2 = m_pos[i]
    #crossover ended
    
    #starting mutation
    percent=0.2
    numChange=int(tot_features*percent)
    pos=np.random.randint(0,tot_features-1,numChange)
    for j in pos:
        offspring1[j] = 1-offspring1[j]
    pos=np.random.randint(0,tot_features-1,numChange)
    for j in pos:
        offspring2[j] = 1-offspring2[j]
    #mutation ended
    
    if np.random.random() >= 0.5:
        return (offspring1, offspring2)
    else:
        return (offspring2, offspring1)

def compare_and_replace(pos, off, fit, data, obj_function):
    agents, features = pos.shape
    newfit = np.zeros((agents))
    temp_pos = np.zeros((agents, features))
    pos, fit = sort_agents(pos, obj_function, data)
    #finding fitnesses of offsprings
    off, newfit = sort_agents(off, obj_function, data)
    i=0
    j=0
    cnt=0
    #merging offsprings and parents and finding the next generation of mayflies
    while(cnt < agents):
        if fit[i] > newfit[j]:
            temp_pos[cnt] = pos[i].copy()
            i+=1
        else:
            temp_pos[cnt] = off[i].copy()
            j+=1
        cnt+=1
    return temp_pos

def trans_function1(velocity):
    t = abs(velocity/(np.sqrt(1+velocity*velocity)))
    return t


if __name__ == '__main__':
    iris = datasets.load_iris()
    MA(10, 20, iris.data, iris.target, save_conv_graph=True)