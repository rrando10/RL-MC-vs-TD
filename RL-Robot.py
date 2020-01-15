"""
ECE517: Reinforcement Learning in Artificial Intelligence 
Project 2: Monte-Carlo and Q-Learning
November 11, 2019
                                                            
Writen By:                                                  
John Geissberger Jr. and Ronald Randolph
"""

import numpy as np
from random import seed
import copy
import random
import math
import sys

import matplotlib.pyplot as plt
import numpy as np

class robot: 

    def __init__(self,d): 
        self.d = d
        self.pos = np.zeros([1,2],dtype = int)
        self.actions = ['N','S','E','W']
        self.bomb_pos = np.zeros([1,2])
        self.sequence = []
        self.random_policy = [0.25,0.25,0.25,0.25] 
        self.time_step = 0
        self.rewards = [] 
 
    
    def initialize(self): 
        """
        Initialize Robot's starting position and other relevant values 
        """
        
        self.time_step = 0
        self.rewards = []
        self.sequence = []

        x = random.randint(1,self.d-1)
        y = random.randint(1,self.d-1)
        self.pos[0,0] = x
        self.pos[0,1] = y
        self.sequence.append((self.time_step,'starting_point',self.pos))
        self.time_step += 1

    
    def choose_action(self):
        """ 
        choose a random action
        """  
        action = self.actions[random.randint(0,3)] 
        return action 

    
    def next_position(self,action): 
        """ 
        This function takes an action and calculates
        the next position of the robot
        """
        
        #First, extract current pos:
        current = np.zeros((1,2))
        current[0,0] = copy.deepcopy(self.pos[0,0])
        current[0,1] = copy.deepcopy(self.pos[0,1])
        
        #Logic to ensure that when the robot goes in the water it stays where it is.
        if(action == 'N'):
            if(self.pos[0,0] - 1 == 0):
                self.pos = self.pos
            else: 
                self.pos[0,0] = self.pos[0,0] -1

        elif(action == 'E'):
            if(self.pos[0,1] + 1 == self.d):
                self.pos = self.pos
            else: 
                self.pos[0,1] = self.pos[0,1]+1
    
        elif(action == 'W'):
            if(self.pos[0,1] - 1 == 0):
                self.pos = self.pos
            else: 
                self.pos[0,1] = self.pos[0,1] -1

        elif(action == 'S'):
            if(self.pos[0,0] + 1 == self.d):
                self.pos = self.pos
            else: 
                self.pos[0,0] = self.pos[0,0]+1

        #store robot coordinates and update timestamp
        self.sequence.append((self.time_step,action,current))
        self.time_step += 1
        return



class environment: 
      
    def __init__(self,d): 
        self.d = d
        self.robot_pos = np.zeros((1,2))
        self.bomb_pos = np.zeros((1,2),dtype = int)
        self.terminate_flag = 0
        self.bomb_sequence = [] 
        self.time_step = 0
        self.qa = np.zeros((d,d,d,d,4))
        self.qa_check = np.full((d,d,d,d,4),-1)
        self.policy = np.zeros((d,d,d,d,4))

    
    def initialize(self): 
        """
        Initialize Bomb's starting position and other relevant Env values 
        """
        
        self.terminate_flag = 0
        self.bomb_sequence = [] 
        self.time_step = 0 

        x = random.randint(1,self.d-1)
        y = random.randint(1,self.d-1)
            
        self.bomb_pos[0,0] = x
        self.bomb_pos[0,1] = y
        
        current = np.zeros((1,2))
        current[0,0] = x
        current[0,1] = y
        self.bomb_sequence.append((self.time_step,current))
        self.time_step += 1

    
    def push_bomb(self,robot):
          
        """ 
        This function updates the bomb's coordinates 
        then returns TRUE if the bomb has been pushed 
        and FALSE otherwise. Also, returns the bomb's
        current coordinates
        """

        #get the last action: 
        t = robot.time_step - 1
        tuple = robot.sequence[t]
        last_action = tuple[1]
        
        #create flag for when the bomb is pushed 
        bomb_pushed = False          

        #extract current bomb position
        curr_bomb_pos = np.zeros((1,2))
        curr_bomb_pos[0,0] = copy.deepcopy(self.bomb_pos[0,0])
        curr_bomb_pos[0,1] = copy.deepcopy(self.bomb_pos[0,1])

        #check for a "push" event
        if((robot.pos == self.bomb_pos).all()):
            
            if(last_action == 'N'):
                self.bomb_pos[0,0] = self.bomb_pos[0,0]-1

            elif(last_action == 'E'): 
                self.bomb_pos[0,1] = self.bomb_pos[0,1]+1
                
            elif(last_action == 'W'):
                self.bomb_pos[0,1] = self.bomb_pos[0,1]-1

            elif(last_action == 'S'):
                self.bomb_pos[0,0] = self.bomb_pos[0,0]+1
            
            #throw flag
            bomb_pushed = True

        #store updated bomb coords
        self.bomb_sequence.append((self.time_step,curr_bomb_pos))
        self.time_step += 1

        #if the bomb has been pushed - return True
        if(bomb_pushed):
            return True, curr_bomb_pos
        else:
            return False, curr_bomb_pos 

    
    def get_reward(self,pushed,last_pos): 
        """ 
        Calculate rewards based off the bomb's movement and the
        one of the following reward schemes: 
    
        0. Every step gives a -1 reward
        1. Every step gives a -1 reward, moving the bomb from the
            center gives a +1 reward, and moving it out gives a 
            +10 reward
        """ 
        scheme = 1 
        
        if(scheme == 1 and pushed == True):
            #First check is to determine if bomb has been pushed into water
            if(self.bomb_pos[0,0] == self.d or self.bomb_pos[0,0] == 0):
                #bomb in water
                self.terminate_flag = 1
                return -1
            #Check for bomb in water by column
            elif(self.bomb_pos[0,1] == self.d or self.bomb_pos[0,1] == 0):
                #bomb in water
                self.terminate_flag = 1
                return -1

            #get distance from center
            center = np.zeros((1,2))
            center[0,0] = center[0,1] = (self.d / 2.0)
    
            #Calculate distances of bomb from center. 
            last_dist_fc = np.linalg.norm(last_pos - center)
            current_dist_fc = np.linalg.norm(self.bomb_pos - center)
        
            #determine if bomb was pushed further away from the center. 
            if(current_dist_fc > last_dist_fc):  
                return -1
            else: 
                return -1
        
        #scheme 0: -1 for all actions
        else:
            return -1

    
    def first_visit_check(self,robot): 
        """ 
        This helper fucntion performs the first-vist check
        for the MC method. It builds an array of values that
        correspond to the earliest timestamp that a state/action
        pair was visited
        """
        
        #init the first-visit check matrix
        self.qa_check = np.full((self.d,self.d,self.d,self.d,self.d),-1)
        
        for i in range(1,len(self.bomb_sequence)):
            
            #get robot & bpmb coords
            bomb_tuple = self.bomb_sequence[i]
            robot_tuple = robot.sequence[i]
      
            #seperate cords into x y for each entity
            bomb_r = bomb_tuple[1].astype(int)[0][0]
            bomb_c = bomb_tuple[1].astype(int)[0][1]
            robot_r = robot_tuple[2].astype(int)[0][0]
            robot_c = robot_tuple[2].astype(int)[0][1]

            #action taken
            action_taken = robot_tuple[1]
            act_in = self.dir_to_index(action_taken)
 
            #Skip if the bomb is in the water
            if(self.check_out_bounds(bomb_r,bomb_c) == -1):
                continue
            
            else: 
                if( self.qa_check[robot_r,robot_c,bomb_r,bomb_c,act_in] == -1):
                    self.qa_check[robot_r,robot_c,bomb_r,bomb_c,act_in] = i

      #update description
    
    
    def mc_update(self,robot,epsilon,constant): 
        """ 
        This is a helper function to the MC method. This 
        function performs the update rule on all first visits 
        to state/action pairs. It then updates the e-greedy 
        policy and returns
        """

        #get earliest occurences of each state
        self.first_visit_check(robot)
    
        T = len(self.bomb_sequence)  
        G = 0

        for i in range(1,T):
            
            #move backwards from end of episode
            index = T-i
            G = G + robot.rewards[index-1][1]
            
            #get bomb/robot coords
            bomb_tuple = self.bomb_sequence[index]
            robot_tuple = robot.sequence[index]

            #seperate cords into x y for each entity
            bomb_r = bomb_tuple[1].astype(int)[0][0]
            bomb_c = bomb_tuple[1].astype(int)[0][1]
            robot_r = robot_tuple[2].astype(int)[0][0]
            robot_c = robot_tuple[2].astype(int)[0][1]

            #action taken
            action_taken = robot_tuple[1]
            act_in = self.dir_to_index(action_taken)

            #skip if bomb is out of bounds
            if(self.check_out_bounds(bomb_r,bomb_c) == -1):
                continue

            else:
                
                #if index is on a state/action pair's earliest occurence, update its Q-value 
                if(index == self.qa_check[robot_r,robot_c,bomb_r,bomb_c,act_in]):
                    self.qa[robot_r,robot_c,bomb_r,bomb_c,act_in] += constant*(G - self.qa[robot_r,robot_c,bomb_r,bomb_c,act_in])

                    #update e-greedy policy
                    self.update_policy(robot_r,robot_c,bomb_r,bomb_c,epsilon)

    
    def Q_Learning(self,robot,epsilon,constant):

        R3 = robot
        Env = self
        retval = 0.0

        #runs a episode until a TS is reached or 1000 steps
        while(Env.terminate_flag != 1):
           
            #grab current state information
            cbomb_r = self.bomb_pos[0][0]
            cbomb_c = self.bomb_pos[0][1]
            crobot_r = robot.pos[0][0]
            crobot_c = robot.pos[0][1]

            #take action
            a = Env.next_action(robot)
            robot.next_position(a)

            #reward and new state
            pushed, bomb_coords = Env.push_bomb(R3)
            reward = Env.get_reward(pushed, bomb_coords)

            #add reward to total return
            R3.rewards.append((R3.time_step-1, reward))
            retval += reward

            #grab next state information and reward
            nbomb_r = self.bomb_pos[0][0]
            nbomb_c = self.bomb_pos[0][1]
            nrobot_r = robot.pos[0][0]
            nrobot_c = robot.pos[0][1]

            #get corresponding index to action
            action = self.dir_to_index(a)
            
            #if next state is terminal, maxQ is 0
            if(self.check_out_bounds(nbomb_r, nbomb_c) == -1):
                maxQ = 0.0

            #get next state's maximum Q-val
            else:
                maxQ = np.amax(Env.qa[nrobot_r, nrobot_c, nbomb_r, nbomb_c])

            #get current state's Q-val
            tmpQ = Env.qa[crobot_r,crobot_c,cbomb_r,cbomb_c,action]
            
            #Q-Learning Update Rule
            Env.qa[crobot_r,crobot_c,cbomb_r,cbomb_c,action] = tmpQ + (constant * (reward + maxQ - tmpQ))

            #update e-greedy policy
            self.update_policy(crobot_r,crobot_c,cbomb_r,cbomb_c,epsilon)

            #step limit check
            if(Env.time_step-1 == 1000):
                Env.terminate_flag = 1

        #return episodic return and #steps
        return retval, Env.time_step-1

    
    def check_out_bounds(self,x,y):
        """ 
        Helper function designed to check if bomb is out of bounds 
        """
        
        if(x == 0 or y == 0):
            return -1
        elif(x == self.d or y == self.d):
            return -1
        else:
            return 1

    
    def update_policy(self,robot_r,robot_c,bomb_r,bomb_c,epsilon):
        """
        This helper function takes the current state/action pair and
        updates an e-greedy policy by Q(s,a)
        """
        
        #get the action with the highest Q-value
        A_star = np.argmax(self.qa[robot_r,robot_c,bomb_r,bomb_c])
                
        #update e-greedy policy by Q-value
        for k in range(0,4): 
            if(k == A_star): 
                self.policy[robot_r,robot_c,bomb_r,bomb_c,k] = (1.0 - epsilon) + (epsilon/4)
            else: 
                self.policy[robot_r,robot_c,bomb_r,bomb_c,k] = (epsilon/4)
    
    
    def dir_to_index(self,a):
        """
        This helper function takes a direction action
        (N,S,E,W) and converts it to (0,1,2,3),repectively
        """

        if(a == 'N'):
            return 0
        if(a == 'S'):
            return 1
        if(a == 'E'):
            return 2
        if(a == 'W'):
            return 3

    
    def print_board(self,robot): 
        """
        Helper function to print_episode()
        This function takes the current location
        of the bomb and robot and prints a board
        """

        #strings for concatenation
        topnbot = "~~"
        emptyr = "~|"

        #coords for bomb and robot
        robot_row = robot.pos[0,0]
        robot_col = robot.pos[0,1]
        bomb_row = self.bomb_pos[0,0]
        bomb_col = self.bomb_pos[0,1]

        #create top/bottom and empty rows string
        for i in range(1,self.d):
            topnbot += "~~"
            emptyr += "-|"
        topnbot += "~"
        emptyr += "~"
            
        #print board with entity locations
        print(topnbot)
        for r in range(1,self.d):
            if((r == robot_row)or(r == bomb_row)):
                curr_row = "~|"
                for c in range(1,self.d):
                    if((r == robot_row) and (c == robot_col)):
                        curr_row += "R|"
                    elif((r == bomb_row) and (c == bomb_col)):
                        curr_row += "B|"
                    else:
                        curr_row += "-|"
                curr_row += "~"
                print(curr_row)                
            else:
                print(emptyr)
        print(topnbot)

    
    def next_action(self,robot): 
            """ 
            Choose an action from the current state using
            an e-greedy policy
            """
    
            robot_r = robot.pos[0,0]
            robot_c = robot.pos[0,1]
            bomb_r = self.bomb_pos[0,0]
            bomb_c = self.bomb_pos[0,1]
            
            #use uniform behavioral policy
            if(np.any(self.policy[robot_r,robot_c,bomb_r,bomb_c]) == False):
                action = robot.choose_action() 
                return action 


            #use e-greedy policy
            else: 
                
                #potential actions 
                L = [0,1,2,3]

                #get the index of the greedy action
                mval = self.policy[robot_r,robot_c,bomb_r,bomb_c,0]
                index = 0
                for i in range(0,4):  
                    tmp = self.policy[robot_r,robot_c,bomb_r,bomb_c,i]
                    if(tmp >= mval): 
                        mval = tmp 
                        index = i 

                #return the greedy action for the state
                rand_val = random.uniform(0,1)
                if(rand_val <= mval): 
                    action = robot.actions[index]

                #epsilon - return a non-greedy action
                else:
                    del L[index]
                    ran_index = random.randint(0,2)
                    action = robot.actions[L[ran_index]]

            return action

    
    def Monte_Carlo(self,robot,alpha,epsilon):
        """ 
        This function performs one episode of Monte Carlo.
        Returns the total return for the episode after completion
        """

        Env = self
        R3 = robot
        retval = 0.0
                
        #runs a episode until a TS is reached or 1000 steps are made
        while(Env.terminate_flag != 1):
            
            #get and take next action
            a = Env.next_action(R3) 
            R3.next_position(a)

            #get reward and next state
            pushed, bomb_coords = Env.push_bomb(R3)
            reward = Env.get_reward(pushed,bomb_coords)

            #add reward to total return
            R3.rewards.append((R3.time_step-1,reward))
            retval += reward

            #limit episodes to 1000 steps
            if(Env.time_step-1 == 1000):                  
                Env.terminate_flag = 1

        #episode created - update Q(s,a)
        Env.mc_update(R3,epsilon,alpha)

        return retval, Env.time_step-1

    
    def Print_Episode(self,robot):
        """
        This function plots out an episode given a starting state
        and a policy.
        """
        
        #runs a episode until a TS is reached or 1000 steps are made
        while(self.terminate_flag != 1):
            
            #print out state information
            print("\n\nS({}): |".format(self.time_step-1),end=" ")
            print("Robot: ({},{}) |".format(robot.pos[0,0],robot.pos[0,1]),end=" ")
            print("Bomb: ({},{}) |".format(self.bomb_pos[0,0],self.bomb_pos[0,1]))
            
            #print current board
            self.print_board(robot)

            #get and take next action
            a = self.next_action(robot) 
            robot.next_position(a)

            #get reward and next state
            pushed, bomb_coords = self.push_bomb(robot)
            reward = self.get_reward(pushed,bomb_coords)

            #print action
            print("Action: robot moves {} to ({},{})".format(a,robot.pos[0,0],robot.pos[0,1]))
            
            #print result/next state
            if(pushed):
                print("Result: bomb is pushed to ({},{})".format(self.bomb_pos[0,0],self.bomb_pos[0,1]))
            else:
                print("Result: bomb remains at ({},{})".format(self.bomb_pos[0,0],self.bomb_pos[0,1]))
            
            #print reward 
            print("Reward:",reward)

            #limit episodes to 1000 steps
            if(self.time_step-1 == 1000):                  
                self.terminate_flag = 1
        
        #print final board
        print("\n\nTS Reached - Final Board")
        self.print_board(robot)



def Learning(environment,robot,alpha,epsilon,nEpisodes,method):
    """
    This function facilitates the learning. It will call 
    either the Q-learning or MC method n times and store
    the returns in an array so it can be plotted.
    """


    #array for returns
    returns = []
    steps = []

    TS = 0.0
    tmp = 0.0
    
    scale = np.arange(0,100000,step=500)
    scale1 = np.arange(0,100000,step=1)
    print(np.shape(scale))    

    print(nEpisodes)
    for i in range(0,nEpisodes):

        #initialize and randomly place bomb and robot
        environment.initialize()
        robot.initialize()
        ret = 0.0

        #output updates every 100 episodes
        if((i % 500) == 0):
            print("{} episodes completed: avg = {} steps".format(i,round((TS/500.0),1)))
            steps.append(round((TS/100.0),1))
            TS = 0.0
        
        #call MC
        if(method == 1):
            ret,tmp = environment.Monte_Carlo(robot,alpha,epsilon)
        
        #call Q-Learning
        elif(method == 2):
            ret,tmp = environment.Q_Learning(robot,epsilon,alpha)
        
        #store return in array
        TS += tmp
        
        returns.append(ret)
        steps.append(tmp)
    
    #re-initialize and print an episode
    environment.initialize()
    robot.initialize()
    environment.Print_Episode(robot)

    plt.plot(scale1,returns)
    plt.title("Returns over 100,000 Episodes Scheme 1")
    plt.xlabel("# Episodes")
    plt.ylabel("Returns")
    plt.show()
    
    plt.plot(scale,steps)
    plt.title("Avg. Steps over 100,000 Episodes Scheme 2")
    plt.xlabel("# Episodes")
    plt.ylabel("Avg. Steps")
    plt.show()


def main():

    #argument check
    if(len(sys.argv) != 6):
	    print("usage: python3 Robot.py [dimension] [alpha] [epsilon] [num episodes] [method]")
	    sys.exit(1)

    #store and set parameters 
    d = int(sys.argv[1])
    a = float(sys.argv[2])
    e = float(sys.argv[3])
    n = int(sys.argv[4])
    m = int(sys.argv[5])
 
    #declare class objects
    Env = environment(d)
    R3 = robot(d)

    Learning(Env,R3,a,e,n,m)
    

if __name__ == "__main__":
    main()
