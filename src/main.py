import random 
import torch
from torch import nn
from torch import tensor

class brain(nn.Module):
    def __init__(self,I,O):
        super(brain,self).__init__()
        self.layer1 = nn.Linear(I,5)
        self.layer2 = nn.Linear(5,5)
        self.layer3 = nn.Linear(5,O)

    def forward(self,inp):
        o1 = self.layer1(inp).clamp(min=0)
        o2 = self.layer2(o1).clamp(min=0)
        o3 = self.layer3(o2).clamp(min=0)
        return o3

policy = [[0,0,0,0],
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,0]]

policy_net = brain(9,4)

env = [[2,-1,10],
        [-1,-10,-1],
        [-1,-1,10]]

lizard_pos = [2,0]

def argmax(x):
    val = 0
    ind = 0
    
    x = tensor(x)
    # print(x)
    return torch.argmax(x)
    for i in range(len(x)):
        if x[i] >val:
            val = x[i]
            ind = i 
    return ind 

epsilon = 1
wrong = 0
learning_rate = 0.07
steps = 0
reward = 0
actions = []
max_reward = 0
for i in range(10000):
    
    again = True
    next_action = -1
    old_state = lizard_pos[0]*3 + lizard_pos[1]
    opts = []
    old_agent_state = torch.zeros(1,9)
    old_agent_state[0][old_state] = 1
    
    # 0 Left
    # 1 Up
    # 2 Right
    # 3 Down
    if lizard_pos[0] >0:
        opts.append(1)
    if lizard_pos[0] <2:
        opts.append(3)

    if lizard_pos[1] >0:
        opts.append(0)
    if lizard_pos[1] <2:
        opts.append(2)
    
    again_count = 0
    # print(opts)
    while again:
        if random.random() < epsilon:
            next_action = random.choice(opts)
        elif again_count<5:
            next_action = argmax(policy_net(old_agent_state))
            # next_action = argmax(policy[lizard_pos[0]*3+lizard_pos[1]])
        else:
            next_action = random.choice(opts)
        again_count+=1
        # print(next_action)
        if next_action == 0:
            #left
            if(lizard_pos[1]>0):
                lizard_pos[1]-=1 
                again = False
                actions.append("L")

        elif next_action == 1:
            #up
            if(lizard_pos[0]>0):
                lizard_pos[0]-=1 
                again = False
                actions.append("U")

        elif next_action == 2:
            #right
            if(lizard_pos[1]<2):
                lizard_pos[1]+=1 
                again = False
                actions.append("R")

        elif next_action == 3:
            #down
            if(lizard_pos[0]<2):
                lizard_pos[0]+=1 
                again = False
                actions.append("D")
        if again:
            wrong+=1
        
    # reward += env[lizard_pos[0]][lizard_pos[1]]
    env[lizard_pos[0]][lizard_pos[1]] =0

    
    x,y = lizard_pos[0], lizard_pos[1]
    old_value = policy[old_state][next_action]
    policy[old_state ][next_action] = (1-learning_rate)*(old_value) + learning_rate*(reward + 0.99*(argmax(policy[x*3+y])))
    # print(i)
    if lizard_pos == [1,1] or lizard_pos==[0,2]:
        lizard_pos=[2,0]
        epsilon -= 0.001
        
        print(f"Reward:{reward} Wrong:{wrong}")
        print(actions)
        print("Restarting Game after steps: ",steps,epsilon)
        # print(actions)
        if max_reward < reward:
            max_reward = reward 
        reward = 0
        steps = 0
        actions = []
        env = [ [ 2, -1,10],
                [-1,-10,-1],
                [-1, -1,7]]

    else:
        steps+=1

act = len(actions)
x,y= lizard_pos[0],lizard_pos[1]

print(f"Actions = {act} Reward = {max_reward} steps = {steps} lizardpos = {x} {y}")
print()