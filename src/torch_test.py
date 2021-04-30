import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optimizer
import matplotlib.pyplot as plt
from visual_weights import *
from collections import *
from transformer_v2 import  *
# import fxcmpy
import pygame as py

py.init()


class N(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(5,2)
        self.layers = 4
        self.gru = nn.GRU(2, 512, self.layers, batch_first=True)
        self.bat = nn.BatchNorm1d(4)
        self.bat1 = nn.BatchNorm1d(4)
        self.bat2 = nn.BatchNorm1d(4)
        self.fc = nn.Linear(512,100)
        self.fc1 = nn.Linear(100,100)
        self.fc2 = nn.Linear(100,5)
        # self.s = nn.Softmax(dim=-1)
    
    def forward(self,x):
        h0 = torch.zeros(self.layers, x.size(0), 512).requires_grad_()

        x = self.embed(x)

        x,hn = self.gru(x,h0)
        x  = self.bat(x)
        x = self.fc(x)
        x = nn.functional.relu(x)

        x = self.bat1(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)

        x = self.bat2(x)
        x  = self.fc2(x)
        # softmaxed = self.s(x)
        return  x


torch.manual_seed(10)
import torch.nn.utils.prune as prune


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Linear(4,2)

    def forward(self,x):
        o =  torch.nn.functional.softmax(self.s(x),dim=-1)
        return Categorical(o)


tensor = torch.tensor
n = NN()

s = Simulator(True)

w = torch.randn(4,2)
i = torch.randn(2,4)
# par = torch.nn.parameter.Parameter(w)

def plot(frame_idx, rewards):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('episodes %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.savefig("one.png")

# m = torch.load("../models/EvalMark-II-LearningRate")

# prune.random_unstructured(mod, name="weight", amount=0.3)
# print(mod.weight)
#
dist = n(i)
optim = optimizer.Adam(n.parameters(),lr=0.01)


# print(log_prob)
# o = torch.log(o)
# exit()
training_rewards = []
for q in range(1000):
    dist = n(i)
    action = dist.sample()
    rewards = []
    print(dist.probs)
    tot = 0
    if action[0] == 0:
        rewards.append(0)
    else:
        rewards.append(1)
    if action[1] == 1:
        rewards.append(0)
    else:
        rewards.append(1)


    log_prob = dist.log_prob(action)
    rewards = torch.tensor(rewards)
    tot = rewards.sum().item()
    training_rewards.append(tot)
    plot(range(q), training_rewards)
    # print(rewards)
    loss = -(log_prob * rewards.detach()).mean()
    # print("Action", action,"  ",loss)
    plt.plot()
    loss.backward()
    s.event_handler()
    s.draw_model(n)
    s.draw()
    optim.step()
    optim.zero_grad()

#
# o = torch.mm(i,par)
# o = torch.softmax(o,dim=-1)
# # print(o)
# c = Categorical(o)
# print(c.logits,c.sample())
# c.logits[0][1] += 1.6
# print(c.logits,c.sample())
# print(c.logits,c.sample())
# print(c.logits,c.sample())
#


    # par.grad = None


# for  i in range(10):
#     a = n(torch.tensor([[2.0, 3.0]]))
#     # print("Prob",a)
#     # g = torch.distributions.Categorical(a)
#
#     # print(g.probs)
#     # log_probs = g.log_prob(tensor([1]))
#     # print("Weights")
#     print(a)
#
#     gg  = torch.log(a[0][1])
#     print(gg)
#     gg.backward()
#
#     # x = torch.log(a)
#     # print(x)
#     # x[0][1].backward()
#
#     # print(log_probs)
#     # log_probs.backward()
#
#     for p in n.parameters():
#         # x = p
#         p = p+ p.grad*0.1
#         print(p.grad)
#     # for p in n.parameters():
#     #     print(p)
#     # n.zero_grad()
#     # p += p.grad*0.01
#
#
# #
# # print(g.entropy().mean())
# # print(g.probs)
# # print(g.log_prob(torch.tensor(1)))
# # # print(g.probs)
# # print(dir(g))
# # # print(g.sample())
# #
