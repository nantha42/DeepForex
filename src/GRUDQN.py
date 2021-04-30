import torch
import torch.nn as nn
import math
from transformer_v2 import *
import pygame as py
from simulator import *
from collections import namedtuple
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
import torch.optim as optim
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

save_at = "../Plots/"


def plot_subplots(data, legends, name):
    names = ['Accuracy', 'Loss']
    plt.figure(figsize=(10, 5))
    for i in range(len(data)):
        plt.subplot(131 + i)
        plt.plot(list(range(0, len(data[i]))), data[i])
        plt.title(legends[i])
        plt.xlabel("Episodes")
    plt.savefig(save_at + name)
    plt.clf()


def plot_multiple(data, legend):
    fig, ax = plt.subplots()
    for line in data:
        print(line)
        plt.plot(list(range(len(line))), line)
    plt.legend(legend)
    plt.show()


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.max_reward = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        transition = Transition(*args)

        if transition.reward > self.max_reward:
            self.max_reward = transition.reward
            self.memory[self.position] = transition
            self.position = (self.position + 1) % self.capacity
        else:
            if self.memory[self.position] != None:
                if self.memory[self.position].reward == self.max_reward:
                    self.position = (self.position + 1) % self.capacity
                    self.memory[self.position] = transition
                    self.position = (self.position + 1) % self.capacity
                else:
                    self.memory[self.position] = transition
                    self.position = (self.position + 1) % self.capacity
            else:
                self.memory[self.position] = transition
                self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, inp, hidden, out, nhiddenlayers=4):
        super().__init__()
        self.layer1 = nn.Linear(inp, hidden)
        self.activation1 = nn.ReLU()
        self.hidden_layers = []

        for x in range(nhiddenlayers):
            self.hidden_layers.append(nn.Linear(hidden, hidden))
            self.hidden_layers.append(nn.Dropout(0.25))

        # self.hidden_layers = [nn.Linear(hidden,hidden),nn.Dropout() for x in range(nhiddenlayers)]
        self.layer3 = nn.Linear(hidden, out)
        self.softmax = nn.Softmax()
        pass

    def forward(self, x):
        o = self.activation1(self.layer1(x))
        for l in self.hidden_layers:
            o = l(o)
        o = self.layer3(o)

        return self.softmax(o)


# Factory Reset
# class DQN1(nn.Module):
#     def __init__(self,inp,hidden,out,nhiddenlayers=4):
#         super().__init__()
#         self.layer1 = nn.Linear(inp,hidden)
#         self.activation1 = nn.ReLU()
#         self.hidden_layers = []
#         count = 0
#         for x in range(nhiddenlayers):
#             count+=2
#             self.hidden_layers.append((str(count-2), nn.Linear(hidden,hidden)))
#             self.hidden_layers.append((str(count-1),nn.Dropout(0.1)) )

#         self.hidden_layers = nn.Sequential(OrderedDict(self.hidden_layers))
#         # self.hidden_layers = [nn.Linear(hidden,hidden),nn.Dropout() for x in range(nhiddenlayers)]
#         self.layer3 = nn.Linear(hidden,out)
#         self.softmax = nn.Softmax()
#         pass

#     def forward(self,x):
#         # state = x[:,-1]
#         # x = x[:,:-1]
#         # embedded_state = self.state_embedder(state.type(torch.LongTensor))

#         # x = torch.cat([x,embedded_state],dim=-1)
#         # print("Modified x ",x)
#         o = self.activation1(self.layer1(x))
#         o = self.hidden_layers(o)
#         o = self.layer3(o)
#         # return o
#         return self.softmax(o)

class DQN1(nn.Module):
    def __init__(self, inp, hidden, out, nhiddenlayers=4):
        super().__init__()
        self.layer1 = nn.Linear(inp, hidden)
        self.activation1 = nn.ReLU()
        self.hidden_layers = []
        self.activations = []
        count = 0
        for x in range(nhiddenlayers):
            count += 2
            self.hidden_layers.append((str(count - 2), nn.Linear(hidden, hidden)))
            self.hidden_layers.append((str(count - 1), nn.Dropout(0.2)))

        self.hidden_layers = nn.Sequential(OrderedDict(self.hidden_layers))
        # self.hidden_layers = [nn.Linear(hidden,hidden),nn.Dropout() for x in range(nhiddenlayers)]
        self.layer3 = nn.Linear(hidden, out)
        self.softmax = nn.Softmax()
        pass

    def forward(self, x):
        self.activations = []
        o = self.activation1(self.layer1(x))
        self.activations.append(o.clone().detach())
        i = 0
        for t in self.hidden_layers:
            o = t(o)
            if i % 2 == 0:
                self.activations.append(o.clone().detach())
            i += 1
        o = self.softmax(self.layer3(o))
        self.activations.append(o.clone().detach())
        return o


class DQN_Memory(nn.Module):
    def __init__(self, inp, hidden, out, nhiddenlayers=4):
        super().__init__()
        self.layer1 = nn.Linear(inp, hidden)
        self.activation1 = nn.ReLU()
        self.hidden_layers = []
        count = 0
        for x in range(nhiddenlayers):
            # count+=3
            # self.hidden_layers.append((str(count-3), nn.Linear(hidden,hidden)))
            # self.hidden_layers.append((str(count-2),nn.Dropout(0.08)) )
            # self.hidden_layers.append((str(count-1),nn.LayerNorm([hidden])) )
            self.hidden_layers.append(nn.Linear(hidden, hidden))
            self.hidden_layers.append(nn.Dropout(0.1))
            self.hidden_layers.append(nn.LayerNorm([hidden]))

        # self.hidden_layers = nn.Sequential(OrderedDict(self.hidden_layers))

        # self.hidden_layers.append(nn.Dropout(0.25))
        # self.hidden_layers = [nn.Linear(hidden,hidden),nn.Dropout() for x in range(nhiddenlayers)]
        self.layer3 = nn.Linear(hidden, out)
        self.softmax = nn.Softmax()
        self.relu = torch.nn.ReLU()
        self.activations = []

    def mode(self, m):
        if m == "eval":
            self.eval()
            for x in self.hidden_layers:
                x.eval()

        elif m == "train":
            self.train()
            for x in self.hidden_layers:
                x.train()

    def forward(self, x):
        o = self.activation1(self.layer1(x))
        self.activations.append(o)
        for i in range(0, len(self.hidden_layers), 3):
            t = self.hidden_layers[i](o)  # hidden layer
            o = o + self.hidden_layers[i + 1](t)  # dropout
            o = self.hidden_layers[i + 2](o)
            self.activations.append(o.clone().detach())
        o = self.layer3(o)
        self.activations.append(o.clone().detach())
        o = self.softmax(o)
        self.activations.append(o.clone().detach())
        return o


n_actions = 4
BATCH_SIZE = 30
GAMMA = 0.90
EPS_START = 0.09
EPS_END = 0.01
EPS_DECAY = 1000
TARGET_UPDATE = 10
steps_done = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
outputs = []
memory = ReplayMemory(5000)

eps_threshold_state = 0


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # transitions = memory.sample(len(memory))
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).type(torch.FloatTensor)
    # print("Non_final_net_states shape ",non_final_next_states.shape)
    state_batch = torch.cat(batch.state).type(torch.FloatTensor)
    action_batch = torch.cat(batch.action).type(torch.FloatTensor)
    reward_batch = torch.cat(batch.reward).type(torch.FloatTensor)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # policy_net.train()
    # target_net.train()
    policy_net.train()
    target_net.train()
    state_action_values = policy_net(state_batch).gather(1, action_batch.type(torch.int64))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    # next_state_values = torch.zeros(len(memory), device=device)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    # print("Next State values",next_state_values.shape)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    # print("Next State values",next_state_values.shape," Reward_batch ",reward_batch.shape)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    # print("Loss Shapes: ",state_action_values.shape," : ",expected_state_action_values.unsqueeze(1).shape)
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in policy_net.parameters():
    #     print(param)

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()


def predict_transformer(data):
    # data = sims.window[0]# taking only the open prices
    data = ((data - data[0]) * 1e5) // 50 + 120 + 1
    data = data.type(torch.LongTensor)
    # print(data)
    enc_inp, tar_inp = data[:64], data[64:]

    # print(enc_inp,tar_inp)
    enc_inp = enc_inp.reshape(1, -1)
    tar_inp = tar_inp.reshape(1, -1)

    src_mask, tar_mask = create_masks(enc_inp, tar_inp)
    # print(src_mask,tar_mask)
    prediction_model.eval()
    output = prediction_model(enc_inp, tar_inp, src_mask, tar_mask)
    return output


num_episodes = 1000

SAVE_INDEX = 1000
LOAD_INDEX = 17
LOAD_FROM_FILE = True
SAVE_TO_FILE = False
TRAIN = False
VISUALIZE = False
SAVE_GRAPH = False
take_random_actions = False

sims = Simulator(VISUALIZE)
sims.set_data("../dataset/EURUSD30min2018-20.csv")
prediction_model = torch.load("../models/LSTM/EvalMark-I-GRU-2")

save_plot_name = "../models/DQN12_2.png"
SAVE_FILENAME = "DQN"
LOAD_FILENAME = "DQN"


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    eps_threshold_state = eps_threshold
    # print(eps_threshold_state)
    if sample > eps_threshold or not take_random_actions:
        with torch.no_grad():
            y = policy_net(state)
            p = y.max(1)[0]
            # if p > 0.7:
            x = y.max(1)[1].view(1, 1)
            # else:
            #     x = torch.tensor([[0]])
            outputs.append(x)
            return x, p
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), 0


if LOAD_FROM_FILE:
    policy_net = torch.load("../models/" + LOAD_FILENAME + "_policy" + str(LOAD_INDEX), map_location=device)
    target_net = torch.load("../models/" + LOAD_FILENAME + "_target" + str(LOAD_INDEX), map_location=device)
    # target_net.load_state_dict(policy_net.state_dict())
else:
    policy_net = DQN1(12, 20, 4, 6)
    target_net = DQN1(12, 20, 4, 6)
    target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.RMSprop(policy_net.parameters(), lr=0.01)

losses = []
rewards = []
prev_state = None
prev_output = None

for episode in range(num_episodes):
    sims.reset()
    state = 0
    profit = 0
    new_profit = 0
    balance = 0
    outputs = []

    with torch.no_grad():
        output = predict_transformer(sims.window[0])
    predicted = torch.argmax(output, dim=-1)

    policy_net.eval()
    target_net.eval()

    # *******************************************
    predicted = predicted - predicted[0][0]
    new_profit_tensor = torch.tensor([[0]])
    profit_change_tensor = torch.tensor([[0]])
    state_tensor = torch.tensor([[state]])
    balance_tensor = torch.tensor([[0]])

    dq_state = torch.cat((predicted, balance_tensor, new_profit_tensor, profit_change_tensor, state_tensor),
                         dim=-1).type(torch.FloatTensor)
    # *******************************************
    all_states = torch.tensor(dq_state.view(1, -1))
    buys = 0
    sells = 0
    closed = 0
    no_actions = 0
    avg_losses = 0
    all_outputs = torch.tensor(predicted)
    current_rewards = []
    tt1 = time.time()

    for steps in range(1000):
        policy_net.eval()
        target_net.eval()
        action_selected, prob = select_action(dq_state.type(torch.FloatTensor))

        if (action_selected == 0):
            no_actions += 1
        if (action_selected == 1):
            buys += 1
            # if prob > 0.6:
            #     print("Buy",prob)
        if action_selected == 2:
            sells += 1
            # if prob > 0.6:
            # print("Sell",prob)
            # print("Sell",prob)
        if action_selected == 3:
            closed += 1
            # print("Close",prob)

        state, new_profit, current_reward = sims.step(action_selected[0][0])
        current_rewards.append(current_reward)
        balance = sims.balance.item()
        with torch.no_grad():
            output = predict_transformer(sims.window[0])

        predicted = torch.argmax(output, dim=-1)
        all_outputs = torch.cat((all_outputs, predicted), dim=0)

        # *******************************************
        predicted = predicted - predicted[0][0]
        new_profit_tensor = torch.tensor([[new_profit]])
        profit_change_tensor = torch.tensor([[new_profit - profit]])
        balance_tensor = torch.tensor([[balance]])
        state_tensor = torch.tensor([[state]])

        next_state = torch.cat((predicted, balance_tensor, new_profit_tensor, profit_change_tensor, state_tensor),
                               dim=-1).type(torch.FloatTensor)
        # print(steps,action_selected,new_profit_tensor,profit_change_tensor)
        profit = new_profit
        # *******************************************
        current_reward = torch.tensor([current_reward])
        memory.push(dq_state, action_selected, next_state, current_reward)
        dq_state = next_state.clone().detach()
        all_states = torch.cat((all_states, dq_state.view(1, -1)))

        loss = 0
        if TRAIN:
            loss = optimize_model()
        if loss != None:
            avg_losses += loss

        if VISUALIZE:
            sims.draw()
            w = []
            wparameters = list(policy_net.parameters())
            for p in range(0, len(policy_net.activations)):
                if p * 2 < len(wparameters):
                    w.append(wparameters[p * 2])
                w.append(policy_net.activations[p])

            sims.plot_weights(w)
            # sims.plot_weights(policy_net.activations)
            py.display.update()
            sims.clock.tick(130)
    balances = sims.balance_history
    equities = sims.equity_history
    plot_multiple([balances, equities], ["balance", "equity"])
    # if prev_state == None:
    #     prev_state = all_states
    # else:
    #     print(prev_state==all_states)
    #     prev_state = all_states

    # if prev_output == None:
    #     prev_output = all_outputs
    #     print("All Outputs shape",all_outputs.shape)
    # else:
    #     print("All Outputs shape",all_outputs.shape,prev_output.shape)
    #     print(prev_output==all_outputs)
    #     prev_output = all_outputs

    if episode % 20 == 19:
        target_net.load_state_dict(policy_net.state_dict())
    eps_threshold_state = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    print("%d No action: %d Buys: %d Sells: %d Closed: %d Balance: %d EPS: %f %f" % (
    episode, no_actions, buys, sells, closed, sims.balance, eps_threshold_state, time.time() - tt1), )
    print("Successfull buys: %d Successfull sells: %d Trades: %d" % (
    sims.successfull_buys, sims.successfull_sells, len(sims.history_orders)))
    losses.append(avg_losses / steps)
    rewards.append(balance)
    if SAVE_GRAPH:
        plot_subplots([losses, rewards, current_rewards], ["loss", "balance", "rewards"], save_plot_name)
    # print(loss,balance)
    break
    if SAVE_TO_FILE and episode % 5 == 0:
        torch.save(policy_net, "../models/" + SAVE_FILENAME + "_policy" + str(SAVE_INDEX))
        torch.save(target_net, "../models/" + SAVE_FILENAME + "_target" + str(SAVE_INDEX))