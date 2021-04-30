
import torch
import torch.nn as nn
import math

from transformer_v2 import *
from matplotlib import pyplot as plt
import pygame as py
from simulator2 import *

from collections import namedtuple
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn

import torch.optim as optim
# torch.random.manual_seed(5)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_subplots(data, legends, name):
    names = ['Accuracy', 'Loss']
    plt.figure(figsize=(10, 5))
    for i in range(len(data)):
        plt.subplot(131 + i)
        plt.plot(list(range(0, len(data[i]))), data[i])
        plt.title(legends[i])
        plt.xlabel("Episodes")
    plt.savefig(name)
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
BATCH_SIZE = 2048
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 10000
TARGET_UPDATE = 10
steps_done = 0

outputs = []
memory = ReplayMemory(4000)

eps_threshold_state = 0


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).type(torch.FloatTensor)
    non_final_next_states = non_final_next_states.to(device)

    state_batch = torch.cat(batch.state).type(torch.FloatTensor)
    action_batch = torch.cat(batch.action).type(torch.int64)
    reward_batch = torch.cat(batch.reward).type(torch.FloatTensor)
    state_batch = state_batch.to(device)
    action_batch = action_batch.to(device)
    reward_batch = reward_batch.to(device)

    policy_net.train()
    target_net.train()
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    # next_state_values = torch.zeros(len(memory), device=device)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    # print("Next State values",next_state_values.shape)

    nnex = target_net(non_final_next_states)
    nnex = nnex.max(1)[0]
    next_state_values[non_final_mask] = nnex.detach()
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

    data = data.type(torch.LongTensor).to(device)
    enc_inp, tar_inp = data[:64].to(device), data[64:].to(device)

    enc_inp = enc_inp.reshape(1, -1)
    tar_inp = tar_inp.reshape(1, -1)
    # print(enc_inp,tar_inp)
    src_mask, tar_mask = create_masks(enc_inp, tar_inp)
    # print(src_mask,tar_mask)
    # print(src_mask,tar_mask)
    prediction_model.eval()
    output = prediction_model(enc_inp, tar_inp, src_mask, tar_mask)
    return output


num_episodes = 1000

SAVE_INDEX = 20
LOAD_INDEX = 20
LOAD_FROM_FILE = True
SAVE_TO_FILE = False
TRAIN = False
VISUALIZE = True
SAVE_GRAPH = False
take_random_actions = False
models_dir = ""

sims = Simulator(VISUALIZE)
sims.set_data(models_dir + "../Dataset/EURUSD30min2015-17.csv")
sims.n_stop = 300
prediction_model = torch.load(models_dir + "../models/EvalMark-II-LearningRate-BenchMark-I",map_location=torch.device('cpu')).to(device)

save_plot_name = models_dir + "../Plots/DQN102_Trans_1_300_lr0.001.png"
SAVE_FILENAME = "DQN"
LOAD_FILENAME = "DQN"


def select_action_prandom(state, take_random=True):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    eps_threshold_state = eps_threshold
    # print(eps_threshold_state)

    if sample > eps_threshold or not take_random:
        with torch.no_grad():
            # print("State ",state)
            state = state.to(device)
            state = state.type(torch.FloatTensor).to(device)
            y = policy_net(state)
            x = y.max(1)[1]
            x = x.view(1, 1).to(device)
            outputs.append(x)
            return x
    else:

        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

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
            # print("State ",state)
            state = state.to(device)
            state = state.type(torch.FloatTensor).to(device)
            y = policy_net(state)
            x = y.max(1)[1]
            x = x.view(1, 1).to(device)
            outputs.append(x)
            return x
    else:

        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


if LOAD_FROM_FILE:
    policy_net = torch.load("../models/" + LOAD_FILENAME + "_policy" + str(LOAD_INDEX), map_location=device)
    target_net = torch.load("../models/" + LOAD_FILENAME + "_target" + str(LOAD_INDEX), map_location=device)
    # target_net.load_state_dict(policy_net.state_dict())
else:
    policy_net = DQN1(15, 64, 4, 2).to(device)
    target_net = DQN1(15, 64, 4, 2).to(device)
    policy_net.required_inputs = ["prediction", "balance_tensor", "equity_tensor", "new_profit_tensor",
                                  "profit_change_tensor", "state_tensor", "position_opened_type", "opened_time"]
    target_net.load_state_dict(policy_net.state_dict())
    target_net.to(device)

optimizer = optim.RMSprop(policy_net.parameters(), lr=0.001)

losses = []
rewards = []
total_rewards = []

prev_state = None
prev_output = None


def test_run():
    losses = []
    rewards = []
    total_rewards = []
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
    new_profit_tensor = torch.tensor([[0]]).to(device)
    profit_change_tensor = torch.tensor([[0]]).to(device)
    state_tensor = torch.tensor([[state]]).to(device)
    balance_tensor = torch.tensor([[0]]).to(device)
    position_opened_type = torch.tensor([[0]]).to(device)
    if sims.buy_position != None:
        position_opened_type = torch.tensor([[0.5]]).to(device)
    elif sims.sell_position != None:
        position_opened_type = torch.tensor([[1]]).to(device)

    all_inputs = {
        "prediction": predicted,
        "balance_tensor": balance_tensor,
        "equity_tensor": sims.equity.clone().detach(),
        "new_profit_tensor": new_profit_tensor,
        "profit_change_tensor": profit_change_tensor,
        "state_tensor": state_tensor,
        "position_opened_type": position_opened_type,
        "opened_time": sims.opened_time
    }
    # print(all_inputs)

    dq_state = torch.tensor(predicted).to(device)
    for inp in policy_net.required_inputs:
        if inp != "prediction":
            dq_state = torch.cat([dq_state, torch.tensor([[all_inputs[inp]]]).to(device)], dim=-1).to(device)
    # print(dq_state.shape)
    # dq_state = torch.cat( (predicted,balance_tensor,new_profit_tensor, profit_change_tensor,state_tensor ),dim=-1).to(device).type(torch.FloatTensor)
    # *******************************************
    dq_state.to(device)
    all_states = torch.tensor(dq_state.view(1, -1))
    buys = 0
    sells = 0
    closed = 0
    no_actions = 0
    avg_losses = 0

    all_outputs = torch.tensor(predicted)

    total_reward = 0
    tt1 = time.time()
    for steps in range(1000):
        policy_net.eval()
        target_net.eval()
        # print(dq_state.shape)

        action_selected = select_action_prandom(dq_state,False)
        if (action_selected == 0):
            no_actions += 1
        if (action_selected == 1):
            buys += 1
        if action_selected == 2:
            sells += 1
        if action_selected == 3:
            closed += 1

        state, new_profit, current_reward = sims.step(action_selected[0][0])
        total_reward += current_reward
        # current_rewards.append(current_reward)
        balance = sims.balance.item()

        with torch.no_grad():
            output = predict_transformer(sims.window[0])

        predicted = torch.argmax(output, dim=-1)
        all_outputs = torch.cat((all_outputs, predicted), dim=0).to(device)

        # *******************************************
        predicted = predicted - predicted[0][0]
        new_profit_tensor = torch.tensor([[new_profit]]).to(device)
        profit_change_tensor = torch.tensor([[new_profit - profit]]).to(device)
        balance_tensor = torch.tensor([[balance]]).to(device)
        state_tensor = torch.tensor([[state]]).to(device)

        position_opened_type = torch.tensor([[0]]).to(device)
        if sims.buy_position != None:
            position_opened_type = torch.tensor([[0.5]]).to(device)
        elif sims.sell_position != None:
            position_opened_type = torch.tensor([[1]]).to(device)
        all_inputs = {
            "prediction": predicted,
            "balance_tensor": balance_tensor,
            "equity_tensor": sims.equity.clone().detach().to(device),
            "new_profit_tensor": new_profit_tensor,
            "profit_change_tensor": profit_change_tensor,
            "state_tensor": state_tensor,
            "position_opened_type": position_opened_type,
            "opened_time": sims.opened_time
        }
        next_state = torch.tensor(predicted).to(device)

        for inp in policy_net.required_inputs:
            if inp != "prediction":
                # print(next_state,all_inputs[inp])
                next_state = torch.cat([next_state, torch.tensor([[all_inputs[inp]]]).to(device)], dim=-1)
        # next_state = torch.cat((predicted,balance_tensor,new_profit_tensor,profit_change_tensor , state_tensor  ),dim = -1).to(device).type(torch.FloatTensor)
        # print(steps,action_selected,new_profit_tensor,profit_change_tensor)

        profit = new_profit
        # *******************************************
        current_reward = torch.tensor([current_reward]).to(device)
        # memory.push(dq_state, action_selected, next_state, current_reward)
        dq_state = next_state.clone().detach()
        dq_state.to(device)
    # print("Balance",balance,"No aciton",)
    print("%d No action: %d Buys: %d Sells: %d Closed: %d Balance: %d Time: %f %s" % (
        episode, no_actions, buys, sells, closed, sims.balance, time.time() - tt1, time.ctime()), )
    return balance

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
    new_profit_tensor = torch.tensor([[0]]).to(device)
    profit_change_tensor = torch.tensor([[0]]).to(device)
    state_tensor = torch.tensor([[state]]).to(device)
    balance_tensor = torch.tensor([[0]]).to(device)
    position_opened_type = torch.tensor([[0]]).to(device)
    if sims.buy_position != None:
        position_opened_type = torch.tensor([[0.5]]).to(device)
    elif sims.sell_position != None:
        position_opened_type = torch.tensor([[1]]).to(device)

    all_inputs = {
        "prediction": predicted,
        "balance_tensor": balance_tensor,
        "equity_tensor": sims.equity.clone().detach(),
        "new_profit_tensor": new_profit_tensor,
        "profit_change_tensor": profit_change_tensor,
        "state_tensor": state_tensor,
        "position_opened_type": position_opened_type,
        "opened_time": sims.opened_time
    }
    # print(all_inputs)
    dq_state = torch.tensor(predicted).to(device)
    for inp in policy_net.required_inputs:
        if inp != "prediction":
            dq_state = torch.cat([dq_state, torch.tensor([[all_inputs[inp]]]).to(device)], dim=-1).to(device)
    # print(dq_state.shape)
    # dq_state = torch.cat( (predicted,balance_tensor,new_profit_tensor, profit_change_tensor,state_tensor ),dim=-1).to(device).type(torch.FloatTensor)
    # *******************************************
    dq_state.to(device)
    all_states = torch.tensor(dq_state.view(1, -1))
    buys = 0
    sells = 0
    closed = 0
    no_actions = 0
    avg_losses = 0
    all_outputs = torch.tensor(predicted)
    total_reward = 0
    tt1 = time.time()
    for steps in range(300):
        policy_net.eval()
        target_net.eval()
        # print(dq_state.shape)

        action_selected = select_action(dq_state)
        if (action_selected == 0):
            no_actions += 1
        if (action_selected == 1):
            buys += 1
        if action_selected == 2:
            sells += 1
        if action_selected == 3:
            closed += 1

        state, new_profit, current_reward = sims.step(action_selected[0][0])
        total_reward += current_reward
        # current_rewards.append(current_reward)
        balance = sims.balance.item()

        with torch.no_grad():
            output = predict_transformer(sims.window[0])

        predicted = torch.argmax(output, dim=-1)
        all_outputs = torch.cat((all_outputs, predicted), dim=0).to(device)

        # *******************************************
        predicted = predicted - predicted[0][0]
        new_profit_tensor = torch.tensor([[new_profit]]).to(device)
        profit_change_tensor = torch.tensor([[new_profit - profit]]).to(device)
        balance_tensor = torch.tensor([[balance]]).to(device)
        state_tensor = torch.tensor([[state]]).to(device)

        position_opened_type = torch.tensor([[0]]).to(device)
        if sims.buy_position != None:
            position_opened_type = torch.tensor([[0.5]]).to(device)
        elif sims.sell_position != None:
            position_opened_type = torch.tensor([[1]]).to(device)

        all_inputs = {
            "prediction": predicted,
            "balance_tensor": balance_tensor,
            "equity_tensor": sims.equity.clone().detach().to(device),
            "new_profit_tensor": new_profit_tensor,
            "profit_change_tensor": profit_change_tensor,
            "state_tensor": state_tensor,
            "position_opened_type": position_opened_type,
            "opened_time": sims.opened_time
        }
        next_state = torch.tensor(predicted).to(device)

        for inp in policy_net.required_inputs:
            if inp != "prediction":
                # print(next_state,all_inputs[inp])
                next_state = torch.cat([next_state, torch.tensor([[all_inputs[inp]]]).to(device)], dim=-1)
        # next_state = torch.cat((predicted,balance_tensor,new_profit_tensor,profit_change_tensor , state_tensor  ),dim = -1).to(device).type(torch.FloatTensor)
        # print(steps,action_selected,new_profit_tensor,profit_change_tensor)

        profit = new_profit
        # *******************************************
        current_reward = torch.tensor([current_reward]).to(device)
        memory.push(dq_state, action_selected, next_state, current_reward)
        dq_state = next_state.clone().detach()
        dq_state.to(device)
        # all_states = torch.cat((all_states,dq_state.view(1,-1))).to(device)
        # print("Balance",balance)
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
            py.display.update()
            sims.clock.tick(30)

    if episode % 10 == 9:
        target_net.load_state_dict(policy_net.state_dict())
    total_rewards.append(total_reward)
    eps_threshold_state = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    print("%d No action: %d Buys: %d Sells: %d Closed: %d Balance: %d EPS: %f Time: %f %s" % (
    episode, no_actions, buys, sells, closed, sims.balance, eps_threshold_state, time.time() - tt1, time.ctime()), )

    losses.append(avg_losses / steps)
    # rewards.append(test_run())
    if SAVE_GRAPH:
        plot_subplots([losses, rewards, total_rewards], ["loss", "balance", "rewards"], save_plot_name)
    # print(loss,balance)
    if SAVE_TO_FILE and episode % 5 == 0:
        torch.save(policy_net, "../models/" + SAVE_FILENAME + "_policy" + str(SAVE_INDEX))
        torch.save(target_net, "../models/" + SAVE_FILENAME + "_target" + str(SAVE_INDEX))