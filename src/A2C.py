import torch
import torch.nn as nn
import math
from transformer_v2 import *
from matplotlib import pyplot as plt
import pygame as py
from simulator2 import *
from collections import namedtuple
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
torch.random.manual_seed(5)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


device = torch.device("cpu")



def plot(frame_idx, rewards,name):

    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.savefig(name)
    plt.clf()

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



class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value





SAVE_INDEX = 103
LOAD_INDEX = 103
LOAD_FROM_FILE = False
SAVE_TO_FILE = False
TRAIN = True
VISUALIZE = False
SAVE_GRAPH = True
take_random_actions = True
models_dir = ""


sims = Simulator(VISUALIZE)
sims.set_data(models_dir + "../Dataset/EURUSD30min2015-17.csv")
prediction_model = torch.load(models_dir + "../models/EvalTransformer-II-1",map_location=torch.device('cpu')).to(device)

save_plot_name = models_dir + "../Plots/DQN103.png"
SAVE_FILENAME = "DQN"
LOAD_FILENAME = "DQN"


num_inputs  = 15
num_outputs = 4

#Hyper params:
hidden_size = 64
lr          = 3e-4
num_steps   = 5

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters())

sims = Simulator(VISUALIZE)
sims.set_data(models_dir + "../Dataset/EURUSD30min2015-17.csv")
sims1 = Simulator(VISUALIZE)
sims1.set_data(models_dir + "../Dataset/EURUSD30min2015-17.csv")
prediction_model = torch.load(models_dir + "../models/EvalTransformer-II-1",map_location=torch.device('cpu')).to(device)

required_inputs = ["prediction", "balance_tensor", "equity_tensor", "new_profit_tensor",
                              "profit_change_tensor", "state_tensor", "position_opened_type", "opened_time"]



def convert_to_state(state,new_profit_tensor):
    with torch.no_grad():
        output = predict_transformer(sims.window[0])
    predicted = torch.argmax(output, dim=-1)
    predicted = predicted - predicted[0][0]
    new_profit_tensor = torch.tensor([[new_profit_tensor]]).to(device)
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
    dq_state = torch.tensor(predicted).to(device)
    for inp in required_inputs:
        if inp != "prediction":
            dq_state = torch.cat([dq_state, torch.tensor([[all_inputs[inp]]]).to(device)], dim=-1).to(device)
    dq_state = dq_state.type(torch.FloatTensor).to(device)
    return dq_state

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    # print(rewards,)
    # print("Masks",masks)
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
        # print("RR",returns)

    return returns


def test_env(vis=False):
    state = sims1.reset()
    done = False
    total_reward = 0
    dq_state = convert_to_state(0,0)
    for i in range(100):
        dist, _ = model(dq_state)
        state,new_profit,cr = sims.step(dist.sample().item())
        dq_state = convert_to_state(state,new_profit)
        total_reward += cr
        if VISUALIZE:
            sims1.draw()
            # w = []
            # wparameters = list(model.actor.parameters())
            # for p in range(0, len(model.actor.activations)):
            #     if p * 2 < len(wparameters):
            #         w.append(wparameters[p * 2])
            #     w.append(model.actor.activations[p])
            #
            # sims.plot_weights(w)
            py.display.update()
            sims1.clock.tick(60)
    return total_reward

test_rewards = []
sims.reset()

state = 0
profit = 0
new_profit = 0
balance = 0


for epoch in range(100):
    if epoch == 50:
        sims.reset()
    dq_state = convert_to_state(state,new_profit)
    # all_states = torch.tensor(dq_state.view(1, -1))
    # all_outputs = torch.tensor(predicted)
    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = 0
    total_reward = 0

    bs = 0
    ss = 0
    cs = 0
    ns = 0


    for _ in range(100):
        # print(_)
        done = 0
        if _ == 19:
            done = 1
        # print(dq_state)
        dist,value = model(dq_state)
        # print(dist)

        action = dist.sample()
        if action == 0:
            ns+=1
        elif action == 1:
            bs += 1
        elif action == 2:
            ss+=1
        elif action == 3:
            cs += 1
        state,new_profit,current_reward = sims.step(action)
        total_reward += current_reward
        next_state = convert_to_state(state,new_profit)

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()

        log_probs.append(log_prob)
        values.append(value)

        current_reward = current_reward.type(torch.FloatTensor)
        rewards.append(current_reward.unsqueeze(1).to(device))
        # tg = torch.FloatTensor(1 - done).unsqueeze(1)

        if not done:
            tg = torch.tensor([[0]])
        else:
            tg = torch.tensor([[1]])

        # print(tg,done)
        masks.append(tg)
        dq_state = next_state
        if VISUALIZE:
            sims.draw()
            # w = []
            # wparameters = list(model.actor.parameters())
            # for p in range(0, len(model.actor.activations)):
            #     if p * 2 < len(wparameters):
            #         w.append(wparameters[p * 2])
            #     w.append(model.actor.activations[p])
            #
            # sims.plot_weights(w)
            py.display.update()
            sims.clock.tick(60)

    print("No %d Buy %d Sell %d Close %d"%(ns,bs,ss,cs))
    test_rewards.append(np.mean([test_env() for _ in range(5)]))

    plot(_*(epoch+1), test_rewards,save_plot_name)

    _, next_value = model(next_state)

    returns = compute_returns(next_value, rewards, masks)
    # print("Log",log_probs)
    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)

    advantage = returns - values

    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()
    print(actor_loss,critic_loss)
    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()