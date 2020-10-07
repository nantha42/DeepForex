import torch
import torch.nn as nn
import math
from transformer_v2 import *
import pygame as py
from simulator import *
from collections import namedtuple
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim



def predict_transformer(data):
    data = ((data-data[0])*1e5)//50 + 120 + 1
    data = data.type(torch.LongTensor)
    enc_inp,tar_inp = data[:64],data[64:]
    enc_inp = enc_inp.reshape(1,-1)
    tar_inp = tar_inp.reshape(1,-1)
    src_mask,tar_mask = create_masks(enc_inp,tar_inp)
    # print(src_mask,tar_mask)
    prediction_model.eval()
    output = prediction_model(enc_inp,tar_inp,src_mask,tar_mask)
    return output

sims = Simulator(True)
sims.set_data("../dataset/EURUSD30min2015-17.csv")
prediction_model = torch.load("../models/EvalMark-II-LearningRate-BenchMark-I")


num_episodes = 5
prev_output = None 



for episode in range(num_episodes):
    sims.reset()
    with torch.no_grad():
        output = predict_transformer(sims.window[0])
    predicted = torch.argmax(output,dim=-1)
    all_outputs = torch.tensor(predicted)
    for steps in range(30):
        
        with torch.no_grad():
            output = predict_transformer(sims.window[0])
        predicted = torch.argmax(output,dim=-1)
        
        all_outputs = torch.cat((all_outputs,predicted),dim=0)
    # print(all_outputs)
    if prev_output == None:
        prev_output = all_outputs
        print("All Outputs shape",all_outputs.shape)
    else:
        print("All Outputs shape",all_outputs.shape,prev_output.shape)
        print(prev_output==all_outputs)
        prev_output = all_outputs