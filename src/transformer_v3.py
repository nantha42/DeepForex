import torch
import math
import numpy as np
import copy
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import ast
from numpy import load
import torch.nn as nn
import random
import time
import matplotlib.pyplot as plt

data_file = "../preprocessed/raw.npy"
save_at = "../models/"

save_model_name = "Mark-III"

def t2v(ta, f, out_features, w, b, w0, b0, arg=None):
    tau = ta.type(torch.FloatTensor)
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        #print(w.shape, t1.shape, b.shape)
        # print(tau.type(),w.type())
        # print(tau.shape,w.shape,b.shape)
        v1 = tau*w + b
    
    v2 = f(torch.matmul(tau, w0) + b0)
    #print(v1.shape)
    return torch.cat([v1, v2], 1)

class SineActivation(nn.Module):
    def __init__(self, in_features):
        super(SineActivation, self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, in_features))
        self.b0 = nn.parameter.Parameter(torch.randn(1, in_features))
        
        self.w = nn.parameter.Parameter(torch.randn(in_features, in_features))
        self.b = nn.parameter.Parameter(torch.randn(1, in_features))
        self.f = torch.sin

    def forward(self, ta):
        tau = ta.type(torch.FloatTensor)
        # print(tau.shape,self.w0.shape)
        
        v1 = torch.matmul(tau,self.w0) + self.b0
        v2 = self.f(torch.matmul(tau,self.w) + self.b)
        v1 = v1.view(tau.size(0),tau.size(1),1)
        v2 = v2.view(tau.size(0),tau.size(1),1)
        x = torch.cat((v1,v2),-1)
        return x

class Time2Vec(nn.Module):
    def __init__(self,seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.sineact = SineActivation(seq_len)
    
    def forward(self,x):
        emb = self.sineact(x)
        emb = emb.masked_fill(x.unsqueeze(-1)==0,0)
        return emb

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # print(vocab_size,d_model)
        self.embed = nn.Embedding(vocab_size+1, d_model,padding_idx=0)

    def forward(self, x):
        embedded = self.embed(x)
        # embedded = embedded.masked_fill(x.unsqueeze(-1)==0,0)
        return embedded
        
if __name__ == '__main__':
    n_tokens = 4
    emb = Embedder(vocab_size=n_tokens,d_model=64)
    a = torch.randint(n_tokens,(1,10))
    print(a)
    print(emb(a))
    