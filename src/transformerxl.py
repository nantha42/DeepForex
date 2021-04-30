from typing import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")


class PositionalEmbedding(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        inv_freq = 1 / (10000 ** (torch.arange(0.0, d, 2.0) / d))
        # register buffer tells pytorch that this tensor is part of the modle
        # this means that it will be saved in the state_dict and moved to the GPU
        # along with the model
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, positions: torch.LongTensor, # (seq, )
               ):
        # outer product
        sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:,None,:]