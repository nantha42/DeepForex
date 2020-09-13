import pandas as pd
import matplotlib.pyplot as plt
import tkinter
import numpy as np
import torch
from essential_functions import *
from mpl_finance import candlestick_ohlc


def get_data(files,ptype):
    data = []
    for file in files:
        f = open(file)
        pair = pd.read_csv(f)
        data.extend(list(pair[ptype]))
        f.close()
    return data

open_data = get_data(["../Dataset/EURUSD30min.csv","../Dataset/EURUSD30min2015-17.csv","../Dataset/EURUSD30min2018-20.csv"],"High")

print(len(open_data))
print(min(open_data),max(open_data))

low = int(min(open_data)*1e5)
high = int(max(open_data)*1e5)
l = 2000
h = 45000
print("Number of tokens",(h-l)//50)

transformed = []
for i in open_data:
    x = int( (i-1)*1e5 )-l
    transformed.append(x//50+1)

print(set(transformed))
ndarr = np.array(transformed)
ndarr = ndarr.reshape(len(transformed),1)
print(ndarr.shape)
# np.save("../dataset/rangedToken.npy",ndarr)

plot(transformed)