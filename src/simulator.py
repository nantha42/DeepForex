import pandas as pd
import matplotlib.pyplot as plt
import tkinter
import numpy as np
from essential_functions import *
from mpl_finance import candlestick_ohlc

f = open("../Dataset/EURUSD30min.csv")
eu = pd.read_csv(f)

ohlc = []
g = eu
last_open = 0
count = 0
common = {}
for i in range(800):
    a = [i,g["Open"][i],g["High"][i],g["Low"][i],g["Close"][i],g["Volume"][i]]
    if last_open != g["Open"][i]:
        a[0] = count
        count+=1
        ohlc.append(a)
        last_open = g["Open"][i]
_Points = 1e-5

#lenght of bear = 14
bear = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
bull = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
avg = 0
ans = []
p = 10
vals = []
for i in range(0,len(bull)):
    vals.append(p)
    p = p*1.5
    print(vals)
not_added = 0
f = open("output.txt","w")

highs = []
vector = []
# avged_open = Expmovingaverage(g["Open"],10)
avged_open = g["Open"]
print(np.array(avged_open).shape)
# exit()
for i in range(1,len(avged_open)):
    v =  (avged_open[i] -avged_open[i-1])/(1*_Points)
    bu = bull.copy()
    be = bear.copy()
    p = 10
    added = False
    for i in range(0,len(bull)):
        if abs(v) < p:
            if v > 0: bu[i] = 1
            else: be[len(bear)-1-i] = 1
            a = be+bu
            vector.append(np.argmax(a)+1)
            f.writelines(str(a)+"\n")
            ans.append(a)
            added = True
            break
        p = p*1.5
    if(added==False):
        not_added +=1
        highs.append(v)
f.close()
f = open("../Dataset/EURUSD30min2015-17.csv")
eu = pd.read_csv(f)

ohlc = []
g = eu
avged_open = g["Open"]
print(np.array(avged_open).shape)
# exit()
for i in range(1,len(avged_open)):
    v =  (avged_open[i] -avged_open[i-1])/(1*_Points)
    bu = bull.copy()
    be = bear.copy()
    p = 10
    added = False
    for i in range(0,len(bull)):
        if abs(v) < p:
            if v > 0: bu[i] = 1
            else: be[len(bear)-1-i] = 1
            a = be+bu
            vector.append(np.argmax(a)+1)
            # f.writelines(str(a)+"\n")
            ans.append(a)
            added = True
            break
        p = p*1.5
    if(added==False):
        not_added +=1
        highs.append(v)
        # vector.append(28)
# print(vector[:100])
print("Set",set(vector))
v=np.array(vector)
plot(vector)
v = v.reshape((len(vector),1))
print(v.shape)
np.save("../dataset/Eavg_open.npy",v)
print("Very hight value",not_added)
print(highs)
fig,ax = plt.subplots()
candlestick_ohlc(ax,ohlc)

# plt.show()
