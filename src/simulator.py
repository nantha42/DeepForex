# import torch
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.finance import candlestick_ohlc
from mpl_finance import candlestick_ohlc

f = open("../Dataset/EURUSD5min.csv")
eu = pd.read_csv(f)
print(len(eu))
data = eu["High"][:20000]
data1 = eu["Low"][:20000]

ohlc = []
g = eu
last_open = 0
count = 0
for i in range(800):

    a = [i,g["Open"][i],g["High"][i],g["Low"][i],g["Close"][i],g["Volume"][i]]
    if last_open != g["Open"][i]:
        a[0] = count
        count+=1
        ohlc.append(a)
        last_open = g["Open"][i]

fig,ax = plt.subplots();
candlestick_ohlc(ax,ohlc)

# ax.plot(list(range(len(data))),data)
# ax.plot(list(range(len(data1))),data1)

plt.show()
print(data[0])



