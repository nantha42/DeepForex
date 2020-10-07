import pandas as pd
import matplotlib.pyplot as plt
import tkinter
import numpy as np
import torch
import time
from essential_functions import *
# from mpl_finance import candlestick_ohlc


def get_data(files,ptype):
    data = []
    for file in files:
        f = open(file)
        pair = pd.read_csv(f)
        data.extend(list(pair[ptype]))
        f.close()
    return data

    
if __name == '__main__':
    open_data = get_data(["../Dataset/EURUSD30min.csv","../Dataset/EURUSD30min2015-17.csv","../Dataset/EURUSD30min2018-20.csv"],"Volume")
    seq_len = 128+16

    low = 0
    high = 0 
    start = 0.0
    hp = 0.0
    lp = 0.0
    count = 0
    avg = 0.0
    out = set()

    vol = 0
    hvol = 0
    mvol = 1e12
    for i in range(len(open_data)):
        if open_data[i] != 0:
            vol += open_data[i]
            hvol = max(hvol,open_data[i])
            mvol = min(mvol,open_data[i])
            count+=1
    print("Average",vol/count)
    print(hvol)
    print(mvol)
    5143046950.85835
    290900614868.1644
        379999.9952

    # for i in range(len(open_data)-seq_len):
    #     a = time.time()
    #     segment = open_data[i:i+seq_len ]
    #     s = torch.tensor(segment).view(-1)
    #     s = ((s- s[0])*1e5)//50 + 120
    #     sett = set(s.tolist())
    #     out = out.union(sett)
    #     if i % 10000==0:
    #         print(i)
    #     print(time.time()-a)
    #     break;

    t = time.time()


    # for i in range(len(open_data)-seq_len):
    #     segment = open_data[i:i+seq_len]
    #     for j in range(1,len(segment)):
    #         low = min(low,segment[j]-segment[0])
    #         high = max(high,segment[j]-segment[0])
    #         avg += abs(segment[j] - segment[j-1])
    #         count+=1
        

    # print("Total Time ",time.time()-t)
    # print(low,high)
    # low = low*1e5
    # high = high*1e5
    # # low,high,avg = -5362.9999999999845,5936.999999999993,48.78423306203581
    # low = abs(low)
    # print(low//50,high//50)

    # print(low*1e5,high*1e5,(avg/count)*1e5 ) 
    # low = 
    # print( avg/count )

    # print( abs(low*1e5)  , high*1e5)
    # raw = np.array(open_data)
    # raw = raw.reshape(-1,1)
    # f = open("../preprocessed/raw_open.npy","wb")
    # np.save(f,raw,)
    # # print(raw.shape)

    # import time
    # h = 6000
    # l = 6000
    # t1 = time.time()
    # test = torch.tensor(open_data[:128])
    # test =  ((test-test[0])*1e5)//50 + 120
    # dt = time.time()-t1
    # print(dt,dt*(100000))


    # print( (h+l)//50)
    # print((240/2)*50)



    # open_data = get_data(["../Dataset/EURUSD30min.csv","../Dataset/EURUSD30min2015-17.csv","../Dataset/EURUSD30min2018-20.csv"],"High")

    # print(len(open_data))
    # print(min(open_data),max(open_data))

    # low = int(min(open_data)*1e5)
    # high = int(max(open_data)*1e5)
    # l = 2000
    # h = 45000
    # print("Number of tokens",(h-l)//50)

    # transformed = []
    # for i in open_data:
    #     x = int( (i-1)*1e5 )-l
    #     transformed.append(x//50+1)

    # print(set(transformed))
    # ndarr = np.array(transformed)
    # ndarr = ndarr.reshape(len(transformed),1)
    # print(ndarr.shape)
    # # np.save("../dataset/rangedToken.npy",ndarr)

    # plot(transformed)