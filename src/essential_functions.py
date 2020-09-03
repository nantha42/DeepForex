import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_data(filename,barType):
    folder = "../Dataset/"
    major = pd.read_csv(open(folder+filename))
    return major[barType].to_numpy()

def movingaverage(data,window_size):
    weights = np.repeat(1.0,window_size)/window_size
    smas = np.convolve(data,weights,'valid')
    return smas

def Expmovingaverage(data,window_size):
    weights = np.exp(np.linspace(-1.,0.,window_size))
    weights/= weights.sum()
    a = np.convolve(data,weights)[:len(data)]
    a[:window_size] = a[window_size]
    return a

def plot(data):
    fig,ax = plt.subplots()
    plt.plot(data)
    plt.show()

def plot_multiple(data):
    fig,ax = plt.subplots()
    for line in data:
        print(line)
        plt.plot(list(range(len(line))),line)
    plt.legend([str(i) for i in range(len(data))])
    plt.show()

if __name__ == '__main__':
    sample_size = 10000
    window_size = 15
    data = get_data("EURUSD30min.csv","Open")[:sample_size]
    mvg = movingaverage(data,window_size)
    expmvg = Expmovingaverage(data,100)
    
    high_data = get_data("EURUSD30min.csv","High")[:sample_size]
    high_mvg = movingaverage(high_data,window_size)

    low_data = get_data("EURUSD30min.csv","Low")[:sample_size]
    low_mvg = movingaverage(low_data,window_size)

    plot_multiple( [data,mvg,high_mvg,low_mvg,expmvg])