
#moving average data
from transformer import *
from essential_functions import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import ast 
from numpy import load
import torch.nn as nn
import time
import matplotlib.pyplot as plt

class Transformer(nn.Module):
    def __init__(self,n_blocks=2,d_model=64,n_heads=4,d_ff=256,dropout=0.2,vocab_size=28):
        super().__init__()
        self.emb = WordPositionEmbedding(vocab_size = vocab_size,d_model=d_model)
        self.decoder_emb = WordPositionEmbedding(vocab_size=vocab_size,d_model=d_model)
        self.encoder = TransformerEncoder(n_blocks=n_blocks,d_model=d_model,n_heads=n_heads,d_ff=d_ff,dropout=0.2)
        self.decoder = TransformerDecoder(n_blocks=n_blocks,d_model=d_model,d_feature=16,n_heads=n_heads,d_ff=d_ff,dropout=0.2)
    
    def forward(self,x):
        g = self.emb(x)
        encoded = self.encoder(g)
        p = self.decoder_emb(x)
        y = self.decoder(p, encoded)
        return y;

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    
    return data

bptt = 1024
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def plot_multiple(data,legend):
    fig,ax = plt.subplots()
    for line in data:
        print(line)
        plt.plot(list(range(len(line))),line)
    plt.legend(legend)
    plt.show()

def plot_subplots(data,legends):
    names = ['Accuracy', 'Loss']
    
    plt.figure(figsize=(10, 5))
    for i in range(len(data)):
        # plt.subplot(200+i)
        plt.subplot(121+i)
        plt.plot(list(range(0,len(data[i])*50,50)),data[i])
        plt.title(legends[i])
        plt.xlabel("Epochs")
    plt.show()

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = 28
    with torch.no_grad():
        
        count = 0
        cum_loss = 0
        acc_count = 0
        accs = 0
        for batch, i in enumerate(range(0, data_source.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i)
            # targets = embs(targets)
            output = model(data)
            output = output.view(-1,n_tokens)
            loss = criterion(output,targets)
            accs += ((torch.argmax(output,dim=1)==targets).sum().item()/output.size(0))
            cum_loss += loss
        print(epoch,"Loss: ",(cum_loss/count),"Accuracy ",accs/count)

    return cum_loss/ (len(data_source) - 1)

if __name__ == '__main__':
    data = []
    dev = torch.device("cuda")
    procsd_data = load("procesd.npy")
    train_data =torch.tensor(procsd_data)[:30000]
    print(train_data.shape)
    
    val_data = torch.tensor(procsd_data)[30000:35000]
    test_data = torch.tensor(procsd_data)[35000:]
    train_data = train_data.to(dev)
    val_data = val_data.to(dev)

    batch_size = 16
    ntokens = 28
    train_data = batchify(train_data,batch_size)
    # print(train_data.shape)
    val_data = batchify(train_data,batch_size)
    test_data = batchify(train_data,batch_size)
    # model = Transformer(n_blocks=3,d_model=64,n_heads=4,d_ff=64,dropout=0.2)
    model = torch.load("modelb1024")
    model.to(dev)
    
    criterion = nn.CrossEntropyLoss()
    lr = 0.5 # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    accuracies = []
    lossies = []
    val_loss = float(-inf)
    for epoch in range(10000):
        count = 0
        cum_loss = 0
        acc_count = 0
        accs = 0
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i)
            # targets = embs(targets)
            output = model(data)
            output = output.view(-1,28)
            loss = criterion(output,targets)
            # print(output.shape)
            # print(torch.argmax(output,dim=1).shape)
            accs += ((torch.argmax(output,dim=1)==targets).sum().item()/output.size(0))
            cum_loss += loss
            loss.backward()
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()
            count+=1
        
        if(epoch%50==1):
            lossies.append(cum_loss/count)
            accuracies.append(accs/count)
            print(epoch,"Loss: ",(cum_loss/count),"Accuracy ",accs/count)
            legend = ["accuracy","Loss"]
            plot_subplots([accuracies,lossies],legend)
            model.eval()

        if(epoch%200)==0:
            torch.save(model,"modela")

    