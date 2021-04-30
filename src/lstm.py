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


data_file = "../preprocessed/raw_open.npy"
save_at = "../models/LSTM/"

save_model_name = "Mark-I-LSTM"

bptt = 72


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # print(vocab_size,d_model)
        self.embed = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)

    def forward(self, x):
        # print(x.shape)
        # print("Embed",self.embed(x).shape)

        embedded = self.embed(x)
        embedded = embedded.masked_fill(x.unsqueeze(-1) == 0, 0)
        return embedded

class CustomDataLoader:
    def __init__(self, source):

        self.batches = list(range(0, source.size(0) - (bptt + bptt // 8), 3))
        self.data = source
        self.sample = random.sample(self.batches, 120)

    def batchcount(self):
        return len(self.batches)

    def shuffle_batches(self):
        random.shuffle(self.batches)

    def get_batch_from_batches(self, i):
        if i == 0:
            random.shuffle(self.batches)
        ind = self.batches[i]
        inp_seq_len = bptt  + 1
        # tar_seq_len = min(int(bptt/8),len(self.data)-1-ind)
        s = time.time()
        sample = self.data[ind:ind + inp_seq_len]  # .view(-1).contiguous()
        sample = ((sample - sample[0]) * 1e5) // 50 + 120 + 1

        if torch.cuda.is_available():
            src = sample[:bptt].cuda()
            tar = sample[1:].cuda()
        else:
            src = sample[:bptt]
            tar = sample[1:]
        return src, tar


def get_batch(source, i):
    data = source[i:i + bptt + 1]
    data = ((data - data[0]) * 1e5) // 50 + 120 + 1
    src = data[:bptt].type(torch.LongTensor)
    tar = data[1:].type(torch.LongTensor)
    return src, tar

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data


def plot_multiple(data,legend):
    fig,ax = plt.subplots()
    for line in data:
        plt.plot(list(range(len(line))),line)
    plt.legend(legend)
    plt.show()


def plot_subplots(data,legends,name):

    plt.figure(figsize=(10, 5))
    for i in range(len(data)):
        plt.subplot(121+i)
        plt.plot(list(range(0,len(data[i])*3,3)),data[i])
        plt.title(legends[i])
        plt.xlabel("Epochs")
    plt.savefig("../Plots/" + name)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,vocab_size):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embed = Embedder(vocab_size, input_dim)

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax()

    def forward(self, x):

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        x = self.embed(x)
        # print("x",x.shape)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out)
        # print("out",out.shape)
        return self.softmax(out,dim=-1)


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.0
    ntokens = 240
    count = 0
    with torch.no_grad():
        cum_loss = 0
        acc_count = 0
        accs = 0
        # print(data_source.shape)
        for batch, i in enumerate(range(0, data_source.size(0) - bptt * 2, bptt)):
            data, targets = get_batch(data_source, i)
            # print(data.shape,targets.shape)
            if torch.cuda.is_available():
                data = data.transpose(0, 1).contiguous().type(torch.LongTensor).to(dev)
                targets = targets.transpose(0, 1).contiguous().type(torch.LongTensor).to(dev)
                trg_output = targets[:, -1:]
                trg_output = trg_output.view(-1)


            else:
                data = data.transpose(0, 1).contiguous().type(torch.LongTensor).to(dev)
                targets = targets.transpose(0, 1).contiguous().type(torch.LongTensor).to(dev)
                trg_output = targets[:, -1:]
                trg_output = trg_output.view(-1)

            # print(src_mask.device,trg_mask.device)
            output = model(data)
            output = output.view(-1, output.size(-1))
            loss = torch.nn.functional.cross_entropy(output, trg_output - 1)
            accs += ((torch.argmax(output, dim=1) == (trg_output - 1)).sum().item() / output.size(0))
            # accs += ((torch.argmax(output,dim=1)==targets).sum().item()/output.size(0))
            cum_loss += loss.item()
            count += 1
        # print(epoch,"Loss: ",(cum_loss/count),"Accuracy ",accs/count)

    return cum_loss / (count), accs / count

if __name__ == '__main__':

    procsd_data = load(data_file)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = torch.tensor(procsd_data)[:int(len(procsd_data) * 0.70)]
    val_data = torch.tensor(procsd_data)[int(len(procsd_data) * 0.70):int(len(procsd_data) * 0.90)]
    test_data = torch.tensor(procsd_data)[int(len(procsd_data) * 0.90):]

    train_data = train_data.contiguous()

    if torch.cuda.is_available():
        train_data = train_data.to(dev)
        val_data = val_data.to(dev)
        test_data = test_data.to(dev)

    batch_size = 32
    ntokens = 240

    train_data = batchify(train_data, batch_size)
    val_data = batchify(val_data, batch_size)
    test_data = batchify(train_data, batch_size)


    ntokens = 240

    model = LSTMModel(256,256,1,256,240)

    model.to(dev)
    criterion = nn.CrossEntropyLoss()

    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    accuracies = []
    lossies = []
    val_loss = []
    val_accuracy = []
    dataLoader = CustomDataLoader(train_data)
    dataLoader.get_batch_from_batches(0)

    for epoch in range(50):
        count = 0
        cum_loss = 0
        acc_count = 0
        accs = 0
        s = time.time()
        # for i in range(len(range(0, train_data.size(0) - bptt))):
        model.train()
        # dataLoader.shuffle_batches()
        for i in range(dataLoader.batchcount()):
            hh = time.time()
            data, targets = dataLoader.get_batch_from_batches(i)
            # print(data.device,targets.device)

            data = data.transpose(0, 1).contiguous().type(torch.LongTensor).to(dev)
            targets = targets.transpose(0, 1).contiguous().type(torch.LongTensor).to(dev)

            # print(data.shape,targets.shape)
            # print(data[0],"target",targets[0])
            trg_output = targets
            trg_output = trg_output.view(-1)
            # print(trg_output)

            output = model(data)
            # print(output.shape,trg_output.shape)
            output = output.view(-1,output.size(-1))
            loss = torch.nn.functional.cross_entropy(output, trg_output - 1)
            accuracy = ((torch.argmax(output, dim=1) == (trg_output - 1)).sum().item() / output.size(0))
            accs += accuracy
            cum_loss += loss.item()


            loss.backward()
            optim.step()
            model.zero_grad()
            optim.zero_grad()

            if i%10==0:
                time_takens = time.time()-hh
                time_takens = " Time taken %s"%(time_takens)
                epoc = "Epoch %s "%(epoch)
                print(epoc,i,"/",dataLoader.batchcount()," Batch Loss", loss.item()," Batch Accuracy ",accuracy,time_takens)
            count += 1

        data,targets = None,None
        if (epoch % 3 == 0):
            lossies.append(cum_loss / count)
            accuracies.append(accs / count)
            legend = ["Accuracy", "Loss"]
            plot_subplots([accuracies, lossies], legend, save_model_name + " Training")
            eval_loss, eval_acc = evaluate(model, val_data)

            print(epoch, "Loss: ", (cum_loss / count), "Accuracy ", accs / count, " Valid_loss: ", eval_loss," Valid_accuracy: ", eval_acc)

            if len(val_loss) > 0 and eval_loss < val_loss[-1]:
                val_accuracy.append(eval_acc)
                val_loss.append(eval_loss)
                torch.save(model, save_at+ "Eval" + save_model_name)

            elif len(val_loss) == 0:
                val_accuracy.append(eval_acc)
                val_loss.append(eval_loss)
                torch.save(model, save_at + "Eval" + save_model_name)
            plot_subplots([val_accuracy, val_loss], legend, save_model_name + " Validation")
        print("Loss: ", loss, "Accuracy: ", accuracy)
