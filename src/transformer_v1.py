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


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # print(vocab_size,d_model)
        self.embed = nn.Embedding(vocab_size+1, d_model,padding_idx=0)
    def forward(self, x):
        # print(x.shape)
        # print("Embed",self.embed(x).shape)
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 500):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len],requires_grad=False)
        return x

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        scores = torch.nn.functional.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(torch.nn.functional.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        # self.ff = FeedForward(d_model).cuda()
        self.ff = FeedForward(d_model)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
        
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data

bptt = 128
class CustomDataLoader:
    def __init__(self,source):
        print("Source",source.shape)
        self.batches = list(range(0, source.size(0) - 2*bptt))
        # random.shuffle(self.batches)
        # print(self.batches)
        self.data = source
        self.sample = random.sample(self.batches,120)

    def batchcount(self):
        return len(self.batches)

    def get_batch_from_batches(self,i):
        if i==0:
            random.shuffle(self.batches)
        ind = self.batches[i]
        seq_len = min(bptt,len(self.data)-1-ind)
        src = self.data[ind:ind+seq_len]
        tar = self.data[ind+seq_len-3:ind+seq_len-3+seq_len+1]
        return src,tar
        
    def get_batch(self,i):
        # print(i,len(self.batches))
        ind = self.sample[i]
        seq_len = min(bptt,len(self.data)-1-ind)
        src = self.data[ind:ind+seq_len]
        tar = self.data[ind+seq_len-3:ind+seq_len-3+seq_len+1]
        # tar = tar.view(-1)
        if(i==len(self.sample)-1):
            random.sample(self.batches,60)
            # print("Data shuffled",self.batches[:10])
        return src,tar

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+seq_len-3:i+seq_len-3+seq_len]
    return data, target

def plot_multiple(data,legend):
    fig,ax = plt.subplots()
    for line in data:
        plt.plot(list(range(len(line))),line)
    plt.legend(legend)
    plt.show()


def plot_subplots(data,legends,name):
    names = ['Accuracy', 'Loss']
    plt.figure(figsize=(10, 5))
    for i in range(len(data)):
        plt.subplot(121+i)
        plt.plot(list(range(0,len(data[i])*3,3)),data[i])
        plt.title(legends[i])
        plt.xlabel("Epochs")
    plt.savefig(name)

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = 860
    count = 0
    with torch.no_grad():
        cum_loss = 0
        acc_count = 0
        accs = 0
        print(data_source.shape)
        for batch, i in enumerate(range(0, data_source.size(0) - bptt*2, bptt)):
            data, targets = get_batch(data_source, i)
            # data,targets = dataLoader.get_batch(i)
            data = data.transpose(0,1).contiguous()
            targets= targets.transpose(0,1).contiguous()
            trg_input = targets[:,:-1]
            trg_output = targets[:,1:].contiguous().view(-1)
            src_mask , trg_mask = create_masks(data,trg_input)
            output = model(data,trg_input,src_mask,trg_mask)
            output = output.view(-1,output.size(-1))
            loss = torch.nn.functional.cross_entropy(output,trg_output-1)
            accs += ((torch.argmax(output,dim=1)==(trg_output-1) ).sum().item()/output.size(0))
            # accs += ((torch.argmax(output,dim=1)==targets).sum().item()/output.size(0))
            cum_loss += loss
            count+=1
        # print(epoch,"Loss: ",(cum_loss/count),"Accuracy ",accs/count)

    return cum_loss/ (count), accs/count

def nopeak_mask(size,cuda_enabled):
    np_mask = np.triu(np.ones((1, size, size)),
    k=1).astype('uint8')
    np_mask =  torch.autograd.Variable(torch.from_numpy(np_mask) == 0)

    if cuda_enabled:
      np_mask = np_mask.cuda()
    return np_mask

def create_masks(src, trg):
    src_mask = (src != 0).unsqueeze(-2)
    if trg is not None:
        trg_mask = (trg != 0).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        # print("Sequence lenght in mask ",size)
        np_mask = nopeak_mask(size,False)
        # print(np_mask.shape,trg_mask.shape)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


if __name__ == '__main__':
    data = []
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    procsd_data = load("../dataset/rangedToken.npy")
    train_data =torch.tensor(procsd_data)[:int(len(procsd_data)*0.70)]
    val_data = torch.tensor(procsd_data)[int(len(procsd_data)*0.70):int(len(procsd_data)*0.90)]
    test_data = torch.tensor(procsd_data)[int(len(procsd_data)*0.90):]
    train_data = train_data.to(dev)
    val_data = val_data.to(dev)
    test_data = test_data.to(dev)

    save_model_name = "../models/860Tokens"

    batch_size = 32
    ntokens = 860
    d_model = 64
    N = 3
    heads = 4
    
    
    train_data = batchify(train_data,batch_size)
    val_data = batchify(val_data,batch_size)
    test_data = batchify(train_data,batch_size)


    model = Transformer(ntokens,ntokens,d_model,N,heads)
    # model = torch.load("modela")

    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model.to(dev)
    
    optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    #########training starts###########

    accuracies = []
    lossies = []
    val_loss = []
    val_accuracy = []
    
    dataLoader = CustomDataLoader(train_data)
    
    
    
    for epoch in range(500):
        count = 0
        cum_loss = 0
        acc_count = 0
        accs = 0
        s = time.time()
        # for i in range(len(range(0, train_data.size(0) - bptt))):
        model.train()
        for i in range(dataLoader.batchcount()):
            hh = time.time()
            data,targets = dataLoader.get_batch_from_batches(i)
            data = data.transpose(0,1).contiguous()
            targets= targets.transpose(0,1).contiguous()
            trg_input = targets[:,:-1]
            trg_output = targets[:,1:].contiguous().view(-1)
            src_mask , trg_mask = create_masks(data,trg_input)
            output = model(data,trg_input,src_mask,trg_mask)
            
            output = output.view(-1,output.size(-1))

            loss = torch.nn.functional.cross_entropy(output,trg_output-1)
            accuracy = ((torch.argmax(output,dim=1)== (trg_output-1) ).sum().item()/output.size(0))
            accs += accuracy
            cum_loss += loss.item();
            loss.backward()
            optim.step()
            model.zero_grad()
            optim.zero_grad()
            if i%10==0:
                time_takens = time.time()-hh
                time_takens = " Time taken %s"%(time_takens)
                epoc = "Epoch %s "%(epoch)
                print(epoc,i,"/",dataLoader.batchcount()," Batch Loss", loss.item()," Batch Accuracy ",accuracy,time_takens)
            count+=1
            
        data,targets = None,None
        print(epoch,"Loss: ",(cum_loss/count),"Accuracy ",accs/count," Time Taken: ",time.time()-s)
        if(epoch%1==0):
            lossies.append(cum_loss/count)
            accuracies.append(accs/count)
            legend = ["accuracy","Loss"]
            plot_subplots([accuracies,lossies],legend,"A&L_v1")
            print("Valdata",val_data.shape)
            eval_loss,eval_acc = evaluate(model,val_data)
            val_accuracy.append(eval_acc)
            val_loss.append(eval_loss)
            plot_subplots([val_accuracy,val_loss],legend,"Val A&L_v1")
            print(epoch,"Loss: ",(cum_loss/count),"Accuracy ",accs/count," Valid_loss: ",eval_loss," Valid_accuracy: ",eval_acc)
            if len(val_loss)>0 and eval_loss < val_loss[-1]:
                val_loss.append(eval_loss)
                torch.save(model,save_model_name)
            else:
                val_loss.append(eval_loss)
                torch.save(model,save_model_name)
        if(epoch%5==0):
            torch.save(model,"Trained_model")