from transformer import *
from essential_functions import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import ast 
from numpy import load
import torch.nn as nn
import time

class Transformer(nn.Module):
    def __init__(self,n_blocks=2,d_model=64,n_heads=4,d_ff=256,dropout=0.2):
        super().__init__()
        self.emb = WordPositionEmbedding(vocab_size = 28,d_model=64)
        self.encoder = TransformerEncoder(n_blocks=2,d_model=64,n_heads=4,d_ff=256,dropout=0.2)
        self.decoder = TransformerDecoder(n_blocks=2,d_model=64,d_feature=16,n_heads=4,d_ff=256,dropout=0.2)
    
    def forward(self,x):
        g = self.emb(x[:-1])
        x = self.encoder(g)
        p = self.emb(x[1:])
        y = self.decoder(p, x[1:])
        return y;

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data


bptt = 35
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.0
    ntokens = 28
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = 28
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        log_interval = 10
        if batch % log_interval == 0 and batch > 0:
            # print(output[:10])
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

if __name__ == '__main__':
    data = []
    device = torch.device("cpu")
    procsd_data = load("../dataset/procesd.npy")
    train_data =torch.tensor(procsd_data)[:30000]
    val_data = torch.tensor(procsd_data)[30000:35000]
    test_data = torch.tensor(procsd_data)[35000:]
    
    batch_size = 32
    ntokens = 28
    train_data = batchify(train_data,batch_size)
    val_data = batchify(train_data,batch_size)
    test_data = batchify(train_data,batch_size)
    emsize = 8 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    # model = Transformer(n_blocks=2,d_model=64,n_heads=4,d_ff=256,dropout=0.2)
    criterion = nn.CrossEntropyLoss()
    lr = 0.05 # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float("inf")
    epochs = 10 # The number of epochs
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(model, val_data)
        # print('-' * 89)
        # print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        #     'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
        #                                 val_loss, math.exp(val_loss)))
        # print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        # scheduler.step()
    # input_data = torch.tensor(inps[:1000])
    # output_data = torch.tensor(outs[:1000])
    
    # train_ds = TensorDataset(input_data,output_data)
    # train_dl = DataLoader(train_ds,)

    # print(input_data.shape)
    