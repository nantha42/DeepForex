from transformer import *
from essential_functions import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import ast 
from numpy import load
import torch.nn as nn
import time

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

bptt = 35
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 5 - i)
    data = source[i:i+seq_len]
    target = source[i+5:i+5+seq_len].view(-1)
    return data, target

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
    # print(train_data.shape)
    val_data = batchify(train_data,batch_size)
    test_data = batchify(train_data,batch_size)
    
    model = Transformer(n_blocks=2,d_model=32,n_heads=4,d_ff=64,dropout=0.2)
    criterion = nn.CrossEntropyLoss()
    lr =  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    embs = nn.Embedding(28,28)
    embs.weight.data = torch.eye(28)
    for epoch in range(100):
        count = 0
        cum_loss = 0
        for batch, i in enumerate(range(0, train_data.size(0) - 5, bptt)):
            data, targets = get_batch(train_data, i)
            # targets = embs(targets)
            output = model(data)
            output = output.view(-1,28)
            loss = criterion(output,targets)
            cum_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count+=1
            # break;
        print(epoch,"Loss: ",(cum_loss/count))

    
    