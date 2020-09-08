import torch
import torch.nn as nn
import numpy as np
import logging
import inspect 
import math
from enum import IntEnum 


logger = logging.getLogger("tensor_shapes")
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(1)

def getclass():
    stack = inspect.stack()
    return stack[3][0].f_locals["self"].__class__

def log_size(tsr: torch.Tensor, name: str):
    cls = getclass()
    logger.log(level=cls.level, msg=f"[{cls.__name__}] {name} size={tsr.shape}")

class TensorLoggingLevels(IntEnum):
    attention = 1
    attention_head = 2
    multihead_attention_block = 3
    enc_dec_block = 4
    enc_dec = 5

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class ScaledDotProductAttention(nn.Module):
    # level = TensorLoggingLevels.attention # Logging level: 
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = k.size(-1) # get the size of the key
        assert q.size(-1) == d_k
        # print(q.shape,k.shape,v.shape)
        attn = torch.bmm(q, k.transpose(Dim.seq, Dim.feature))
        attn = attn / math.sqrt(d_k)
        attn = torch.exp(attn)
        # log_size(attn, "attention weight") 
        
        if mask is not None: attn = attn.masked_fill(mask, 0)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        # log_size(output, "attention output size") 
        return output

class AttentionHead(nn.Module):
    # level = TensorLoggingLevels.attention_head
    def __init__(self, d_model, d_feature, dropout=0.1):
        super().__init__()
        # We will assume the queries, keys, and values all have the same feature size
        self.attn = ScaledDotProductAttention(dropout)
        self.query_tfm = nn.Linear(d_model, d_feature)
        self.key_tfm = nn.Linear(d_model, d_feature)
        self.value_tfm = nn.Linear(d_model, d_feature)

    def forward(self, queries, keys, values, mask=None):
        Q = self.query_tfm(queries) # (Batch, Seq, Feature)
        K = self.key_tfm(keys) # (Batch, Seq, Feature)
        V = self.value_tfm(values) # (Batch, Seq, Feature)
        # log_size(Q, "queries, keys, vals")
        # compute multiple attention weighted sums
        x = self.attn(Q, K, V)
        return x

class MultiHeadAttention(nn.Module):
    # level = TensorLoggingLevels.multihead_attention_block
    def __init__(self, d_model, d_feature, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_feature = d_feature
        self.n_heads = n_heads
        # in practice, d_model == d_feature * n_heads
        # print(d_model,d_feature,n_heads)
        assert d_model == d_feature * n_heads

        # Note that this is very inefficient:
        # I am merely implementing the heads separately because it is 
        # easier to understand this way
        self.attn_heads = nn.ModuleList([
            AttentionHead(d_model, d_feature, dropout) for _ in range(n_heads)
        ])
        self.projection = nn.Linear(d_feature * n_heads, d_model) 
    
    def forward(self, queries, keys, values, mask=None):
        # log_size(queries, "Input queries")
        x = [attn(queries, keys, values, mask=mask) # (Batch, Seq, Feature)
             for i, attn in enumerate(self.attn_heads)]
        # log_size(x[0], "output of single head")
        
        # reconcatenate
        x = torch.cat(x, dim=Dim.feature) # (Batch, Seq, D_Feature * n_heads)
        # log_size(x, "concatenated output")
        x = self.projection(x) # (Batch, Seq, D_Model)
        # log_size(x, "projected output")
        return x

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class EncoderBlock(nn.Module):
    # level = TensorLoggingLevels.enc_dec_block
    def __init__(self, d_model=512, d_feature=64,
                 d_ff=2048, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout)
        self.layer_norm1 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm2 = LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # log_size(x, "Encoder block input")
        att = self.attn_head(x, x, x, mask=mask)
        # log_size(x, "Attention output")
        # Apply normalization and residual connection
        x = x + self.dropout(self.layer_norm1(att))
        # Apply position-wise feedforward network
        pos = self.position_wise_feed_forward(x)
        # log_size(x, "Feedforward output")
        # Apply normalization and residual connection
        x = x + self.dropout(self.layer_norm2(pos))
        # log_size(x, "Encoder size output")
        return x

class TransformerEncoder(nn.Module):
    level = TensorLoggingLevels.enc_dec
    def __init__(self, n_blocks=6, d_model=512,
                 n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoders = nn.ModuleList([
            EncoderBlock(d_model=d_model, d_feature=d_model // n_heads,
                         d_ff=d_ff, dropout=dropout,n_heads=n_heads)
            for _ in range(n_blocks)
        ])
    
    def forward(self, x: torch.FloatTensor, mask=None):
        for encoder in self.encoders:
            x = encoder(x)
        return x

class DecoderBlock(nn.Module):
    # level = TensorLoggingLevels.enc_dec_block
    def __init__(self, d_model=512, d_feature=64,
                 d_ff=2048, n_heads=8, dropout=0.1):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout)
        self.attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout)
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.layer_norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, 
                src_mask=None, tgt_mask=None):
        # Apply attention to inputs
        att = self.masked_attn_head(x, x, x, mask=src_mask)
        x = x + self.dropout(self.layer_norm1(att))
        # Apply attention to the encoder outputs and outputs of the previous layer
        att = self.attn_head(queries=x, keys=enc_out, values=enc_out, mask=tgt_mask)
        x = x + self.dropout(self.layer_norm2(att))
        # Apply position-wise feedforward network
        pos = self.position_wise_feed_forward(x)
        x = x + self.dropout(self.layer_norm2(pos))
        return x

class TransformerDecoder(nn.Module):
    # level = TensorLoggingLevels.enc_dec
    def __init__(self, n_blocks=6, d_model=512, d_feature=64,
                 d_ff=2048, n_heads=8, dropout=0.1):
        super().__init__()
        self.position_embedding = PositionalEmbedding(d_model)
        self.decoders = nn.ModuleList([
            DecoderBlock(d_model=d_model, d_feature=d_model // n_heads,
                         d_ff=d_ff, dropout=dropout,n_heads=n_heads)
            for _ in range(n_blocks)
        ])
        self.linear = nn.Linear(d_model,28)
        self.soft = nn.Softmax(dim=2)
        
    def forward(self, x: torch.FloatTensor, 
                enc_out: torch.FloatTensor, 
                src_mask=None, tgt_mask=None):
        for decoder in self.decoders:
            x = decoder(x, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
        # print(x.shape)
        y = self.linear(x)
        y = self.soft(y)
        # print(y.shape)
        return y

class PositionalEmbedding(nn.Module):
    level = 1
    def __init__(self, d_model, max_len=512):
        super().__init__()        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)
        
    def forward(self, x):
        return self.weight[:, :x.size(1), :] # (1, Seq, Feature)

class WordPositionEmbedding(nn.Module):
    level = 1
    def __init__(self, vocab_size, d_model=512):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        
    def forward(self, x: torch.LongTensor, mask=None) -> torch.FloatTensor:
        a = self.word_embedding(x)
        b = self.position_embedding(x)
        
        # print(a.shape,b.shape)
        return a + b


class TransformerModel(nn.Module):
    
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        # print(ninp,dropout)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)