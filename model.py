import torch
from torch import Tensor
import torch.nn as nn
#import torch.nn.functional as F
from typing import Optional,Tuple
import math


class PositionalEncoding(nn.Module):

    def __init__(self,embedding_dim:int,sentence_len:int=100):

        super().__init__()
        self.pe = torch.zeros((sentence_len,embedding_dim))

        for pos in range(0,sentence_len):
            for i in range(0,embedding_dim//2):
                self.pe[pos, 2*i] = math.sin(pos / math.pow(10000, 2 * i / embedding_dim))
                self.pe[pos, 2*i+1] = math.cos(pos / math.pow(10000, 2 * i / embedding_dim))

        self.register_buffer('positionalencoding',self.pe)


    def forward(self,x:Tensor)->Tensor:

        x = x + self.pe[:x.size(1),:].to(x)
        return x



class ScaledDotProductAttention(nn.Module):
    
    def __init__(self,dropout:float=0.2):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,
                query:Tensor,
                key:Tensor,
                value:Tensor,
                mask:Optional[Tensor]=None)->Tuple[Tensor,Tensor]:

        qk = torch.matmul(query,key.transpose(2,3))
        qk /= math.sqrt(key.shape[-1])

        if mask is not None:
            qk = qk.masked_fill(mask==0,1e-9)

        attention_weights = self.softmax(qk)
        out = torch.matmul(attention_weights,value)

        return out,attention_weights




class MultiHeadAttention(nn.Module):

    def __init__(self,heads:int,embedding_dim:int,dropout:float=0.2):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.heads = heads
        self.dk = embedding_dim // heads

        self.Q = nn.Linear(embedding_dim,embedding_dim)
        self.K = nn.Linear(embedding_dim,embedding_dim)
        self.V = nn.Linear(embedding_dim,embedding_dim)
        self.scores = ScaledDotProductAttention()
        self.dropout = nn.Dropout(p=dropout)
        self.ff = nn.Linear(embedding_dim,embedding_dim)

    def forward(self,
                query:Tensor,
                key:Tensor,
                value:Tensor,
                mask:Optional[Tensor]=None)->Tuple[Tensor,Tensor]:

        batch_size = query.size(0)

        q_out = self.Q(query).view(batch_size,-1,self.heads,self.dk).transpose(1,2)
        k_out = self.K(key).view(batch_size,-1,self.heads,self.dk).transpose(1,2)
        v_out = self.V(value).view(batch_size,-1,self.heads,self.dk).transpose(1,2)

        attn_mask = mask.unsqueeze(1).repeat(1,self.heads,1,1)

        scores,attn_prob = self.scores(q_out,k_out,v_out,attn_mask)
        scores = scores.transpose(1,2).contiguous().view(batch_size,-1,self.embedding_dim)
        out = self.ff(scores)
        out = self.dropout(out)

        return out,attn_prob



class FeedForward(nn.Module):
    
    def __init__(self,embedding_dim:int,dropout:float=0.2):

        super().__init__()

        self.inner_dim = 1024

        self.ff = nn.Sequential(
                nn.Linear(embedding_dim,self.inner_dim),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(self.inner_dim,embedding_dim),
                nn.Dropout(p=dropout),)


    def forward(self,x:Tensor)->Tensor:

        out = self.ff(x)
        return out




class DecoderLayer(nn.Module):

    def __init__(self,heads:int,embedding_dim:int,dropout:float=0.2):

        super().__init__()

        self.mha = MultiHeadAttention(heads,embedding_dim,dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.ffn = FeedForward(embedding_dim,dropout)



    def forward(self,x:Tensor,attn_mask:Tensor)->Tuple[Tensor,Tensor]:

        attn,attn_prob = self.mha(x,x,x,attn_mask)
        attn = self.dropout1(attn)
        attn = self.norm1(x + attn)

        out = self.ffn(attn)
        out = self.dropout2(out)
        out = self.norm2(attn + out)

        return out,attn_prob



class Decoder(nn.Module):

    def __init__(self,vocab_size:int,heads:int,embedding_dim:int,N:int,dropout:float=0.2):

        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.N = N
        stacked_decoder_layers = []

        self.token_embed = nn.Embedding(vocab_size,embedding_dim)
        self.positionalencoding = PositionalEncoding(embedding_dim)

        for _ in range(N):
            stacked_decoder_layers.append(DecoderLayer(heads,embedding_dim,dropout))

        self.decoder = nn.Sequential(*stacked_decoder_layers)


    def forward(self,x:Tensor):
        
        dec_out = self.token_embed(x) + self.positionalencoding(self.token_embed(x))

        dec_attn_pad_mask = self.create_padding_mask(x,x,0)
        dec_attn_decoder_mask = self.create_look_ahead_mask(x)

        dec_self_attn_mask = torch.gt((dec_attn_pad_mask+dec_attn_decoder_mask),0)

        self_attn_probs = []

        for _ in range(self.N):
            x_out,self_attn_prob = self.decoder[_](dec_out,dec_self_attn_mask)
            self_attn_probs.append(self_attn_prob)

        return x_out,self_attn_probs

    def create_padding_mask(self,q,k,pad):
        batch_size = q.size(0)
        q_len = q.size(1)
        k_len = k.size(1)
        mask = k.data.eq(pad).unsqueeze(1).expand(batch_size, q_len, k_len)
        return mask

    def create_look_ahead_mask(self,seq):
        look_ahead_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
        look_ahead_mask = look_ahead_mask.triu(diagonal=1) # upper triangular part of a matrix(2-D)
        return look_ahead_mask






class GPT2(nn.Module):

    def __init__(self,vocab_size,heads,embedding_dim,N,dropout=0.2):
        super().__init__()

        self.decoder = Decoder(vocab_size,heads,embedding_dim,N,dropout)
        self.projection = nn.Linear(embedding_dim,vocab_size,bias=False)

    def forward(self,x):

        decoded_x,dec_self_attn_probs = self.decoder(x)
        logits = self.projection(decoded_x)

        return  logits.contiguous(), dec_self_attn_probs

    
