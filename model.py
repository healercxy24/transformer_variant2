# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:49:16 2022

@author: njucx
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import math
import copy
from data_process import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda')

#%% Transformer
        
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence.
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        print('postion', position.size())
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]
        print('pe', pe.size())
        self.register_buffer('pe', pe)


    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> x = torch.randn(50, 128, 18)
            >>> pos_encoder = PositionalEncoding(18, 0.1)
            >>> output = pos_encoder(x)
        """
        
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)                     


class Transformer(nn.Module):
    r"""
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        nlayers: the number of sublayers of both encoder and decoder
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    """

    def __init__(self, d_model, nhead, dim_feedforward, en_nlayers, de_nlayers, de_layer_size, dropout) -> None:
        super(Transformer, self).__init__()

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, norm_first=True)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, en_nlayers, encoder_norm)
        
        self.layers = []       
        self.layers.append(nn.Linear(d_model, de_layer_size))
        #for i in range(1, de_nlayers-1):
            #self.layers.append(nn.Linear(de_layer_size, de_layer_size))
        self.layers.append(nn.Linear(de_layer_size, 1))
        
        self.layers = nn.ModuleList(self.layers)

        self.d_model = d_model
        self.nhead = nhead
        
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()
    
    def init_weights(self):
        initrange = 0.1
        nn.init.xavier_uniform_(self.encoder.weight, -initrange, initrange)
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, src, src_mask) -> Tensor:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).

        Shape:
            - src: :(S, N, E)

            - output: :(T, N, E)

            S is the source sequence length, 
            T is the target sequence length, 
            N is the batch size, 
            E is the feature number
        """

        if src.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src must be equal to d_model")
        
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=src_mask)
        #output = self.activation(memory)
        
        for layer in self.layers:
            output = self.dropout(self.activation(layer(output)))
        return output


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss