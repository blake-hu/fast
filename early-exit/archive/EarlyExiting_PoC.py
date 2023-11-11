"""
Proof of Concept for an Early Exiting Transformers
Use Encoder-Decoder Architecture, trained on WikiText2
"""

import torch
import torch.nn as nn
import math
from copy import deepcopy

from util import *

class EarlyExitLayer(nn.Module):

    def __init__(self, d_model: int, nhead: int):
        super().__init__()

        
    def forward(self):
        return 0


class DecoderModel(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_hid: int, nlayers: int, ntoken: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Embedding(ntoken, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, d_hid, dropout)
        early_exiter = nn.Linear(in_features=d_model, out_features = 2)
        self.decoder = self._get_clones(decoder_layer, nlayers)
        self.classfiers = self._get_clones(early_exiter, nlayers)

        self.summation = nlayers*(nlayers+1)/2
        self.criterion = nn.CrossEntropyLoss()

    def _get_clones(self, layer, nlayers):
        return nn.ModuleList([layer for i in range(nlayers)])
    
    def _get_threshold(self, t: int, N: int):
        thresh = (0.9*0.9+ 0.1*math.exp(-2*t/N)) # lambda is 0.9
        return min(1, max(0, thresh))
            

    def _get_mask(self, len):
        mask = torch.tril(torch.ones(len, len))
        mask = mask.float().masked_fill_(mask == 0, float('-inf')).masked_fill_(mask == 1, float(0.0))


    def forward(self, x: Tensor, targets: Tensor, thresh: float = 0.9):
        total_loss = 0
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        src_mask = self._get_mask()
        for i in range(len(self.decoder)):
            out = self.decoder[i](x, src_mask)
            guess = self.classifier[i](out)
            guess = guess.view(-1, len(targets))
            total_loss += self.criterion(guess, targets) * (i/self.summation)

        return total_loss

