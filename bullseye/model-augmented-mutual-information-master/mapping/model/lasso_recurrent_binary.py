import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict

from ..base import BaseModel
from .input_weighting import *


class LassoRNNBinary(BaseModel):
    """
    Data has dimension (N, T, M)
        - N samples
        - T dimensions for each feature
        - M features
    """
    def __init__(self, num_features, rnn_states, rnn_layers, decoder_layers, rnn_type='GRU', rnn_drop=0.0, decoder_drop=0.0):
        super().__init__()
        assert rnn_type in ['GRU', 'LSTM', 'RNN']
        self.num_features = num_features

        self.input_weights = InputWeighting(num_features)
        self.rnn = getattr(nn, rnn_type)(num_features, rnn_states, num_layers=rnn_layers, dropout=rnn_drop, batch_first=True)

        self.out = nn.Sequential(OrderedDict([
            ('lin0', nn.Linear(rnn_states, decoder_layers)),
            ('elu0', nn.ReLU()),
            ('drop0', nn.Dropout(decoder_drop)),
            ('lin1', nn.Linear(decoder_layers, 1)),
        ]))


    def forward(self, x):
        _,h = self.rnn(self.input_weights(x))
        return torch.sigmoid(self.out(h[-1,...]))
