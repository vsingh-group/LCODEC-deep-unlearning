import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict

from ..base import BaseModel
from .jeffreys import *
from .delta_dropout import DeltaDropout

        
class MappingRNNBinary(BaseModel):
    """
    Data has dimension (N, T, M)
        - N samples
        - M features
        - T dimensions for each feature

    Embedding has dimension (N, dim_z_per, num_features=M)

    self.encoders is module list, e.g.
        # self.encoders = nn.ModuleList([
        #     nn.Linear(xdim, zdim)
        #     for i in range(num_features)
        # ])
    """
    def __init__(self, num_features, dim_z_per, 
                 rnn_states, rnn_layers, decoder_layers, 
                 rnn_type='GRU', block_drop=0.0, delta=None, augment=True, rnn_drop=0.0, decoder_drop=0.0, z_offest=2.0):
        super().__init__()
        assert rnn_type in ['GRU', 'LSTM', 'RNN']

        self.dim_z_per = dim_z_per
        self.num_features = num_features
        self.dim_z_total = num_features*(dim_z_per+1) if augment else num_features*dim_z_per
        self.p_drop = block_drop
        self.delta = delta if delta is not None else num_features-1
        self.augment = augment

        # x -> z
        self.encoders_rnn = nn.ModuleList([
            getattr(nn, rnn_type)(1, rnn_states, num_layers=rnn_layers, dropout=rnn_drop, batch_first=True)
            for i in range(num_features)])

        self.encoders_lin = nn.ModuleList([
            nn.Linear(rnn_states, dim_z_per)
            for i in range(num_features)])

        # block dropout
        # self.block_drop = BlockDropout(block_drop, force_nonzero=False)
        self.block_drop = DeltaDropout(self.delta)

        # z -> y
        self.decoder = nn.Sequential(OrderedDict([
            ('ffnn0', nn.Linear(self.dim_z_total, decoder_layers)),
            ('elu1', nn.ReLU()),
            ('drop0', nn.Dropout(decoder_drop)),
            ('ffnn1', nn.Linear(decoder_layers, 1)),
            ('sigmoid0', nn.Sigmoid())
        ]))


    def _encode(self, x, apply_block_drop=True):
        z = Variable(x.new(x.shape[0], self.dim_z_per, self.num_features))
        for i in range(self.num_features):
            _,h = self.encoders_rnn[i](x[:,:,i].view(x.shape[0],x.shape[1],1))
            z[:,:,i] = self.encoders_lin[i](h[-1,...])
        return self.block_drop(z, enabled=apply_block_drop)


    def _predict(self, z):
        # return self.decoder(self.block_drop(z, enabled=apply_block_drop).reshape(z.shape[0],-1))
        # z = self.block_drop(z, enabled=apply_block_drop)
        if self.augment:
            w = Variable(z.new(z.shape[0], self.num_features))
            w[:] = torch.all(z != 0, dim=1).detach()

            z = torch.cat((z, w.reshape(w.shape[0], 1, -1)), dim=1)

        return self.decoder(z.reshape(z.shape[0],-1))


    def divergence(self, z1, z2, eps=1e-5):
        p = self._predict(z1)
        q = self._predict(z2)
        # p = self._predict(z1/(1-self.p_drop), apply_block_drop=False)
        # q = self._predict(z2/(1-self.p_drop), apply_block_drop=False)
        return jeffreys_bernoulli(p, q, eps=eps)


    def forward(self, x):
        z = self._encode(x, apply_block_drop=False)
        p = self._predict(z)
        return p,z
