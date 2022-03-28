import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict

from ..base import BaseModel
from .delta_dropout import DeltaDropout
from .diagonal_linear import DiagonalLinear
from .jeffreys import *


class MappingFFGaussian(BaseModel):
    """
    Data has dimension (N, T, M)
        - N samples
        - T dimensions for each feature
        - M features
    """
    def __init__(self, xdim, ydim, num_features, dim_z_per, 
                 encoder_layers, decoder_layers, 
                 block_drop=0.0, delta=None, augment=True, encoder_drop=0.0, decoder_drop=0.0):
        super().__init__()
        self.dim_z_per = dim_z_per
        self.num_features = num_features
        self.p_drop = block_drop
        self.augment = augment
        self.delta = delta if delta is not None else num_features-1

        # x -> z
        self.encoders = nn.Sequential(OrderedDict([
            ('diaglin0', DiagonalLinear(num_features, xdim, encoder_layers)),
            ('elu0', nn.ELU()),
            ('drop0', nn.Dropout(encoder_drop)),
            ('diaglin2', DiagonalLinear(num_features, encoder_layers, dim_z_per)),
        ]))

        # block dropout
        # self.block_drop = BlockDropout(block_drop, force_nonzero=False)
        self.block_drop = DeltaDropout(self.delta)

        # z -> y
        self.decoder = nn.Sequential(OrderedDict([
            ('ffnn0', nn.Linear(num_features*(dim_z_per+1) if augment else num_features*dim_z_per, decoder_layers)),
            ('elu0', nn.ELU()),
            ('drop0', nn.Dropout(decoder_drop)),
        ]))

        self.mu = nn.Sequential(OrderedDict([
            ('ffnn1', nn.Linear(decoder_layers, ydim)),
        ]))

        self.lv = nn.Sequential(OrderedDict([
            ('ffnn1', nn.Linear(decoder_layers, ydim))
        ]))


    def _encode(self, x, apply_block_drop=True):
        z = self.encoders(x.permute(0,2,1).contiguous().view(x.shape[0],-1))
        return self.block_drop(z.view(z.shape[0], self.num_features, self.dim_z_per).permute(0,2,1), enabled=apply_block_drop)


    def _predict(self, z):
        # z = self.block_drop(z, enabled=apply_block_drop).contiguous()
        if self.augment:
            w = Variable(z.new(z.shape[0], self.num_features))
            w[:] = torch.all(z != 0, dim=1).detach()

            z = torch.cat((z, w.reshape(w.shape[0], 1, -1)), dim=1)

        h = self.decoder(z.reshape(z.shape[0],-1))
        return self.mu(h),self.lv(h)


    def divergence(self, z1, z2):
        mu1,lv1 = self._predict(z1)
        mu2,lv2 = self._predict(z2)
        # mu1,lv1 = self._predict(z1/(1-self.p_drop))
        # mu2,lv2 = self._predict(z2/(1-self.p_drop))
        return jeffreys_normal(mu1, lv1, mu2, lv2)


    def forward(self, x):
        z = self._encode(x, apply_block_drop=False)
        mu,lv = self._predict(z)
        return mu,lv,z
