import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DiagonalLinear(nn.Module):
    """
        keep track of nonzero parameters separately

        Flow:
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
    """
    def __init__(self, n_copies, ip_dim, op_dim, bias=True):
        super().__init__()
        self.n_copies, self.ip_dim, self.op_dim = n_copies, ip_dim, op_dim
        self.bias = bias

        # full weight matrices
        self.W1, self.b1 = None, None

        # define diagonal parameters
        self.w1_shape = (n_copies*op_dim, n_copies*ip_dim)
        self.weights1 = nn.ParameterList([
            nn.Parameter(torch.randn(op_dim, ip_dim))
            for i in range(n_copies)])

        if self.bias:
            self.b1_shape = (n_copies*op_dim,)
            self.bias1 = nn.ParameterList([
                nn.Parameter(torch.randn(op_dim))
                for i in range(n_copies)])

        # weight initialization
        for param in self.parameters():
            if param.dim() < 2:
                nn.init.constant_(param, 0)
            else:
                torch.nn.init.xavier_uniform_(param)


    def _build_weights(self):
        self.W1 = Variable(self.weights1[0].new(size=self.w1_shape))
        self.W1.fill_(0.0)

        if self.bias:
            self.b1 = Variable(self.bias1[0].new(size=self.b1_shape))
            self.b1.fill_(0.0)

        for i in range(self.n_copies):
            self.W1[i*self.op_dim:(i+1)*self.op_dim, i*self.ip_dim:(i+1)*self.ip_dim] = self.weights1[i]

            if self.bias:
                self.b1[i*self.op_dim:(i+1)*self.op_dim] = self.bias1[i]


    def forward(self, x):
        self._build_weights()
        return F.linear(x, self.W1, bias=self.b1) if self.bias else F.linear(x, self.W1)
