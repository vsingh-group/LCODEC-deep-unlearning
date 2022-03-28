import numpy as np
import time
import torch

import random

from codec import codec2 as scikit_codec2
from codec import codec3 as scikit_codec3
from torch_codec import codec2 as torch_codec2
from torch_codec import codec3 as torch_codec3

seed = 12345
print("Setting seeds to: ", seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

n = 10000
p = 2
X = np.random.rand(n,p).astype(np.float16)
print(X[0])
Y = np.mod((X[:,0] + X[:,1]), 1.0).astype(np.float16)


print('Continuous:')
print('\tCODEC2 X1 to Y: ', scikit_codec2(X[:,1], Y))
print('\tCODEC2 X1 and X2 to Y: ', scikit_codec2(X, Y))
print('\tCODEC3 X1 to Y Given X2: ', scikit_codec3(X[:,0], Y, X[:,1]))

Xb = np.random.binomial(size=n, n=1, p=0.5).astype(np.float16)
Zb = np.random.binomial(size=n, n=1, p=0.5).astype(np.float16)
Yb = Xb

Xn = Xb + 0.01*np.random.normal(size=(n)).astype(np.float16)
Yn = Yb + 0.01*np.random.normal(size=(n)).astype(np.float16)
Zn = Zb + 0.01*np.random.normal(size=(n)).astype(np.float16)

print('Binary:')
print('\tCODEC3 X1 to Y Given X2: ', scikit_codec3(Xb,Yb,Zb))
print('Randomized:')
print('\tCODEC3 X1 to Y Given X2: ', scikit_codec3(Xn,Yn,Zn))

X = torch.Tensor(X)
print(X[0])
Y = torch.Tensor(Y)

Xb = torch.Tensor(Xb)
Yb = torch.Tensor(Yb)
Zb = torch.Tensor(Zb)

Xn = torch.Tensor(Xn)
Yn = torch.Tensor(Yn)
Zn = torch.Tensor(Zn)


print('Continuous:')
print('\tCODEC2 X1 to Y: ', torch_codec2(X[:,1], Y))
print('\tCODEC2 X1 and X2 to Y: ', torch_codec2(X, Y))
print('\tCODEC3 X1 to Y Given X2: ', torch_codec3(X[:,0], Y, X[:,1]))

print('Binary:')
print('\tCODEC3 X1 to Y Given X2: ', torch_codec3(Xb,Yb,Zb))
print('Randomized:')
print('\tCODEC3 X1 to Y Given X2: ', torch_codec3(Xn,Yn,Zn))
