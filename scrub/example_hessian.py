import torch
from torch import nn
import pdb
from autograd_lib import autograd_lib
from collections import defaultdict

from attrdict import AttrDefault

from autograd_lib import util as u
from autograd_lib import autograd_lib
from data_utils import getDatasets


def simple_model(d, num_layers):
    """Creates simple linear neural network initialized to identity"""
    layers = []
    for i in range(num_layers):
        layer = nn.Linear(d, d, bias=False)
        layer.weight.data.copy_(torch.eye(d))
        layers.append(layer)
    return torch.nn.Sequential(*layers)


def least_squares(data, targets=None):
    """Least squares loss (like MSELoss, but an extra 1/2 factor."""
    if targets is None:
        targets = torch.zeros_like(data)
    err = data - targets.view(-1, data.shape[1])
    return torch.sum(err * err) / 2 / len(data)



model: u.SimpleMLP = u.SimpleMLP([784, 20, 10], nonlin=True, bias=False)
# model.layers[0].weight.data.copy_(torch.eye(2))
autograd_lib.register(model)
loss_fn = torch.nn.CrossEntropyLoss()


full_dataset, val_dataset = getDatasets(name='mnist', data_augment=False)

data = full_dataset[0][0]
targets = torch.tensor([full_dataset[0][1]])
# data = u.to_logits(torch.tensor([[0.7, 0.3]]))
# targets = torch.tensor([0])
# data = data.repeat([3, 1])
# targets = targets.repeat([3])

n = 1

activations = {}
KFAC_hessians = defaultdict(lambda: AttrDefault(float))

for i in range(n):
    def save_activations(layer, A, _):
        activations[layer] = A
        KFAC_hessians[layer].AA += torch.einsum("ni,nj->ij", A, A)

    with autograd_lib.module_hook(save_activations):
        data_batch = data[i: i+1]
        targets_batch = targets[i: i+1]
        Y = model(data_batch)
        loss = loss_fn(Y, targets_batch)

    def compute_hess(layer, _, B):
        KFAC_hessians[layer].BB += torch.einsum("ni,nj->ij", B, B)

    with autograd_lib.module_hook(compute_hess):
        autograd_lib.backward_hessian(Y, loss='CrossEntropy', retain_graph=True)

num_layers = len(model.layers)
print("Number of layers in MLP: ", num_layers)
pdb.set_trace()

for i in range(num_layers):
    hess_layer = KFAC_hessians[model.layers[i]]
    hess = torch.einsum('kl,ij->kilj', hess_layer.BB / n, hess_layer.AA / n)
    

d=1
n=1
model = simple_model(1, 5)
data = torch.ones((n, d))
targets = torch.ones((n, d))
loss_fn = least_squares

autograd_lib.register(model)

hess = defaultdict(float)
hess_diag = defaultdict(float)
hess_kfac = defaultdict(lambda: AttrDefault(float))

activations = {}
def save_activations(layer, A, _):
    activations[layer] = A

    # KFAC left factor
    hess_kfac[layer].AA += torch.einsum("ni,nj->ij", A, A)

with autograd_lib.module_hook(save_activations):
    output = model(data)
    loss = loss_fn(output, targets)

def compute_hess(layer, _, B):
    A = activations[layer]
    BA = torch.einsum("nl,ni->nli", B, A)

    # full Hessian
    hess[layer] += torch.einsum('nli,nkj->likj', BA, BA)

    # Hessian diagonal
    hess_diag[layer] += torch.einsum("ni,nj->ij", B * B, A * A)

    # KFAC right factor
    hess_kfac[layer].BB += torch.einsum("ni,nj->ij", B, B)


with autograd_lib.module_hook(compute_hess):
    autograd_lib.backward_hessian(output, loss='LeastSquares')

for layer in model.modules():
    print(hess[layer])

for layer in model.modules():
    print(hess_diag[layer])

for layer in model.modules():
    print(hess_kfac[layer])

pdb.set_trace()