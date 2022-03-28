import torch


def mse_loss(output, target):
    """
    Inputs have shape: (N,d)
    """
    if output.dim() == 1 or target.dim() == 1:
        return torch.mean((output.view(-1) - target.view(-1))**2)
    else:
        return torch.mean(torch.sum((output - target)**2, dim=1))


def gaussian_loss(output, target, offset=10):
    """
    Inputs have shape: (N,1) or (N,1): scalar gaussian
    """
    target = target.view(-1)
    mu,logvar = output[0].view(-1),output[1].view(-1)
    return torch.mean(0.5*(-logvar).exp()*(mu-target)**2 + logvar) + offset


def bce_weighted(output, target, w1=1.0, w0=1.0, eps=1e-5):
    target = target.reshape(target.shape[0], -1)
    output = output.reshape(output.shape[0], -1)*(1.-2.*eps)+eps
    ce1 = -w1 * target * torch.log(output)
    ce0 = -w0 * (1.0-target) * torch.log(1.0-output)
    return (ce1 + ce0).mean()
