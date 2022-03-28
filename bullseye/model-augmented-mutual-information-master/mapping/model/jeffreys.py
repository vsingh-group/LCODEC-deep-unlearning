""" Jeffreys divergences """
import torch


def jeffreys_normal(mu1, lv1, mu2, lv2):
    mu1, lv1 = mu1.view(mu1.shape[0], -1), lv1.view(lv1.shape[0], -1)
    mu2, lv2 = mu2.view(mu2.shape[0], -1), lv2.view(lv2.shape[0], -1)
    return (0.25*((-lv1).exp() + (-lv2).exp())*(mu1-mu2)**2 + 0.25*((lv1-lv2).exp() + (lv2-lv1).exp()) - 0.5).sum(dim=1)


def jeffreys_bernoulli(p, q, eps=1e-5):
    p = p.view(p.shape[0], -1)*(1.-2.*eps)+eps
    q = q.view(q.shape[0], -1)*(1.-2.*eps)+eps
    return 0.5*(p-q)*(p.log() - q.log() - (1-p).log() + (1-q).log())
