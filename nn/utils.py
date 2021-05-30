import torch


def sample_bernoulli(p, shape):
    x = torch.distributions.Bernoulli(p).sample(shape)
    return x.type(torch.int32)
