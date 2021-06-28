import torch
import numpy as np

from mighty.utils.common import set_seed
from nn.kwta import *
from nn.nn_utils import sample_bernoulli
from randomness import check_randomness

set_seed(0)

N = 200
K_ACTIVE = 10
x = sample_bernoulli((100, N), p=0.1)
w = ParameterWithPermanence(torch.rand(N, N), sparsity=0.05)
check_randomness(w.numpy())

for epoch in range(10):
    y = KWTAFunction.apply(x @ w, K_ACTIVE)
    w.update(x_pre=x, x_post=y, alpha=1)
    check_randomness(w.numpy())
