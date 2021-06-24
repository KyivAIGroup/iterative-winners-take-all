"""
Let y1 = f(x1, W) and y2 = f(x2, W).
Learning the weights either for (x1, y1) or (x2, y2) should decorrelate y1 and y2 signals.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from mighty.loss import ContrastiveLossSampler
from mighty.utils.common import set_seed
from mighty.utils.data import DataLoader
from nn.kwta import *
from nn.trainer import TrainerIWTA
from nn.nn_utils import sample_bernoulli, NoShuffleLoader

set_seed(0)

N_x = N_y = N_h = 200
s_x = 0.1
s_w_xh = 0.05
s_w_xy = 0.05
s_w_hy = 0.1
s_w_yy = 0.01
s_w_hh = 0.1
s_w_yh = 0.05

WITH_PERMANENCE = True


class TrainerIWTADecorrelation(TrainerIWTA):
    pass


class RandomDataset(TensorDataset):
    def __init__(self, *args, **kwargs):
        labels = torch.arange(2)
        super().__init__(x12, labels)


x12 = sample_bernoulli((2, N_x), p=s_x)

if WITH_PERMANENCE:
    w_xy = ParameterWithPermanence(torch.rand(N_x, N_y), sparsity=s_w_xy, learn=False)
    w_xh = ParameterWithPermanence(torch.rand(N_x, N_h), sparsity=s_w_xh, learn=False)
    w_hy = ParameterWithPermanence(torch.rand(N_h, N_y), sparsity=s_w_hy)
    w_hh = ParameterWithPermanence(torch.rand(N_h, N_h), sparsity=s_w_hh, learn=False)
    w_yh = ParameterWithPermanence(torch.rand(N_y, N_h), sparsity=s_w_yh, learn=False)
    w_yy = ParameterWithPermanence(torch.rand(N_y, N_y), sparsity=s_w_yy, learn=False)
else:
    w_xy = nn.Parameter(sample_bernoulli((N_x, N_y), p=s_w_xy), requires_grad=False)
    w_xh = nn.Parameter(sample_bernoulli((N_x, N_h), p=s_w_xh), requires_grad=False)
    w_hy = nn.Parameter(sample_bernoulli((N_h, N_y), p=s_w_hy), requires_grad=False)
    w_hh = None
    w_yy = None
    w_yh = None

iwta = IterativeWTAInhSTDP(w_xy=w_xy, w_xh=w_xh, w_hy=w_hy, w_hh=w_hh, w_yy=w_yy, w_yh=w_yh)
# iwta = KWTANet(w_xy=w_xy, w_xh=w_xh, w_hy=w_hy, kh=10, ky=10)
print(iwta)

data_loader = DataLoader(RandomDataset, transform=None,
                         loader_cls=NoShuffleLoader)
criterion = ContrastiveLossSampler(nn.CosineEmbeddingLoss(margin=0),
                                   pairs_multiplier=5)
trainer = TrainerIWTADecorrelation(model=iwta, criterion=criterion,
                                   data_loader=data_loader, verbosity=1)
# trainer.monitor.advanced_monitoring(level=MonitorLevel.FULL)
iwta.set_monitor(trainer.monitor)
trainer.train(n_epochs=30)
