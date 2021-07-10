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
from mighty.utils.domain import MonitorLevel
from nn.kwta import *
from nn.trainer import TrainerIWTA
from nn.nn_utils import sample_bernoulli, NoShuffleLoader

set_seed(0)

N_x = N_y = N_h = 200
s_x = 0.1
s_w_xh = s_w_xy = s_w_hy = s_w_yy = s_w_hh = s_w_yh = 0.05


class TrainerIWTADecorrelation(TrainerIWTA):
    N_CHOOSE = None
    LEARNING_RATE = 0.1
    pass


class RandomDataset(TensorDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(x12, labels)


x12 = sample_bernoulli((100, N_x), p=s_x)
labels = torch.arange(x12.size(0), device=x12.device)

Permanence = PermanenceFixedSparsity

w_xy = Permanence(sample_bernoulli((N_x, N_y), p=s_w_xy), learn=False)
w_xh = Permanence(sample_bernoulli((N_x, N_h), p=s_w_xh), learn=False)
w_hy = Permanence(sample_bernoulli((N_h, N_y), p=s_w_hy), learn=True)
w_hh = Permanence(sample_bernoulli((N_h, N_h), p=s_w_hy), learn=True)
w_yy = None
w_yh = None

iwta = IterativeWTA(w_xy=w_xy, w_xh=w_xh, w_hy=w_hy, w_hh=w_hh, w_yy=w_yy, w_yh=w_yh)
# iwta = KWTANet(w_xy=w_xy, w_xh=w_xh, w_hy=w_hy, kh=10, ky=10)
print(iwta)

data_loader = DataLoader(RandomDataset, transform=None,
                         loader_cls=NoShuffleLoader)
criterion = ContrastiveLossSampler(nn.CosineEmbeddingLoss(margin=0))
trainer = TrainerIWTADecorrelation(model=iwta, criterion=criterion,
                                   data_loader=data_loader, verbosity=1)
trainer.monitor.advanced_monitoring(level=MonitorLevel.SIGN_FLIPS | MonitorLevel.WEIGHT_HISTOGRAM)
trainer.train(n_epochs=50)
